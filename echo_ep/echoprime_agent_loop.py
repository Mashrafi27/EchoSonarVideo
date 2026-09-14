"""Custom AgentLoop for the frozen-EchoPrime + Qwen3-8B-text GRPO track.

Not `SingleTurnAgentLoop` (verl's stock single-turn loop): that loop's image/video handling
(`AgentLoopBase.process_vision_info`/`apply_chat_template`, `AgentLoopWorker.
_compute_multi_modal_inputs`) is entirely gated on `self.processor is not None` -- our text-only
Qwen3-8B has no real combined HF processor at all (by design: our multimodal splice is a custom
precomputed-embedding path, not something a real AutoProcessor produces), so that whole code path
silently no-ops for us rather than erroring. Building a fake `ProcessorMixin`/`ImageProcessor`
pair just to satisfy that gate would be more code than this class, and less honest about what's
actually happening. This loop does the tokenize+generate steps directly instead.

`kwargs["extra_info"]` (confirmed available in `run()` the same way `tool_agent_loop.py` already
reads `kwargs["extra_info"]["interaction_kwargs"]`) carries `study_uuid` straight from the
parquet row -- no need to embed it into the prompt messages at all, unlike images that need
inline `<image>` markers in the stock dataset path.
"""
import os
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from echo_ep.modeling import VIEW_TOKEN

CACHE_DIR = os.environ.get("ECHOPRIME_VIDEO_CACHE_DIR", "build/echoprime_video_cache")
MAX_VIEWS = 50


@register("echoprime_agent")
class EchoPrimeAgentLoop(AgentLoopBase):
    """Single-turn, no tools -- matches EchoSonar-R's own (tool-less) recipe. Mirrors
    `SingleTurnAgentLoop.run()`'s structure, with steps 1-2 (image extraction, HF-processor
    chat templating) replaced by a direct tokenize + our own view-embeddings cache lookup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        study_uuid = kwargs["extra_info"]["study_uuid"]

        cache = torch.load(os.path.join(CACHE_DIR, f"{study_uuid}.pt"))
        view_embeddings = cache["embeddings"][:MAX_VIEWS]
        n_views = view_embeddings.shape[0]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        view_token_id = self.tokenizer.convert_tokens_to_ids(VIEW_TOKEN)
        actual_views = sum(1 for t in prompt_ids if t == view_token_id)
        assert actual_views == n_views, (
            f"study {study_uuid}: prompt has {actual_views} {VIEW_TOKEN!r} placeholders but the "
            f"cache has {n_views} views -- the parquet row's prompt text and the cache must "
            f"agree on view count (see echo_ep/generate_grpo_parquet.py)")

        # Ban VIEW_TOKEN from the model's own generated response: under the untrained cold-start
        # policy (near-uniform sampling, ~8.9 nats entropy), it can and does occasionally sample
        # this special token as ordinary output text. modeling.py's _splice_view_embeddings scans
        # the WHOLE prompt+response sequence for view_token_id with no way to tell "this one came
        # from the model's own response, not the prompt" -- a stray one downstream trips
        # "N view-token placeholders but M view embeddings" during compute_log_prob/update_actor,
        # since only the prompt's real views have embeddings. Confirmed this session: crashed a
        # ~4hr run at step 24/150 this exact way. Suppressing it at generation time is the fix,
        # not loosening the assertion -- the assertion is correct, the model just shouldn't be
        # able to produce this token as free text in the first place.
        sampling_params = dict(sampling_params)
        logit_bias = dict(sampling_params.get("logit_bias") or {})
        logit_bias[view_token_id] = -100
        sampling_params["logit_bias"] = logit_bias

        # Hard stop at the closing answer tag instead of relying on the model's own EOS
        # prediction. Confirmed this session (not a config bug -- eos_token_id is correctly wired
        # end to end): a fresh cold-start checkpoint just doesn't reliably predict <|im_end|> after
        # answering at all -- it either never closes <answer> within any practical budget, or
        # closes it and then keeps rambling into another disconnected block. Since SFT (which
        # teaches this stopping behavior) hasn't run yet for this ablation, truncate structurally:
        # once "</answer>" appears, stop there. include_stop_str_in_output=True keeps the tag
        # itself in the text -- echo_rl.reward.score's _ANSWER_RE needs the closing tag present to
        # match at all, and compute_score would score 0 outcome if we swallowed it.
        stop = list(sampling_params.get("stop") or [])
        if "</answer>" not in stop:
            stop.append("</answer>")
        sampling_params["stop"] = stop
        sampling_params["include_stop_str_in_output"] = True

        # (num_items, feature_size, hidden) is the shape vLLM's `is_embeddings()`
        # (vllm/multimodal/parse.py) requires to treat this as precomputed embeddings rather
        # than raw pixels needing HF processing -- feature_size=1, one soft token per view (see
        # echo_ep/vllm_model.py).
        image_data = view_embeddings.unsqueeze(1)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            # Consumed by the patched AgentLoopWorker._compute_multi_modal_inputs
            # (external/verl-hf-rollout-registry.patch) to reach the actor's training forward
            # pass -- NOT "images"/"videos" keys, this project's own shape.
            multi_modal_data={"view_embeddings": view_embeddings, "view_counts": n_views},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result
