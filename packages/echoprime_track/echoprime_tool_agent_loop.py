"""Multi-turn AgentLoop for the frozen-EchoPrime + Qwen3-8B-text GRPO track, adding
select_frames/zoom on top of the turn-0 all-views observation (echoprime_agent_loop.py's
single-turn loop stays as-is for anything not using tools -- NOTE: that file is on the OLDER
single-modality VIEW_TOKEN architecture, superseded by Task 2's dual clip/detr splice; it was
NOT updated as part of this task, see this module's own docstring notes below for why that
doesn't block this task). Mirrors the SAME accumulate-and-resend pattern verl's own
ToolAgentLoop (external/verl/verl/experimental/agent_loop/tool_agent_loop.py) uses for images:
prompt_ids grows every turn, and multi-modal data grows every turn -- each generate() call
resends the FULL accumulated prompt_ids/embeddings, relying on vLLM's own prefix caching
(confirmed this session by reading tool_agent_loop.py's _handle_generating_state/
_handle_processing_tools_state directly, not assumed).

No real HF processor exists for this text-only model (same reason echoprime_agent_loop.py gives
for not subclassing ToolAgentLoop), so tool-response placeholder text is built by hand, at the
EXACT token count grid.py's resolution produces -- never via apply_chat_template's image-count
inference, which needs a real processor we don't have.

=== Real structural changes made against the brief's draft (this task's Step 5 investigation) ===

1. **request_id is NOT a literal continued vLLM request.** Confirmed by reading
   `AsyncLLMServerManager.generate` (external/verl/verl/experimental/agent_loop/agent_loop.py):
   the `request_id` we pass in is used ONLY for `_acquire_server` (sticky routing to the same
   physical vLLM replica, for prefix-cache locality) -- the actual per-call vLLM `request_id`
   is freshly minted every turn (`request_id=uuid4().hex,  # use new request_id for each turn`,
   verbatim comment in that file). So "does generate() tolerate being called twice under the
   SAME request_id with a growing image_data tensor" is answered by construction: every turn is
   already an independent vLLM-level request from the engine's point of view, whether or not we
   reuse our own `request_id` value -- reusing it only affects load-balancer routing, not
   generation correctness. The documented fallback (fresh request_id per turn) is unnecessary;
   kept a single request_id per trajectory for the routing benefit only.

2. **`image_data=` cannot carry clip/detr data at all -- real, load-bearing bug in the brief's
   draft.** Read `vLLMHttpServer.generate` (external/verl/verl/workers/rollout/vllm_rollout/
   vllm_async_server.py): `image_data`/`video_data` are hardcoded to `multi_modal_data["image"]`/
   `multi_modal_data["video"]`. Our custom "clip"/"detr" modalities (packages/echoprime_track/
   vllm_model.py, Task 4, confirmed on real AMD infra) are NOT the same as vLLM's built-in
   "image" modality -- `MultiModalDataParser._get_subparsers` only recognizes 4 hardcoded keys,
   and even routing through the "image" key's own embedding path mis-tags the data as modality
   "image" (see vllm_model.py's own docstring, Bug A/Bug B from Task 4's investigation), which
   `EchoPrimeQwen3ForCausalLMVLLM.embed_multimodal` never reads (it only pops
   `clip_embeds`/`detr_embeds`). Sending clip_tensor through `image_data=` as the brief's draft
   does would hit exactly the "Expected there to be 1 clip items ... but only found 0" failure
   Task 4 already diagnosed for a different reason.

   Fixed at the verl layer (there is no way to fix this from AgentLoop code alone -- the stock
   API surface simply has no clip/detr-shaped kwarg): added `external/verl-mm-generate-
   passthrough.patch`, a new, narrowly-scoped patch (separate from Task 5's paused
   `external/verl-hf-rollout-registry.patch`, not touched) adding `clip_data`/`detr_data` kwargs
   to `AsyncLLMServerManager.generate` and `vLLMHttpServer.generate`, mirroring `image_data`/
   `video_data` exactly (same Optional[Any]=None shape, same "only add the key if not None"
   merge into multi_modal_data). Applied to the live `external/verl` submodule working tree
   (same convention as the other 3 patches there); `scripts/check_train_env.py` gained a
   matching presence check ("verl's generate() passes through clip/detr multi-modal data") so a
   fresh `git submodule update --init` that drops this patch fails loud instead of silently
   routing clip/detr data through the wrong key. This loop below calls
   `self.server_manager.generate(clip_data=..., detr_data=...)`, NOT `image_data=`.

3. **DETR data was computed but never sent.** The brief's draft built `detr_chunks`/
   `n_detr_turn0` (turn-0's DETR_TOKEN placeholder count) but never passed them into
   `generate()` at all -- yet Task 7's prompt (`generate_grpo_parquet.py`) bakes in real
   DETR_TOKEN placeholders whenever a view has RT-DETR detections, so any study with detections
   would have hit "expected N detr items ... found 0" on the very first real generate() call.
   Fixed: `detr_data` (built once at turn 0, constant across turns -- select_frames/zoom only
   touch the clip grid, DETR is turn-0-only context per the spec) is now passed on every call.

4. **`response_mask` mislabeled every token, tool-injected text included, as
   model-generated.** `AgentLoopOutput.response_mask`'s own docstring (agent_loop.py): "1 for
   LLM generated token, 0 for tool response token" -- stock ToolAgentLoop's
   `_handle_processing_tools_state` honors this (`agent_data.response_mask += [0] * ...` for
   tool text). The brief's draft set `response_mask = [1] * len(response_ids_all)` uniformly
   AFTER the loop, which would tell the actor to compute policy-gradient loss on text the model
   never generated (our own hand-built `<tool_response>...</tool_response>` placeholder text).
   Fixed: mask is now built per-segment inside the loop, 1 for real generation, 0 for injected
   tool-response text, matching stock semantics exactly.

5. **No response_length budget check before extending another turn.** Stock ToolAgentLoop
   checks `len(response_mask) >= response_length` right after each generate() call, and again
   before appending tool-response text, terminating early rather than silently growing past
   budget (only relying on a final slice). Added the same two checks -- avoids wasted generate()
   calls once the response is already at its cap, and keeps `response_mask`/`response_ids`
   coherent with what's actually charged against `response_length` turn by turn (the brief's
   draft only truncated once, at the very end, after doing all the extra work anyway).

6. **Per-turn stop condition.** echoprime_agent_loop.py (single-turn) already established, with
   real evidence (a 4hr training run crash), that this checkpoint class cannot be trusted to
   close `<answer>`/`<tool_call>` and then actually stop generating -- it either runs past the
   closing tag or never emits EOS in a practical budget. The brief's multi-turn draft had no
   stop condition at all. Added `stop=["</tool_call>", "</answer>"]`,
   `include_stop_str_in_output=True` (same rationale, same technique as the single-turn loop) --
   otherwise a single "turn" could rando-continue well past the point parse_action needs to see,
   corrupting the accumulate-and-resend trajectory.

MAX_TOOL_TURNS is set to 8, not the brief's placeholder 4: `packages/tool_env/config.py`'s
`EnvConfig.max_tool_calls` (the actual per-episode tool-call cap the image-based track uses) has
a real default of 8 (`ECHO_MAX_TOOL_CALLS` env override), read directly in this session rather
than left unexamined. Our loop resolves exactly one tool call per turn, so MAX_TOOL_TURNS (tool
rounds, i.e. generate() calls after turn 0) maps 1:1 onto that same per-episode cap.
"""
import json
import os
import re
from typing import Any
from uuid import uuid4

import h5py
import torch

# verl (and the vllm/torch-cu stack it pulls in transitively via TokenOutput's typing etc.) is
# only installed in the real training environments (the AMD MI210 box's
# .tmp_work/rocm_validation_20260914/env, or the CUDA box's requirements-train.txt env) -- NOT
# in the lighter `icardio3` env used for pure-logic development/testing on the login node (see
# this task's own brief: "Test: ... pure-logic pieces only ... the actual async run() against a
# real vLLM server is manual verification"). Guarded so `resolve_tool_call` below stays
# importable and unit-testable with plain torch/pytest wherever verl isn't installed;
# EchoPrimeToolAgentLoop itself (which genuinely needs verl) is simply absent from this module's
# namespace in that case, exactly like it would be unusable anyway without a real verl install.
try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    from verl.utils.profiler import simple_timer
    from verl.utils.rollout_trace import rollout_trace_op
    from verl.workers.rollout.replica import TokenOutput
    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False

from echoprime_track.darya_cache import load_clip_tokens, load_detr_tokens
from echoprime_track.grid import resolve_temporal_group, group_token_slice, spatial_subset
from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN
from echoprime_track.prompts import validate_tool_prompt
from tool_env.parse import parse_action

# Same env-var convention as Task 4's reward-side plan (packages/verl_bridge/reward.py's
# ECHO_DETR_H5). Defaults point at combined cache files covering the full study pool
# (both train and eval splits), keyed by dicom_uuid with no cross-split collision.
# ECHO_CLIP_H5 and ECHO_DETR_H5 can override these defaults if needed.
_CLIP_H5_PATH = os.environ.get(
    "ECHO_CLIP_H5", os.path.join(os.environ.get("ECHO_BUILD_DIR", "build"), "clip_tokens_all.h5"))
_DETR_H5_PATH = os.environ.get(
    "ECHO_DETR_H5", os.path.join(os.environ.get("ECHO_BUILD_DIR", "build"), "detections_all.h5"))

# packages/tool_env/config.py's EnvConfig.max_tool_calls default (the image-based track's real
# per-episode tool-call cap, ECHO_MAX_TOOL_CALLS-overridable there) -- confirmed by reading that
# file directly this session, not left at the brief's unexamined placeholder of 4. Our loop
# resolves exactly one tool call per turn, so this many tool ROUNDS (generate() calls after
# turn 0) matches that same per-episode budget 1:1.
MAX_TOOL_TURNS = 8

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def resolve_tool_call(call: dict, view_grids: dict):
    """`call` = {"name": ..., "arguments": {...}} (tool_env.parse.parse_action's shape).
    `view_grids` = {view_name: (393, 768) tensor}, this study's cached clip-token grids.

    Returns (tokens, text): `tokens` is a (K, 768) tensor to splice as K new CLIP_TOKEN
    placeholders, or None on any error (unknown view/tool/bad args); `text` is what to show the
    model (either a description of what it now sees, or the error)."""
    name = call.get("name")
    args = call.get("arguments", {})
    view = args.get("view")
    grid = view_grids.get(view)
    if grid is None:
        avail = ", ".join(sorted(view_grids.keys()))
        return None, f"unknown view {view!r}; available: {avail}"

    if name == "select_frames":
        try:
            group = resolve_temporal_group(args.get("frame_indices", []))
        except ValueError as e:
            return None, str(e)
        tokens = grid[group_token_slice(group)]
        return tokens, f"{view}: frames from temporal group {group}"

    if name == "zoom":
        try:
            group = resolve_temporal_group(args.get("frame_indices", []))
            idxs = spatial_subset(group, tuple(args.get("bbox", ())))
        except ValueError as e:
            return None, str(e)
        tokens = grid[idxs]
        return tokens, f"{view}: zoom on {len(idxs)} patches in temporal group {group}"

    return None, f"unknown tool {name!r}; available: select_frames, zoom"


if not _VERL_AVAILABLE:
    class EchoPrimeToolAgentLoop:  # pragma: no cover - only reached with verl uninstalled
        """Placeholder present only so `from echoprime_track.echoprime_tool_agent_loop import
        EchoPrimeToolAgentLoop` fails with a clear message instead of ImportError deep inside
        verl, when this module is imported somewhere verl isn't installed (e.g. the login
        node's `icardio3` env, used for `resolve_tool_call`'s pure-logic tests)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EchoPrimeToolAgentLoop needs verl installed (this env doesn't have it) -- "
                "resolve_tool_call is still importable and usable without verl.")
else:
    @register("echoprime_tool_agent")
    class EchoPrimeToolAgentLoop(AgentLoopBase):
        """Turn 0 = Darya-format all-views prompt (already baked into the parquet's `prompt`
        column by generate_grpo_parquet.py -- this loop just tokenizes it, same as
        EchoPrimeAgentLoop). Then up to MAX_TOOL_TURNS rounds of: parse a <tool_call>, resolve it
        against the study's cached grid, splice the result in as new CLIP_TOKEN placeholders,
        continue generation. Stops early on <answer> or MAX_TOOL_TURNS, same shape as
        tool_env/env.py's turn budget for the image-based track."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length
            self._clip_h5 = None
            self._detr_h5 = None

        @property
        def clip_h5(self):
            # Lazy, fork-safe (opened once per process on first access) -- same pattern
            # report_generation/sft_thinking/dataset.py::EchoVQAThinkingDataset uses.
            if self._clip_h5 is None:
                self._clip_h5 = h5py.File(_CLIP_H5_PATH, "r")
            return self._clip_h5

        @property
        def detr_h5(self):
            if self._detr_h5 is None:
                self._detr_h5 = h5py.File(_DETR_H5_PATH, "r")
            return self._detr_h5

        @rollout_trace_op
        async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
            messages = list(kwargs["raw_prompt"])
            validate_tool_prompt(messages)
            study_uuid = kwargs["extra_info"]["study_uuid"]
            dicom_uuids_by_view = json.loads(kwargs["extra_info"]["dicom_uuids_by_view"])

            # Read Darya's h5 caches directly, per dicom_uuid, lazily -- same access pattern her
            # own training code uses (report_generation/sft_thinking/dataset.py::
            # EchoVQAThinkingDataset), proven at real training scale. No intermediate per-study
            # cache (see Task 7's design correction note). self.clip_h5/self.detr_h5 are lazy
            # h5py.File properties, opened once per process on first access -- same fork-safe
            # pattern as her dataset class.
            view_grids = {}  # {view_name: (393, 768) torch.Tensor}, only views actually covered
            for view, dicom_uuid in dicom_uuids_by_view.items():
                tokens = load_clip_tokens(self.clip_h5, dicom_uuid)
                if tokens is not None:
                    view_grids[view] = torch.from_numpy(tokens)
            detr_tokens_by_view = {
                view: torch.from_numpy(load_detr_tokens(self.detr_h5, dicom_uuid))
                for view, dicom_uuid in dicom_uuids_by_view.items()
            }

            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            # Darya's SFT supplies <think>\n as context and masks it from the loss.
            # Supply the same prefix so generation starts inside the think block.
            # Markdown can be thinking content; allow enough tokens to reach </think>.
            prompt_text = prompt_text + "<think>\n"
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            clip_token_id = self.tokenizer.convert_tokens_to_ids(CLIP_TOKEN)
            detr_token_id = self.tokenizer.convert_tokens_to_ids(DETR_TOKEN)

            clip_chunks = []  # list of (K, 768) tensors, in prompt order
            detr_chunks = [t for t in detr_tokens_by_view.values() if t.shape[0] > 0]
            n_clip_turn0 = sum(1 for t in prompt_ids if t == clip_token_id)
            n_detr_turn0 = sum(1 for t in prompt_ids if t == detr_token_id)
            # Sanity check mirrors echoprime_agent_loop.py's existing assertion pattern -- and
            # Task 7's build_row uses this SAME darya_cache.load_clip_tokens/load_detr_tokens
            # logic to build the prompt text, so this should always agree (both read the same h5
            # files the same way); a mismatch here means the parquet and the h5 files disagree
            # (e.g. the parquet was built against a different/stale h5 snapshot).
            assert n_clip_turn0 == sum(g.shape[0] for g in view_grids.values()), (
                f"study {study_uuid}: prompt has {n_clip_turn0} {CLIP_TOKEN!r} placeholders but "
                f"the h5 files give {sum(g.shape[0] for g in view_grids.values())} -- prompt "
                "construction (generate_grpo_parquet.py) and the h5 files disagree")
            assert n_detr_turn0 == sum(t.shape[0] for t in detr_tokens_by_view.values()), (
                f"study {study_uuid}: prompt has {n_detr_turn0} {DETR_TOKEN!r} placeholders but "
                f"the h5 files give {sum(t.shape[0] for t in detr_tokens_by_view.values())} -- "
                "prompt construction (generate_grpo_parquet.py) and the h5 files disagree")
            if view_grids:
                clip_chunks.append(torch.cat(list(view_grids.values()), dim=0))
            # detr_chunks is fixed at turn 0 -- select_frames/zoom only ever touch the clip grid
            # (spec: DETR is turn-0-only context, no DETR-returning tool exists), so unlike
            # clip_chunks this never grows across turns.
            detr_tensor = torch.cat(detr_chunks, dim=0).unsqueeze(1) if detr_chunks else None

            sampling_params = dict(sampling_params)
            logit_bias = dict(sampling_params.get("logit_bias") or {})
            logit_bias[clip_token_id] = -100
            logit_bias[detr_token_id] = -100
            sampling_params["logit_bias"] = logit_bias
            # Same rationale as echoprime_agent_loop.py's hard `</answer>` stop (an SFT/cold-start
            # checkpoint here is not reliably trusted to emit EOS right after closing a
            # structural tag) -- generalized to also stop at `</tool_call>`, since a multi-turn
            # round must end there for parse_action/resolve_tool_call to see exactly one clean
            # call per turn.
            stop = list(sampling_params.get("stop") or [])
            for tag in ("</tool_call>", "</answer>"):
                if tag not in stop:
                    stop.append(tag)
            sampling_params["stop"] = stop
            sampling_params["include_stop_str_in_output"] = True

            # request_id is a routing hint only (sticky session -> prefix-cache locality on the
            # same replica), NOT a literal continued vLLM-level request -- AsyncLLMServerManager.
            # generate mints a fresh per-call request_id internally regardless (see module
            # docstring point 1). Kept stable across turns for that routing benefit.
            request_id = uuid4().hex
            response_ids_all = []
            response_mask = []  # 1 = model-generated token, 0 = tool-injected token
            # (AgentLoopOutput docstring's own contract)
            response_logprobs_all = []
            have_logprobs = True  # downgrades to False (-> None on output) the first turn a
            # generate() call doesn't return log_probs, matching stock ToolAgentLoop/
            # EchoPrimeAgentLoop's "only carry logprobs if the server actually returned them"
            # rule.
            metrics = {}
            turn = 0

            for turn in range(MAX_TOOL_TURNS + 1):
                clip_data = torch.cat(clip_chunks, dim=0).unsqueeze(1) if clip_chunks else None
                with simple_timer(f"generate_turn{turn}", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        clip_data=clip_data,
                        detr_data=detr_tensor,
                    )
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = (
                        output.num_preempted if output.num_preempted is not None else -1)
                else:
                    metrics["num_preempted"] += (
                        output.num_preempted if output.num_preempted is not None else 0)

                new_ids = output.token_ids
                response_ids_all += new_ids
                response_mask += [1] * len(new_ids)
                if output.log_probs and have_logprobs:
                    response_logprobs_all += output.log_probs
                else:
                    have_logprobs = False
                prompt_ids = prompt_ids + new_ids
                text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

                # Budget check mirrors stock ToolAgentLoop's _handle_generating_state: stop once
                # the response is already at cap rather than silently paying for another turn
                # only to discard it in the final slice.
                if len(response_mask) >= self.response_length:
                    break

                parsed = parse_action(text)
                if parsed.answer is not None or not parsed.calls or turn == MAX_TOOL_TURNS:
                    break

                call = parsed.calls[0]  # spec: 1 tool call resolved per turn
                tokens, tool_text = resolve_tool_call(call, view_grids)
                if tokens is None:
                    tool_response = f"<tool_response>{tool_text}</tool_response>"
                else:
                    clip_chunks.append(tokens)
                    placeholder = CLIP_TOKEN * tokens.shape[0]
                    tool_response = f"<tool_response>{tool_text}\n{placeholder}</tool_response>"
                tool_ids = self.tokenizer(tool_response, add_special_tokens=False)["input_ids"]

                # Same budget check as above, mirroring stock's second checkpoint (right before
                # committing tool-response tokens into the trajectory).
                if len(response_mask) + len(tool_ids) >= self.response_length:
                    break

                response_ids_all += tool_ids
                response_mask += [0] * len(tool_ids)  # tool-injected text: no policy-gradient
                # loss on it.
                if have_logprobs:
                    response_logprobs_all += [0.0] * len(tool_ids)
                prompt_ids = prompt_ids + tool_ids

            clip_counts = int(torch.cat(clip_chunks, dim=0).shape[0]) if clip_chunks else 0
            detr_counts = int(torch.cat(detr_chunks, dim=0).shape[0]) if detr_chunks else 0

            result = AgentLoopOutput(
                prompt_ids=prompt_ids[: len(prompt_ids) - len(response_ids_all)],
                response_ids=response_ids_all[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=(
                    response_logprobs_all[: self.response_length]
                    if have_logprobs and response_logprobs_all else None),
                multi_modal_data={
                    "clip_embeddings": torch.cat(clip_chunks, dim=0) if clip_chunks else None,
                    "clip_counts": clip_counts,
                    "detr_embeddings": torch.cat(detr_chunks, dim=0) if detr_chunks else None,
                    "detr_counts": detr_counts,
                },
                # Historical metric: 2 means a single generation, with no tool continuation.
                num_turns=2 + turn,
                metrics=metrics,
                extra_fields={},
            )
            result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            return result
