"""HFRollout subclass that threads EchoPrime view embeddings through generation.

`external/verl/verl/workers/rollout/hf_rollout.py`'s `HFRollout` (registered under name "hf" via
`external/verl-hf-rollout-registry.patch`) only ever passes `input_ids`/`attention_mask`/
`position_ids` into `self.module.generate(...)` -- it has no concept of the generic
`multi_modal_inputs` non-tensor-batch channel verl's FSDP actor already understands (see
`echo_ep/rl_dataset.py`'s docstring). Rollout is the one place that channel doesn't reach on its
own, so this subclass pulls it out of `prompts.non_tensor_batch` and adds it to the `generate()`
call -- everything else is copied from `HFRollout._generate_minibatch` unchanged (verl gives no
narrower extension point to override just the kwargs going into `.generate()`).
"""
import contextlib

import torch
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.hf_rollout import HFRollout

__all__ = ["HFRolloutVideo"]


class HFRolloutVideo(HFRollout):
    # Stock `HFRollout.__init__(self, module, config)` doesn't match how
    # `fsdp_workers.py:_build_rollout` actually constructs a rollout
    # (`get_rollout_class(...)(config=..., model_config=..., device_mesh=...)`) -- confirming,
    # along with the missing resume/release/update_weights below, that this class was never
    # actually wired end-to-end in this verl version. `self.module` (the live FSDP-wrapped
    # model to generate from) is bound afterward by the small addition this project's patch
    # makes to `_build_rollout` itself (`external/verl-hf-rollout-registry.patch`) -- there's no
    # separate inference engine here to hand a module to at construction time.
    def __init__(self, config, model_config, device_mesh=None, **kwargs):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.module = None

    # `BaseRollout` declares these abstract for engines with a SEPARATE inference process/GPU
    # pool that needs explicit weight syncing and memory release (vLLM, sglang, trtllm) -- none
    # of that applies here: HFRollout generates through the SAME live FSDP-wrapped actor module
    # used for training, so there's no separate weights to push and no separate memory to
    # release/resume. Stock `HFRollout` doesn't implement these either (confirmed by reading it
    # in full), which is presumably why it's never been runnable through the normal
    # `get_rollout_class` path before this project's registry patch.
    async def resume(self, tags: list[str]):
        pass

    async def release(self):
        pass

    async def update_weights(self, weights, **kwargs):
        pass

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if not do_sample:
            kwargs = {"do_sample": False, "num_beams": 1}
        elif is_validate:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        # -- EchoSonarVideo addition: pull view_embeddings/view_counts out of the generic
        # multi_modal_inputs channel and thread them into generate(). Our model's
        # `prepare_inputs_for_generation` (echo_ep/modeling.py) only consumes these on the first
        # decode step -- see that method's docstring for why later steps don't need them.
        mm_inputs = extract_multi_modal_inputs(prompts.non_tensor_batch["multi_modal_inputs"])
        view_embeddings = mm_inputs["view_embeddings"].to(idx.device)
        view_counts = mm_inputs["view_counts"].to(idx.device)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                view_embeddings=view_embeddings,
                view_counts=view_counts,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,
                return_dict_in_generate=True,
                use_cache=True,
            )

        seq = output.sequences
        generated_batch_size = seq.size(0)

        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
