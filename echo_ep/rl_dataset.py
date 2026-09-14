"""GRPO dataset for the frozen-EchoPrime + Qwen3-8B-text track (`data.custom_cls` target).

NOT a `verl.utils.dataset.rl_dataset.RLHFDataset` subclass. That class's `__getitem__` was
refactored (verl commit 51d27376, "major refactor RLHFDataset for multi-modal data") to defer
ALL chat-template/tokenization work to the async AgentLoop machinery -- its own docstring says
so directly ("For rollout, apply_chat_template has been moved to AgentLoop, so we only return
raw_prompt here"). `HFRollout` (`external/verl/verl/workers/rollout/hf_rollout.py`) was never
updated for that: it still reads `prompts.batch["input_ids"]`/`attention_mask`/`position_ids`
directly, expecting them to already exist -- the OLDER eager-tokenization shape. This class
replicates that older shape directly (confirmed via `git show 51d27376~1:verl/utils/dataset/
rl_dataset.py`), using the plain-tokenizer branch (no HF vision processor -- our "multimodal"
content isn't image/video pixels for a processor, it's precomputed EchoPrime view embeddings).

`view_embeddings`/`view_counts` reach the model via `non_tensor_batch["multi_modal_inputs"]`,
the SAME generic per-example-dict channel verl's FSDP actor already unpacks as `**kwargs` into
the model's forward (`verl/workers/actor/dp_actor.py:264,374`, via
`verl/utils/model.py:extract_multi_modal_inputs`, which `torch.cat(dim=0)`s each key across the
batch) -- no patch to verl's actor needed, only to the rollout (`echo_ep/hf_rollout_video.py`),
which doesn't have this generic passthrough at all.
"""
import os

import torch
from torch.utils.data import Dataset

from echo_ep.dataset import SYSTEM_PROMPT
from echo_ep.modeling import VIEW_TOKEN


class EchoPrimeGRPODataset(Dataset):
    """One row = one QA pair (question + `reward_key`-shaped ground truth, from `build/rl.jsonl`
    via `echo_ep.generate_grpo_parquet` -- no `thinking` needed, GRPO scores the policy's own
    generated reasoning, not a reference trace)."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        """Signature matches verl's `create_rl_dataset` call convention exactly
        (`verl/trainer/main_ppo.py:create_rl_dataset`: `dataset_cls(data_files=..., tokenizer=...,
        processor=..., config=..., max_samples=...)`), even though this class ignores
        `processor` entirely (no HF vision processor involved -- our "multimodal" content is
        precomputed EchoPrime embeddings, not image/video pixels for a processor to encode).
        `config` is the `data:` block of the yaml -- `cache_dir`/`max_views`/`truncation` are
        this project's own extra keys on it, not stock verl fields."""
        import pyarrow.parquet as pq

        if isinstance(data_files, str):
            data_files = [data_files]
        self.rows = []
        for f in data_files:
            self.rows.extend(pq.read_table(f).to_pylist())
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]

        self.tokenizer = tokenizer
        self.cache_dir = config.get("cache_dir", "build/echoprime_video_cache")
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.max_views = config.get("max_views", 50)
        self.truncation = config.get("truncation", "error")
        self.view_token_id = tokenizer.convert_tokens_to_ids(VIEW_TOKEN)
        assert self.view_token_id != tokenizer.unk_token_id, (
            f"{VIEW_TOKEN!r} not in the tokenizer's vocab -- the checkpoint at `model.path` "
            "must be one whose tokenizer already had this special token added and saved "
            "(see echo_ep/train_sft.py's build_model_and_tokenizer / the checkpoint-prep script)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import postprocess_data

        row = self.rows[idx]
        study_uuid = row["extra_info"]["study_uuid"]
        cache = torch.load(os.path.join(self.cache_dir, f"{study_uuid}.pt"))
        view_embeddings = cache["embeddings"][:self.max_views]
        n_views = view_embeddings.shape[0]

        user_turn = f"{VIEW_TOKEN * n_views}\n{row['question']}"
        raw_prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_turn}],
            add_generation_prompt=True, tokenize=False)

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = postprocess_data(
            input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"],
            max_length=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True, truncation=self.truncation)
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "multi_modal_inputs": {
                "view_embeddings": view_embeddings,
                "view_counts": torch.tensor([n_views]),
            },
            "raw_prompt_ids": self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            "data_source": row["data_source"],
            "ability": row["ability"],
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"],
            "index": idx,
        }
