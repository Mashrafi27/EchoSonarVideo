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
content isn't image/video pixels for a processor, it's precomputed clip/DETR feature tokens).

Task 7 (Darya-format prompts): `echoprime_track/generate_grpo_parquet.py` now bakes the FULL
per-view prompt text (view name + CLIP_TOKEN/DETR_TOKEN placeholders) into the parquet's
`prompt` column at generation time -- this class just tokenizes that directly via
`apply_chat_template` instead of rebuilding it from `extra_info` a second time (removes what
used to be duplicated prompt-construction logic between generate_grpo_parquet.py and this
file). Clip/DETR feature tensors themselves are NOT stored in the parquet (they're 393x768 /
Nx256 floats per view -- far too large for a parquet cell); this class reads them lazily,
per dicom_uuid, straight out of Darya's h5 caches via darya_cache.py, the SAME access pattern
her own training code uses (report_generation/sft_thinking/dataset.py::EchoVQAThinkingDataset)
and the SAME one generate_grpo_parquet.py used to compute placeholder counts at build time --
one shared source of truth for the h5-reading logic (darya_cache.py), used at both generation
time and here.

`clip_embeddings`/`clip_counts`/`detr_embeddings`/`detr_counts` reach the model via
`non_tensor_batch["multi_modal_inputs"]`, the SAME generic per-example-dict channel verl's FSDP
actor already unpacks as `**kwargs` into the model's forward (`verl/workers/actor/dp_actor.py:
264,374`, via `verl/utils/model.py:extract_multi_modal_inputs`, which `torch.cat(dim=0)`s each
key across the batch) -- no patch to verl's actor needed, only to the rollout
(`echoprime_track/hf_rollout_video.py`), which doesn't have this generic passthrough at all.
These four names match `modeling.py::EchoPrimeQwen3ForCausalLM.forward`'s own parameter names
exactly (it replaced the old single-modality `view_embeddings`/`view_counts` with one pair per
modality, clip and detr, spliced independently via `_splice_positions`).
"""
import json
import os

import h5py
import torch
from torch.utils.data import Dataset

from echoprime_track.darya_cache import load_clip_tokens, load_detr_tokens
from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN


class EchoPrimeGRPODataset(Dataset):
    """One row = one QA pair (question + `reward_key`-shaped ground truth, from `build/rl.jsonl`
    via `echoprime_track.generate_grpo_parquet` -- no `thinking` needed, GRPO scores the policy's own
    generated reasoning, not a reference trace). Restricted at parquet-build time to the two
    verifiable question types (`generate_grpo_parquet.VERIFIABLE_QUESTION_TYPES`)."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        """Signature matches verl's `create_rl_dataset` call convention exactly
        (`verl/trainer/main_ppo.py:create_rl_dataset`: `dataset_cls(data_files=..., tokenizer=...,
        processor=..., config=..., max_samples=...)`), even though this class ignores
        `processor` entirely (no HF vision processor involved -- our "multimodal" content is
        precomputed clip/DETR feature tokens, not image/video pixels for a processor to encode).
        `config` is the `data:` block of the yaml -- `clip_h5_path`/`detr_h5_path`/
        `max_prompt_length`/`truncation` are this project's own extra keys on it, not stock
        verl fields."""
        import pyarrow.parquet as pq

        if isinstance(data_files, str):
            data_files = [data_files]
        self.rows = []
        for f in data_files:
            self.rows.extend(pq.read_table(f).to_pylist())
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]

        self.tokenizer = tokenizer
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.truncation = config.get("truncation", "error")

        # Lazy h5 opening (fork-safe for DataLoader workers) -- see clip_h5/detr_h5 properties
        # below, same pattern report_generation/sft_thinking/dataset.py::
        # EchoVQAThinkingDataset.clip_h5/detr_h5 uses.
        self._clip_h5_path = config["clip_h5_path"]
        self._detr_h5_path = config["detr_h5_path"]
        self._clip_h5 = None
        self._detr_h5 = None

        self.clip_token_id = tokenizer.convert_tokens_to_ids(CLIP_TOKEN)
        self.detr_token_id = tokenizer.convert_tokens_to_ids(DETR_TOKEN)
        assert self.clip_token_id != tokenizer.unk_token_id, (
            f"{CLIP_TOKEN!r} not in the tokenizer's vocab -- the checkpoint at `model.path` "
            "must be one whose tokenizer already had this special token added and saved "
            "(see echoprime_track/modeling.py / the checkpoint-prep script)")
        assert self.detr_token_id != tokenizer.unk_token_id, (
            f"{DETR_TOKEN!r} not in the tokenizer's vocab -- the checkpoint at `model.path` "
            "must be one whose tokenizer already had this special token added and saved "
            "(see echoprime_track/modeling.py / the checkpoint-prep script)")

    @property
    def clip_h5(self) -> h5py.File:
        if self._clip_h5 is None:
            self._clip_h5 = h5py.File(self._clip_h5_path, "r")
        return self._clip_h5

    @property
    def detr_h5(self) -> h5py.File:
        if self._detr_h5 is None:
            self._detr_h5 = h5py.File(self._detr_h5_path, "r")
        return self._detr_h5

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import postprocess_data

        row = self.rows[idx]

        raw_prompt = self.tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False)

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = postprocess_data(
            input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"],
            max_length=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True, truncation=self.truncation)
        position_ids = compute_position_id_with_mask(attention_mask)

        # `dicom_uuids_by_view` was JSON-encoded by generate_grpo_parquet.py::write_parquet
        # specifically so this dict's key ORDER survives the parquet round-trip intact
        # (pyarrow's struct-type inference for a raw dict does not preserve per-row insertion
        # order -- see that function's docstring). That order must match the CLIP_TOKEN/
        # DETR_TOKEN placement order in `row["prompt"]`'s text (both built from the same
        # dict, in the same iteration order, at generation time) -- modeling.py's
        # `_splice_positions` fills each modality's placeholder positions, in prompt order,
        # from this tensor's rows, in the order concatenated below. Iterating the SAME
        # decoded dict here (rather than re-deriving view order some other way) is what keeps
        # the two in sync.
        dicom_uuids_by_view = json.loads(row["extra_info"]["dicom_uuids_by_view"])

        clip_chunks, detr_chunks = [], []
        for dicom_uuid in dicom_uuids_by_view.values():
            clip_tokens = load_clip_tokens(self.clip_h5, dicom_uuid)
            if clip_tokens is not None:
                clip_chunks.append(torch.from_numpy(clip_tokens))
            detr_tokens = load_detr_tokens(self.detr_h5, dicom_uuid)
            if detr_tokens.shape[0] > 0:
                detr_chunks.append(torch.from_numpy(detr_tokens))

        clip_embeddings = torch.cat(clip_chunks, dim=0) if clip_chunks \
            else torch.zeros((0, 768), dtype=torch.float32)
        detr_embeddings = torch.cat(detr_chunks, dim=0) if detr_chunks \
            else torch.zeros((0, 256), dtype=torch.float32)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "multi_modal_inputs": {
                "clip_embeddings": clip_embeddings,
                "clip_counts": torch.tensor([clip_embeddings.shape[0]]),
                "detr_embeddings": detr_embeddings,
                "detr_counts": torch.tensor([detr_embeddings.shape[0]]),
            },
            "raw_prompt_ids": self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            "data_source": row["data_source"],
            "ability": row["ability"],
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"],
            "index": idx,
        }
