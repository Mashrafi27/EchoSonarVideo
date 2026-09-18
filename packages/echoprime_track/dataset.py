"""SFT dataset + collator for the frozen-EchoPrime + Qwen3-8B-text cold-start.

Source: `data/raw_vqa/train_vqa_with_thinking.jsonl` (128,215 records, 5,061 studies, every
record has a populated `thinking` field -- verified this session). `messages` there is
already plain text-only chat (no image refs), so unlike the existing Qwen3-VL SFT path
(`verl_bridge/sft_dataset.py`, `data_core/data/trace.py`) there is no tool-call trajectory to
build -- this is a direct question -> reasoning -> answer SFT record, matching EchoSonar-R's
own (tool-less) cold-start recipe.

Video input is the STUDY's cached per-view embeddings (`echoprime_track.build_video_cache`'s output),
looked up by `study_uuid` -- never raw video at train time (see modeling.py's docstring for
why: the encoder is frozen, caching once per study and reusing across that study's ~25 QA
pairs is correct and avoids 25x redundant encoder passes).
"""
import glob
import json
import os

import torch
from torch.utils.data import Dataset

from echoprime_track.modeling import VIEW_TOKEN

# Matches Darya's real SFT checkpoint format: no <answer> tags -- her checkpoint was trained on
# plain text after </think> and will ignore a <answer> instruction it never saw.  The tool
# descriptions give the model the exact call format so GRPO reward can reinforce correct use.
SYSTEM_PROMPT = (
    "You are an expert cardiologist reviewing a multi-view echocardiography study. "
    "Reason step by step inside <think> </think>. "
    "You may call tools between reasoning turns to retrieve additional clip tokens for specific "
    "frames or regions:\n"
    '  select_frames: <tool_call>{"name": "select_frames", "arguments": {"view": "<view>", '
    '"frame_indices": [0, 1, ...]}}</tool_call>\n'
    '  zoom: <tool_call>{"name": "zoom", "arguments": {"view": "<view>", '
    '"frame_indices": [0, 1, ...], "bbox": [x0, y0, x1, y1]}}</tool_call>\n'
    "After your final </think>, give your answer directly. Keep it concise and clinically precise."
)


def _assistant_answer(messages: list) -> str:
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    raise ValueError("no assistant turn in messages")


def _user_question(messages: list) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    raise ValueError("no user turn in messages")


class EchoPrimeSFTDataset(Dataset):
    """One example = one QA pair. `__getitem__` returns raw fields; tokenization + view-token
    splicing happens in the collator (batch-level, so padding/truncation see the whole batch)."""

    def __init__(self, jsonl_path: str, cache_dir: str, max_views: int = 50):
        self.cache_dir = cache_dir
        self.max_views = max_views
        self.records = []
        skipped_no_cache = 0
        cached_studies = {os.path.basename(p)[:-3]
                           for p in glob.glob(os.path.join(cache_dir, "*.pt"))}
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["study_uuid"] not in cached_studies:
                    skipped_no_cache += 1
                    continue
                self.records.append(rec)
        print(f"[echoprime_track] dataset: {len(self.records)} QA pairs "
              f"({skipped_no_cache} skipped, study not yet cached) "
              f"from {jsonl_path}", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        cache = torch.load(os.path.join(self.cache_dir, f"{rec['study_uuid']}.pt"))
        view_embeddings = cache["embeddings"][:self.max_views]
        question = _user_question(rec["messages"])
        answer = _assistant_answer(rec["messages"])
        return {
            "study_uuid": rec["study_uuid"],
            "question": question,
            "thinking": rec["thinking"],
            "answer": answer,
            "view_embeddings": view_embeddings,  # (N, 512)
        }


class EchoPrimeCollator:
    """Tokenizes a batch, splices VIEW_TOKEN placeholders (one per view, at the start of the
    user turn), masks everything but the assistant turn out of `labels`, and concatenates
    view embeddings across the batch (unpadded -- modeling.py's view_counts tells the model
    where each example's slice starts/ends)."""

    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.view_token_id = tokenizer.convert_tokens_to_ids(VIEW_TOKEN)
        assert self.view_token_id != tokenizer.unk_token_id, (
            f"{VIEW_TOKEN!r} not in the tokenizer's vocab -- call "
            "tokenizer.add_special_tokens({'additional_special_tokens': [VIEW_TOKEN]}) and "
            "model.lm.resize_token_embeddings(len(tokenizer)) before training")

    def __call__(self, batch: list) -> dict:
        input_ids_list, labels_list = [], []
        for ex in batch:
            n_views = ex["view_embeddings"].shape[0]
            view_prefix = VIEW_TOKEN * n_views
            user_turn = f"{view_prefix}\n{ex['question']}"
            assistant_turn = f"<think>{ex['thinking']}</think>\n{ex['answer']}"

            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_turn}],
                add_generation_prompt=True, tokenize=True)
            full_ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_turn},
                 {"role": "assistant", "content": assistant_turn}],
                add_generation_prompt=False, tokenize=True)

            if len(full_ids) > self.max_length:
                # Truncate the ASSISTANT tail, never the view-token prefix -- losing view
                # placeholders would desync input_ids/view_embeddings splicing in modeling.py.
                full_ids = full_ids[:self.max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            input_ids_list.append(full_ids)
            labels_list.append(labels)

        max_len = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
            labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)

        view_counts = [int((ids == self.view_token_id).sum()) for ids in input_ids]
        view_embeddings = torch.cat([ex["view_embeddings"] for ex in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "view_embeddings": view_embeddings,
            "view_counts": view_counts,
        }
