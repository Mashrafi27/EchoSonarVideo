"""SFT dataset + collator for the frozen-EchoPrime + Qwen3-8B-text cold-start.

Source: `Archive 2 (1)/train_vqa_with_thinking.jsonl` (128,215 records, 5,061 studies, every
record has a populated `thinking` field -- verified this session). `messages` there is
already plain text-only chat (no image refs), so unlike the existing Qwen3-VL SFT path
(`echo_verl/sft_dataset.py`, `echo_rl/data/trace.py`) there is no tool-call trajectory to
build -- this is a direct question -> reasoning -> answer SFT record, matching EchoSonar-R's
own (tool-less) cold-start recipe.

Video input is the STUDY's cached per-view embeddings (`echo_ep.build_video_cache`'s output),
looked up by `study_uuid` -- never raw video at train time (see modeling.py's docstring for
why: the encoder is frozen, caching once per study and reusing across that study's ~25 QA
pairs is correct and avoids 25x redundant encoder passes).
"""
import glob
import json
import os

import torch
from torch.utils.data import Dataset

from echo_ep.modeling import VIEW_TOKEN

# Matches echo_verl/generate_trainset.py's _SYSTEM convention (<think>/<answer> tags) for
# continuity with the rest of the project, even though this track has no tools.
SYSTEM_PROMPT = (
    "You are an expert cardiologist reviewing a multi-view echocardiography study. "
    "Reason step by step inside <think> </think>, then give your final answer inside "
    "<answer> </answer>. Keep the answer concise and clinically precise, in the same style "
    "a report would use."
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
        print(f"[echo_ep] dataset: {len(self.records)} QA pairs "
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
            assistant_turn = f"<think>{ex['thinking']}</think>\n<answer>{ex['answer']}</answer>"

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
