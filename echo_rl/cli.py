import argparse
import json
import os
from collections import Counter
from echo_rl.config import Config
from echo_rl.data.studies import study_dir, index_study
from echo_rl.data.builders import iter_jsonl, sft_record, rl_record
from echo_rl.data.gold import load_all
from echo_rl.data.split import assign_split, load_canonical_split
from echo_rl.data.balance import resample_indices
from echo_rl.data.answers import last_answer, parse_yes_no


def _clip_cache(cfg):
    cache = {}
    def get(uuid):
        if uuid not in cache:
            try:
                cache[uuid] = index_study(study_dir(cfg.preprocessed_dir, uuid))
            except FileNotFoundError:
                cache[uuid] = None
        return cache[uuid]
    return get


def _out(cfg, name):
    os.makedirs(cfg.out_dir, exist_ok=True)
    return os.path.join(cfg.out_dir, name)


def build_sft(cfg, limit, split="train"):
    get = _clip_cache(cfg)
    canonical = load_canonical_split(cfg.study_split)
    n = 0
    with open(_out(cfg, "sft.jsonl"), "w") as w:
        for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
            if limit and i >= limit:
                break
            clips = get(rec["study_uuid"])
            if not clips:
                continue
            sp = assign_split(rec["study_uuid"], canonical)
            if split != "all" and sp != split:
                continue
            w.write(json.dumps(sft_record(rec, clips, cfg)) + "\n")
            n += 1
    print(f"[build-sft] wrote {n} records")


def build_rl(cfg, limit, split="train"):
    get = _clip_cache(cfg)
    gold = load_all(cfg.gold_dir) if os.path.isdir(cfg.gold_dir) else {}
    canonical = load_canonical_split(cfg.study_split)
    records, labels = [], []
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        clips = get(rec["study_uuid"])
        if not clips:
            continue
        sp = assign_split(rec["study_uuid"], canonical)
        if split != "all" and sp != split:
            continue
        out = rl_record(rec, clips, cfg, gold)
        out["split"] = sp
        records.append(out)
        labels.append("abn" if out["reward_key"]["is_abnormal"] else "norm")
    with open(_out(cfg, "rl.jsonl"), "w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]
    train_labels = [labels[i] for i in train_idx]
    pool = resample_indices(train_labels, len(train_labels), cfg.seed) if train_labels else []
    pool = [train_idx[j] for j in pool]
    with open(_out(cfg, "rl_pool.json"), "w") as f:
        json.dump({"indices": pool}, f)
    print(f"[build-rl] wrote {len(records)} records; balanced pool {len(pool)}")


def build_eval(cfg, limit, split="test"):
    """Held-out evaluation records from the TEST vqa file (no thinking traces).

    Separate from build-rl because it reads cfg.vqa_test, not cfg.vqa_train, and
    writes eval.jsonl. Same rl_record shape, so the scorers in echo_rl.reward.score
    apply unchanged. The split filter still runs: test_vqa.jsonl should be pure
    test, and a non-test study appearing here is a leak worth failing loudly on.
    """
    get = _clip_cache(cfg)
    gold = load_all(cfg.gold_dir) if os.path.isdir(cfg.gold_dir) else {}
    canonical = load_canonical_split(cfg.study_split)
    records, skipped_split, skipped_clips = [], 0, 0
    for i, rec in enumerate(iter_jsonl(cfg.vqa_test)):
        if limit and i >= limit:
            break
        sp = assign_split(rec["study_uuid"], canonical)
        if split != "all" and sp != split:
            skipped_split += 1
            continue
        clips = get(rec["study_uuid"])
        if not clips:
            skipped_clips += 1
            continue
        out = rl_record(rec, clips, cfg, gold)
        out["split"] = sp
        records.append(out)
    with open(_out(cfg, "eval.jsonl"), "w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")
    qt = Counter(r["question_type"] for r in records)
    print(f"[build-eval] wrote {len(records)} records to eval.jsonl "
          f"(skipped {skipped_split} wrong-split, {skipped_clips} missing-clips)")
    print(f"[build-eval] question types: {dict(qt)}")


def stats(cfg, limit, split="train"):
    get = _clip_cache(cfg)
    canonical = load_canonical_split(cfg.study_split)
    joined = total = 0
    qt = Counter(); yn = Counter()
    sampled_studies = []
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        total += 1
        sampled_studies.append(rec["study_uuid"])
        if get(rec["study_uuid"]):
            joined += 1
        qt[rec["question_type"]] += 1
        if rec["question_type"] == "abnormality_classification":
            yn[parse_yes_no(last_answer(rec["messages"]))] += 1
    split_sizes = Counter(assign_split(u, canonical) for u in sampled_studies)
    missing = sum(1 for u in sampled_studies if u not in canonical)
    print(f"[stats] join {joined}/{total}; types {dict(qt)}; yes/no {dict(yn)}")
    print(f"[stats] split sizes (canonical) {dict(split_sizes)}; missing from canonical map {missing}/{total}")


def run(argv=None):
    p = argparse.ArgumentParser(prog="echo_rl")
    p.add_argument("cmd", choices=["build-sft", "build-rl", "build-eval", "stats"])
    p.add_argument("--limit", type=int, default=0)
    # Default split follows the command: building an eval set from the train split
    # would silently emit zero records.
    p.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    args = p.parse_args(argv)
    split = args.split or ("test" if args.cmd == "build-eval" else "train")
    cfg = Config.from_env()
    {"build-sft": build_sft, "build-rl": build_rl, "build-eval": build_eval,
     "stats": stats}[args.cmd](cfg, args.limit, split)


if __name__ == "__main__":
    run()
