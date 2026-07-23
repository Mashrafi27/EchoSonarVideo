import argparse
import json
import os
from collections import Counter
from echo_rl.config import Config
from echo_rl.data.studies import study_dir, index_study
from echo_rl.data.builders import iter_jsonl, sft_record, rl_record
from echo_rl.data.gold import load_all
from echo_rl.data.split import assign_split, load_test_studies
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


def build_sft(cfg, limit):
    get = _clip_cache(cfg)
    n = 0
    with open(_out(cfg, "sft.jsonl"), "w") as w:
        for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
            if limit and i >= limit:
                break
            clips = get(rec["study_uuid"])
            if not clips:
                continue
            w.write(json.dumps(sft_record(rec, clips, cfg)) + "\n")
            n += 1
    print(f"[build-sft] wrote {n} records")


def build_rl(cfg, limit):
    get = _clip_cache(cfg)
    gold = load_all(cfg.gold_dir) if os.path.isdir(cfg.gold_dir) else {}
    tests = load_test_studies(cfg.vqa_test) if os.path.exists(cfg.vqa_test) else set()
    records, labels = [], []
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        clips = get(rec["study_uuid"])
        if not clips:
            continue
        out = rl_record(rec, clips, cfg, gold)
        out["split"] = assign_split(rec["study_uuid"], gold.get(rec["study_uuid"], {}).get("designation"), tests)
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


def stats(cfg, limit):
    get = _clip_cache(cfg)
    joined = total = 0
    qt = Counter(); yn = Counter()
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        total += 1
        if get(rec["study_uuid"]):
            joined += 1
        qt[rec["question_type"]] += 1
        if rec["question_type"] == "abnormality_classification":
            yn[parse_yes_no(last_answer(rec["messages"]))] += 1
    print(f"[stats] join {joined}/{total}; types {dict(qt)}; yes/no {dict(yn)}")


def run(argv=None):
    p = argparse.ArgumentParser(prog="echo_rl")
    p.add_argument("cmd", choices=["build-sft", "build-rl", "stats"])
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    cfg = Config.from_env()
    {"build-sft": build_sft, "build-rl": build_rl, "stats": stats}[args.cmd](cfg, args.limit)


if __name__ == "__main__":
    run()
