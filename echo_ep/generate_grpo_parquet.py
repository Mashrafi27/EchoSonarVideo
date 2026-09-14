"""GRPO parquet builder for the frozen-EchoPrime + Qwen3-8B-text track.

Reuses `build/rl.jsonl` (already built by `echo_rl build-rl`, same file `echo_verl.generate_
trainset` reads for the tool-based track) directly -- it already carries `study_uuid`,
`question`, `question_type`, and a correctly-shaped `reward_key`
({kind, target, gold, is_abnormal}, from `echo_rl.data.builders._reward_key`), which is exactly
what `echo_verl.reward.compute_score` -> `echo_rl.reward.score.total_reward` expect. Only the
row SHAPE differs from `echo_verl.generate_trainset.build_row`: no tools, no 19-view overview
image menu -- the `prompt` field is built here directly (system + user, `VIEW_TOKEN * n_views`
embedded as plain text, matching `echo_ep.dataset.SYSTEM_PROMPT`'s convention), and `agent_name`
points at `echo_ep.echoprime_agent_loop.EchoPrimeAgentLoop` (registered as "echoprime_agent"),
which reads `extra_info.study_uuid` directly at rollout time -- no per-QA-pair image/video
columns needed at all, unlike the tool-based track's 19-view overview menu.

`n_views` has to match the CACHED study's real view count exactly (asserted again at rollout
time in echo_ep/echoprime_agent_loop.py) -- read once per row from the cache file itself, not
guessed, so the parquet and the cache never disagree even if a study's cache changes later.
"""
import json

from echo_ep.dataset import SYSTEM_PROMPT
from echo_ep.modeling import VIEW_TOKEN

_DATA_SOURCE = "echoprime_grpo"
_AGENT_NAME = "echoprime_agent"


def build_row(rl_rec: dict, n_views: int) -> dict:
    user_turn = f"{VIEW_TOKEN * n_views}\n{rl_rec['question']}"
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": _AGENT_NAME,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_turn}],
        "reward_model": {"ground_truth": json.dumps(rl_rec["reward_key"]), "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {"study_uuid": rl_rec["study_uuid"],
                        "question_type": rl_rec["question_type"],
                        "need_tools_kwargs": False},
    }


def write_parquet(rows: list, path: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)


def main(argv=None) -> int:
    import argparse
    import glob
    import os

    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-jsonl", default="build/rl.jsonl")
    ap.add_argument("--study-list", default=None,
                     help="only include rows for these study_uuids (one per line) -- e.g. "
                          "build/train_study_ids.txt for train, unset for no filtering")
    ap.add_argument("--out", default="build/echoprime_grpo_train.parquet")
    ap.add_argument("--cache-dir", default="build/echoprime_video_cache",
                     help="only include rows whose study is actually cached")
    ap.add_argument("--max-views", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                     help="shuffle rows before applying --limit, so a capped sample is a real "
                          "random sample instead of just whatever rl.jsonl happens to list "
                          "first (confirmed this session: the original 200/20-row parquets "
                          "were an uncontrolled --limit cutoff over file order, not a real "
                          "sample, and additionally never filtered by rl.jsonl's own train/val "
                          "split field at all -- see docs/OPEN_ISSUES.md)")
    ap.add_argument("--one-per-study", action="store_true",
                     help="exactly one randomly-chosen QA pair per study, covering every "
                          "study in --rl-jsonl (after --study-list/cache filtering) -- matches "
                          "the project's own definition of one epoch: one pass over all "
                          "training videos, one sampled QA pair per video. Requires "
                          "--shuffle-seed (the 'randomly-chosen' part).")
    args = ap.parse_args(argv)

    if args.one_per_study and args.shuffle_seed is None:
        ap.error("--one-per-study requires --shuffle-seed")

    allowed = None
    if args.study_list:
        with open(args.study_list) as fh:
            allowed = {line.strip() for line in fh if line.strip()}

    cache_files = {os.path.basename(p)[:-3]: p
                   for p in glob.glob(os.path.join(args.cache_dir, "*.pt"))}
    view_counts = {}

    candidates = []
    skipped_split, skipped_cache = 0, 0
    with open(args.rl_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            study_uuid = rec["study_uuid"]
            if allowed is not None and study_uuid not in allowed:
                skipped_split += 1
                continue
            if study_uuid not in cache_files:
                skipped_cache += 1
                continue
            candidates.append(rec)

    if args.one_per_study:
        import random
        rng = random.Random(args.shuffle_seed)
        by_study = {}
        for rec in candidates:
            by_study.setdefault(rec["study_uuid"], []).append(rec)
        candidates = [rng.choice(qs) for qs in by_study.values()]
        rng.shuffle(candidates)
    elif args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(candidates)

    rows = []
    for rec in candidates:
        study_uuid = rec["study_uuid"]
        if study_uuid not in view_counts:
            cache = torch.load(cache_files[study_uuid])
            view_counts[study_uuid] = min(cache["embeddings"].shape[0], args.max_views)
        rows.append(build_row(rec, view_counts[study_uuid]))
        if args.limit is not None and len(rows) >= args.limit:
            break

    write_parquet(rows, args.out)
    print(f"[echo_ep] wrote {len(rows)} rows -> {args.out} "
          f"(skipped {skipped_split} not-in-split, {skipped_cache} not-yet-cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
