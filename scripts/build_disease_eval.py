#!/usr/bin/env python3
"""Write a classification-only eval file, sampled by STUDY, for the per-disease
comparison against EchoSonar-R Table 1.

    python scripts/build_disease_eval.py --studies 400 --parts 2

Sampling is by study, never by question. Per-disease F1 is prevalence-sensitive,
and our test set runs 2.5% to 54.6%: capping questions per disease would rewrite
the prevalence and make every F1 incomparable to theirs. Taking whole studies
keeps each disease's prevalence exactly as it is in the full 1,215-study set.

`--parts` splits the sample into files that each fit inside one 8h SLURM wall.
Concatenating the resulting episode files reproduces the unsplit run.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from echo_verl.eval.diseases import disease_of                 # noqa: E402
from echo_rl.data.answers import parse_yes_no                  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-jsonl", default="build/eval.jsonl")
    ap.add_argument("--out-prefix", default="build/eval_disease")
    ap.add_argument("--studies", type=int, default=None,
                    help="sample this many studies (default: all 1,215, their protocol)")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in open(args.eval_jsonl) if l.strip()]
    recs = [r for r in recs if r["question_type"] == "abnormality_classification"]

    studies = sorted({r["study_uuid"] for r in recs})
    rng = random.Random(args.seed)
    rng.shuffle(studies)
    if args.studies:
        studies = studies[: args.studies]
    keep = set(studies)
    recs = [r for r in recs if r["study_uuid"] in keep]
    rng.shuffle(recs)

    pos, tot = Counter(), Counter()
    for r in recs:
        d = disease_of(r["question"])
        tot[d] += 1
        if parse_yes_no(str(r["answer"])) == "yes":
            pos[d] += 1

    print(f"{len(studies)} studies, {len(recs)} questions, {len(tot)} diseases")
    for d in sorted(tot, key=lambda x: -pos[x] / tot[x]):
        print(f"  {d:26s} n={tot[d]:5d}  positives={pos[d]:4d}  prev={100*pos[d]/tot[d]:5.1f}%")

    for i in range(args.parts):
        part = recs[i::args.parts]
        path = f"{args.out_prefix}_part{i}.jsonl" if args.parts > 1 else f"{args.out_prefix}.jsonl"
        with open(path, "w") as w:
            for r in part:
                w.write(json.dumps(r) + "\n")
        print(f"wrote {len(part)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
