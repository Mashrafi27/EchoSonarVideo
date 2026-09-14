"""Sample exactly one QA pair per study from build/rl.jsonl, for the
"one study = one prompt per epoch" training scheme.

Rationale: build/rl.jsonl has ~25 QA pairs per study on average (128,215 pairs /
5,061 train studies), so a naive full pass is 4,007 steps at batch 32 -- neither
reference paper (EchoSonar-R, DeepEyes) trains on that scale; both use a bounded
iteration count over a curated/reduced set. This script gives each study exactly
one vote per "epoch": ~5,061 prompts -> ~158 steps at batch 32, matching the step
budget we already independently landed on.

Usage, one call per epoch, DIFFERENT --seed each time so the same study surfaces
a different question on the next pass (echoing EchoSonar-R's own view-sampling
design: "one clip per unique view is randomly selected per study, providing
implicit data augmentation across epochs", Table S1 note):

    python scripts/sample_one_qa_per_study.py --seed 0 --out build/rl_epoch0.jsonl
    python packages/verl_bridge/generate_trainset.py --rl-jsonl build/rl_epoch0.jsonl \\
        --out build/rl_train_epoch0.parquet

Then point run_grpo.sh's TRAIN_FILES at build/rl_train_epoch0.parquet, and repeat
with --seed 1, --seed 2, ... for subsequent epochs, resuming from the previous
epoch's checkpoint via CKPT_HOME / trainer.resume_from_path.

Not stratified by question_type on purpose -- the request was "one QA pair per
study", not "one QA pair per (study, question_type)". Structure Description and
Abnormality Classification are ~88% of the corpus (Table S1), so over many epochs
those types will still dominate the OVERALL exposure, same skew the full dataset
already has. If that turns out to matter, stratify by question_type per study
instead of picking uniformly across all of that study's records.
"""
import argparse
import collections
import json
import random


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-jsonl", default="build/rl.jsonl",
                     help="full QA-pair jsonl to sample from (data_core.cli build-rl output)")
    ap.add_argument("--out", required=True,
                     help="where to write the one-per-study jsonl, e.g. build/rl_epoch0.jsonl")
    ap.add_argument("--seed", type=int, required=True,
                     help="different seed -> different QA pair per study; keep a record of "
                          "which seed built which epoch's file")
    args = ap.parse_args(argv)

    by_study = collections.defaultdict(list)
    with open(args.rl_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_study[rec["study_uuid"]].append(line)

    rng = random.Random(args.seed)
    type_counts = collections.Counter()
    with open(args.out, "w") as out:
        # sorted() over the dict keys so the same seed always produces the same
        # sample regardless of jsonl line order / dict iteration order
        for study_uuid in sorted(by_study):
            lines = by_study[study_uuid]
            chosen = rng.choice(lines)
            out.write(chosen + "\n")
            type_counts[json.loads(chosen)["question_type"]] += 1

    print(f"{len(by_study)} studies -> {args.out} (seed={args.seed})")
    print("question_type distribution of the sample:")
    for qtype, n in type_counts.most_common():
        print(f"  {qtype}: {n} ({100 * n / len(by_study):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
