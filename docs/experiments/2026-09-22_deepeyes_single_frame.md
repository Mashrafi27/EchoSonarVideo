# DeepEyes single-frame inference, 2026-09-22

The user requested one question with one random frame from one view, solely to
check inference. Slurm job 209262 completed with exit 0 on `auh7-1b-gpu-259`,
using one AMD Instinct MI210. Total job time was 1m17s; generation took 14.28s.
It produced 76 tokens from 215 input tokens, ended at EOS, and neither truncated
nor requested a tool. No training or aggregate evaluation was run.

## Model and inputs

- Model: `ChenShawn/DeepEyes-7B`, revision
  `3b31e4b342592d14cb12bb8434a6ea55e9cd78c8`, loaded from AMD
  `/vast/users/mohammad.yaqub/project/EchoSonarVideo/checkpoints/deepeyes_7b`.
- Source commit: `55af99e249356af82d4c2b08b0651ae210fc810b`, with uncommitted
  single-frame and runtime changes. Exact runner SHA256:
  `dbe751d35940f9e5ad546a57be0bd0483f78d39187bc9e6ff3906f803690ba32`.
  Source copies are retained in the run's `source/` directory.
- Dataset: one `structure_description` QA pair from `build/eval.jsonl`
  (31,209 rows / 1,215 studies), with zero study overlap against
  `build/rl.jsonl` (5,061 studies). Seed 0 selects the question, then a view,
  then a frame from that clip's PNG files, rather than its fixed preview.
- Selected input: one A3C view, frame index 13 (zero-based) of 18 frames.
  The question asks about the right atrium. The generated answer and reference
  both describe normal dimensions; this does not establish visual grounding or
  clinical correctness from the randomly chosen frame.
- Generation: greedy, maximum 2,048 new tokens, image cap 200,704 pixels,
  fast processor, BF16, SDPA, Transformers 4.57.3, PyTorch 2.9.1+git8907517.

## Exact launch

From `/vast/users/mohammad.yaqub/project/EchoSonarVideo`:

```bash
sbatch --parsable --job-name=deepeyes_single_frame --time=00:15:00 \
  --nodelist=auh7-1b-gpu-247 scripts/run_deepeyes_baseline.sbatch \
  --single-frame --seed 0 --max-new-tokens 2048
scontrol update JobId=209262 NodeList=auh7-1b-gpu-259
```

The pending job was moved to node 259 after node 247 had an eight-hour estimated
wait. No extra inference job was submitted. The launcher's W&B mode was offline.

## Artifacts and validation

AMD run directory:
`/vast/users/mohammad.yaqub/project/EchoSonarVideo/build/deepeyes_baseline_20260922/run_209262/`.
`sample.jsonl`, `predictions.jsonl`, and `metadata.json` retain the selected
question, reference, exact image path/hash, prompt, raw output, completion
status, settings, and dataset hashes. Raw records and the image remain on AMD.

W&B was recorded locally in
`wandb/offline-run-20260922_111200-2ktf2xgq/` under that directory; no online link
exists. Automatic approval review rejected online logging and an optional local
image copy, so neither transfer occurred.

All eight focused baseline tests passed on AMD. Input/image validation passed,
GPU arithmetic passed, and the real model load and inference completed.
This check establishes that the released model accepts one of our images and
questions and returns an answer. The next research step is to inspect the
example before selecting a larger evaluation.
