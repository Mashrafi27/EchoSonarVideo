# Meeting prep — 2026-08-31

## The ask

Sign-off to run GRPO on this box (`sn4622121129`, 4x RTX A6000 48 GB) now,
instead of waiting on the AMD cluster. Direction is unchanged: cold-start SFT ->
GRPO, tool use as the differentiator vs EchoSonar-R, target is matching or
beating their Table 1 and Table 3.

## Where we are

- **Cold-start SFT is done.** `s5-balanced-19k`: seeded stratified sample of
  19,734 / 102,098 records over 4,028 studies, balanced by question type. Trained
  to step616, merged to a loadable HF checkpoint, evaluated.
- **We have not beaten EchoSonar-R on anything yet.** Say this up front.
- **On the comparable metric, SFT-only is below their SFT-only.** Their Table 1
  is a macro over per-disease binary F1 / balanced accuracy. Our per-disease
  prevalences match theirs to within 0.1%, so the test set is confirmed the same
  and the row is buildable.
  - EchoSonar-R: GRPO 49.4 / 67.4 | SFT-only 45.1 / 65.1 | Qwen3-VL base 19.6 / 50.3
  - Our exact macro-over-11 for step616 is in the `*-perdisease` wandb runs and
    needs to be read off and stated before the meeting.
- **SFT regressed balanced abnormality classification vs base.** Base plain
  predicts "yes" 24/100; SFT step616 predicts "yes" 5/100 with 6 unparsable.
  Plain accuracy up (0.74 -> 0.79), balanced accuracy down (0.631 -> 0.569). SFT
  learned the majority "no". This is the expected failure the balanced-accuracy
  rule exists to catch, not a surprise.
- **NLG metric code checks out.** Our BLEU-4 0.013-0.015 vs their reported 0.010
  on the base row. METEOR / BERTScore / GREEN not computed on purpose.

## Why GRPO is the fix, not more SFT

- Decision already on file: do not re-run SFT for the classification regression.
  GRPO gets a **balanced classification reward**, so the policy cannot farm 0.8
  by always answering "no".
- Tool use (`select_view`, `select_frames`, `zoom`) is our only structural
  difference from EchoSonar-R. Same dataset, same recipe otherwise. GRPO is where
  the tool policy actually gets trained; SFT only cold-starts the format.

## Why not the AMD cluster

- No served ROCm vLLM and no route to one. Two from-source builds failed
  identically on a kernel-driver ceiling (`amdgpu-dkms 6.10.5.60303` too old for
  the ROCm 2.9-series KFD ABI). A control run of the working torch build on the
  same GPU succeeded right after, so it is the driver, not us.
- The two remaining AMD options are an admin kernel update or read access to a
  colleague's private `.sif`. Both are outside what we control and neither has a
  timeline.
- SFT and evaluation still work on AMD and stay there. Only GRPO, which needs a
  live vLLM engine, moves here.

## What running GRPO here takes

1. **Data + checkpoint transfer.** No `/vast` mount on this box. The preprocessed
   data and the merged SFT checkpoint have to be copied over. Critical path.
   Candidate location: `/hdd2/ahmedaly`.
2. **Build the training env** from `requirements-train.txt` (CUDA wheels: torch
   cu129, vllm, flash-attn). Never installed from before; first install is also
   its first test.
3. **Non-SLURM launch.** This box has no scheduler. The `scripts/*.sbatch` files
   need a plain `torchrun` / `ray` entry point.
4. **A `check_train_env.py` equivalent for this box** before trusting any run,
   proving the served vLLM answers a real multi-turn multi-image tool call, not
   just that it imports. The ROCm gate exists because an earlier one went green
   while vLLM silently could not serve.
5. **Strip the ROCm workarounds** that don't apply: `sdpa` -> `flash_attention_2`,
   drop the ROCm-only sbatch traps. (FP8 is still out; A6000 is Ampere.)

## Risks to raise before the professor does

- **Does 8B GRPO fit on 4x A6000 48 GB?** FSDP-sharded policy + optimizer +
  reference model + a colocated vLLM engine, with multi-image multi-turn rollouts
  that inflate KV cache. Plausible with param/optimizer offload and a capped vLLM
  memory fraction, but it will need tuning and may force smaller rollout batches
  or shorter sequences. Not free.
- **No NVLink on A6000** — FSDP all-gather runs over PCIe, so step time will be
  slower than the AMD 8x MI210 SFT runs.
- **Initial-observation mismatch (unresolved).** SFT opens with an N-view
  thumbnail strip; the RL rollout opens with a single `<video>` over the full
  clip. Cold-start SFT is teaching an opening the rollout never shows. Needs a
  decision before GRPO: SFT adopts the clip, or RL adopts the strip.
- First GRPO run on this stack is also the first end-to-end test of the native
  verl agent-loop + echo tool on a real served engine. Budget debugging time.

## What we need from the professor

- Sign-off to commit the next stretch to standing up and running GRPO here.
- A call on the initial-observation mismatch, or agreement that we decide it and
  report back.
