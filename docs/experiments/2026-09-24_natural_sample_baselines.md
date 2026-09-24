# Released visual-tool models on natural samples, 2026-09-24

Quick inference check of the four released checkpoints on ordinary photos and
short clips, to see how they behave outside echo. Ten sample photographs with
one hand-written detail question each went to DeepEyes, Chain-of-Focus and
Mini-o3. Three ten-second sample clips went to Video-CoM. No benchmark dataset
was used. Expected answers were written by hand after viewing each file, are
shown only for inspection and were not verified by anyone else. This is an
execution and sanity check, not an accuracy result.

## Samples

Photos are stable Picsum IDs 1, 26, 22, 40, 48, 60, 96, 106, 119 and 133
(Unsplash photographs, served at 1600x1067). Clips are the 720p ten-second
Big Buck Bunny, Jellyfish and Sintel samples from test-videos.co.uk. Questions,
expected answers, source URLs and licences are in
`packages/eval/natural_samples_20260924.json`. Downloaded files, prepared inputs,
annotated clips, run traces and reports live under ignored
`build/natural_baselines_20260924/` locally and on iCardio.

Clip inputs: 16 frames spread evenly over each clip from the original
preprocessing helper's annotated video (300, 300 and 240 source frames at 30,
29.97 and 24 fps; initial frames passed at 1.6 fps). Nothing was skipped.

## Checkpoints, code and runtime

| Model | Released checkpoint | Weight revision | Source commit |
| --- | --- | --- | --- |
| DeepEyes | `ChenShawn/DeepEyes-7B` | `3b31e4b342592d14cb12bb8434a6ea55e9cd78c8` | `11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1` |
| Chain-of-Focus | `xintongzhang/CoF-rl-model-7b` | `7c34733a3b333302ea9e3f320e2be1334e471bd5` | `431ff0dd703f83d8a624ec1ba1d2ef6a834b78bc` |
| Mini-o3 | `Mini-o3/Mini-o3-7B-v1` | `e82da9a10cca125a1e26c27760ef80840c0fbcd1` | `2c5a0dedb5279eff2c0e6049aac05de97bf7a2b3` |
| Video-CoM | `MBZUAI/Video-CoM` | `16fc4301f645b528d77faeb285c0c88eedc76643` | `aa07959e6c1be94caca9ca0dd4e65a2051077114` |

Same checkpoints, pinned upstream clones and adapters as the
[echo baseline run](2026-09-22_visual_tool_baselines.md) and the
[DeepEyes view-context run](2026-09-22_deepeyes_view_context.md). The only
protocol change is the new `--domain natural` switch (commit `7de9ca3`): the
plain question is sent with no echo framing, no view label and no held-out
checks. DeepEyes uses its pinned HRBench loop (stop at `<|im_end|>` and
`</tool_call>`, 8,192 tokens per call, at most 11 calls, no multiple-choice
options). Video-CoM's clip prompt says "These are 16 frames sampled evenly from
one short video clip." All other prompt text is unchanged.

All runs used iCardio RTX A6000 GPUs 1 to 3 in parallel, the isolated
`.venv_deepeyes_cuda` environment (PyTorch 2.5.1+cu121, Transformers 4.57.3,
BF16, SDPA), greedy decoding, offline W&B. Timing is from a shared machine.

## Results

| Model | Completed answers / attempted | Model calls | Delivered tool observations | Seconds |
| --- | --- | --- | --- | --- |
| DeepEyes | 10/10 | 20 | 10 | 92.3 |
| Chain-of-Focus | 10/10 | 12 | 2 | 49.5 |
| Mini-o3 | 9/10 | 31 | 21 | 147.2 |
| Video-CoM | 3/3 | 7 | 4 | 17.3 |

No `addCriterion`, no token truncation, no runtime errors in any run.

DeepEyes zoomed exactly once on every photo, then answered. All ten final
responses still end with an undelivered `<tool_call>` after `</answer>`, the
same artifact seen on echo frames. Chain-of-Focus answered eight photos
directly and zoomed once on two (calendar, cars). Mini-o3 zoomed on nine of
ten photos; on the flowers it repeated the same crop of observation 2 eleven
times with near-identical reasoning until the twelve-call limit, without ever
writing the answer its own reasoning had reached ("centers appear to be
yellow"). Video-CoM called `FIND_FRAME` on each clip and `SPATIAL_ZOOM` once
(jellyfish watermark).

### Photo answers against the hand-written expectation

| # | Question (short) | Expected | DeepEyes | Chain-of-Focus | Mini-o3 |
| --- | --- | --- | --- | --- | --- |
| 1 | Laptop model name | MacBook Air | MacBook Air | MacBook Air | MacBook Air |
| 2 | Colour of object below pen | dark red | red | red | red |
| 3 | Shoe colour | tan/brown | brown | brown | brown |
| 4 | Animal and nose colour | cat, pinkish brown | cat, pink with black tip | cat, pink | cat, pink |
| 5 | Object right of laptop | black smartphone | smartphone | smartphone | smartphone |
| 6 | Object right of glasses | dark red notebook | notebook | notebook | notebook |
| 7 | Brand on controller | SONY | Xbox | XBOX | SONY |
| 8 | Flower and centre colour | pink, yellow-orange | pink, darker pink/maroon | pink, reddish-brown | no answer (turn limit) |
| 9 | Calendar number and day | 18, Thu | 18, Thursday | 18, Thursday | 18, Thursday |
| 10 | Round lights on right car | six, outer ones orange | four, white | four, red | five, white |

By my reading, questions 1 to 6 and 9 are answered acceptably by all three.
Question 7 is a clear miss for DeepEyes and Chain-of-Focus: the controller
says SONY, DeepEyes zoomed on the label and still said Xbox. Question 8
centres are read as dark pink or reddish-brown rather than yellow. Question 10
disagrees with my own count, which is itself uncertain; treat it as
unresolved rather than as a model error.

### Clip answers (Video-CoM)

| # | Question (short) | Expected | Video-CoM |
| --- | --- | --- | --- |
| 1 | What is at the base of the tree | dark burrow entrance with rocks | hole |
| 2 | Website in bottom-right corner | http://jell.yfish.us | http://jellyfish.us |
| 3 | What the red-haired character holds | long wooden spear or staff | sword |

The watermark read is close but not exact after a spatial zoom on a 720p
frame. "Sword" versus "spear" is a judgment call on blurred frames.

## Reports and provenance

- [Comparison](../../build/natural_baselines_20260924/report.md) with every
  answer, and per-model reports with exact prompts, input images, crops and
  raw turns: [DeepEyes](../../build/natural_baselines_20260924/deepeyes_run/report.md),
  [Chain-of-Focus](../../build/natural_baselines_20260924/chain_of_focus_run/report.md),
  [Mini-o3](../../build/natural_baselines_20260924/mini_o3_run/report.md),
  [Video-CoM](../../build/natural_baselines_20260924/video_com_run/report.md).
- Each run's `metadata.json` records the command, Git commit `7de9ca3`,
  model path and revision, runtime versions and offline W&B directory. No
  public W&B link exists.

Commands, from `/home/mashrafimonon/EchoSonarVideo` with `R=build/natural_baselines_20260924`:

```bash
PYTHONPATH=packages .venv_deepeyes_cuda/bin/python -m eval.prepare_natural_inputs --images-dir $R/assets/images --videos-dir $R/assets/videos --upstream build/visual_baselines_20260922/video_com --out-dir $R/inputs
CUDA_VISIBLE_DEVICES=1 bash scripts/run_visual_tool_baseline_cuda.sh --model chain_of_focus --domain natural --sample-jsonl $R/inputs/sample.jsonl --out-dir $R/chain_of_focus_run
CUDA_VISIBLE_DEVICES=1 bash scripts/run_visual_tool_baseline_cuda.sh --model video_com --domain natural --sample-jsonl $R/inputs/video_sample.jsonl --video-manifest $R/inputs/video_inputs/manifest.json --out-dir $R/video_com_run
CUDA_VISIBLE_DEVICES=2 bash scripts/run_visual_tool_baseline_cuda.sh --model mini_o3 --domain natural --sample-jsonl $R/inputs/sample.jsonl --out-dir $R/mini_o3_run
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=packages HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline .venv_deepeyes_cuda/bin/python packages/eval/run_deepeyes_tools.py --backend transformers --protocol hrbench --domain natural --sample-jsonl $R/inputs/sample.jsonl --revision-file build/deepeyes_icardio_20260922/inputs/model_revision.txt --out-dir $R/deepeyes_run
PYTHONPATH=packages .venv_deepeyes_cuda/bin/python -m eval.report_visual_tool_baselines $R --comparison
```

Thirteen focused tests passed on iCardio before launch
(`packages/eval/tests/test_visual_tool_baselines.py`,
`packages/eval/tests/test_deepeyes_tools.py`), including new checks that the
natural domain sends the plain question without echo context.
