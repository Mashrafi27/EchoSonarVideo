# Chain-of-Focus, Mini-o3 and Video-CoM inference, 2026-09-22

The requested comparison reuses the ten question/view pairs from the DeepEyes
view-context run. It checks released-checkpoint inference and real tool use;
it does not establish clinical accuracy. No training or external judge is used.

## Checkpoints and source

| Model | Released checkpoint | Weight revision | Source commit |
| --- | --- | --- | --- |
| Chain-of-Focus | `xintongzhang/CoF-rl-model-7b` | `7c34733a3b333302ea9e3f320e2be1334e471bd5` | `431ff0dd703f83d8a624ec1ba1d2ef6a834b78bc` |
| Mini-o3 | `Mini-o3/Mini-o3-7B-v1` | `e82da9a10cca125a1e26c27760ef80840c0fbcd1` | `2c5a0dedb5279eff2c0e6049aac05de97bf7a2b3` |
| Video-CoM | `MBZUAI/Video-CoM` | `16fc4301f645b528d77faeb285c0c88eedc76643` | `aa07959e6c1be94caca9ca0dd4e65a2051077114` |

Official sources: [Chain-of-Focus](https://github.com/xtong-zhang/Chain-of-Focus),
[Mini-o3](https://github.com/Mini-o3/Mini-o3),
[Video-CoM](https://github.com/mbzuai-oryx/Video-CoM).
Downloaded checkpoints live at `checkpoints/chain_of_focus`, `checkpoints/mini_o3`
and `checkpoints/video_com` on iCardio. Pinned upstream clones are ignored artifacts
under `build/visual_baselines_20260922/`; their tracked source remains unchanged.

## Data and runtime

The sample is `build/deepeyes_icardio_20260922/inputs/sample.jsonl`: ten distinct
held-out studies, two questions per type, selection seed 1. The runner verifies
membership in `build/eval.jsonl` and exclusion from `build/rl.jsonl`: 1,215 test
studies / 31,209 test rows; 5,061 train studies / 128,215 train rows. The legacy
per-record `split` field is not used. One literal view label is `none`.

Chain-of-Focus and Mini-o3 receive the same original frame as DeepEyes, their
own preprocessing, the question, and this added text:

```text
This image is a frame from an echocardiography video.
View: {recorded view label}
```

All inference runs sequentially on iCardio RTX A6000 GPU 1 using the isolated
`.venv_deepeyes_cuda` environment: PyTorch 2.5.1+cu121, Transformers 4.57.3,
BF16 and SDPA. This serving backend differs from the authors' environments;
exact numerical reproduction is not claimed. W&B is offline only, with paths
and IDs recorded in each run's metadata. No public W&B link exists.

## Protocols

Chain-of-Focus calls the original `eval/vstar/vllm_inference.py` loop, replacing
only its vLLM serving interface. It retains its system/user instructions,
six-call limit, 512 new tokens per call, greedy decoding, EOS boundary,
minimum resolution 112 squared, crop enlargement 1.5 and scale-up factor 2.
The release labels exhausted loops `success`; the report separately requires
an actual completed answer. Invalid bounding boxes can return a full-frame
fallback, which is identified separately from valid crop requests.

Mini-o3 uses a standalone adapter for the released async rollout, its exact
system/observation prompts, crop helpers and resizing function. It uses relative
coordinates, original-image/prior-observation selection, 12 calls/images,
8,192 new tokens per call, 32,768-token context with the released 2,000-token
reserve, and the `</grounding>` stop retained in output. Greedy Avg@1 is an
explicitly documented evaluation option. Python literal parsing replaces
upstream `eval`; model-generated code is never executed. The upstream decoded
token ceiling of 151664 is preserved. Conversation text is re-rendered through
the Hugging Face chat template rather than appended through vLLM token IDs.
All eleven follow-up prompts from the first completed episode were checked
against the original append protocol: exact text and token IDs before image-token
expansion matched. This checks conversation boundaries, not backend numerical parity.

The first Mini-o3 attempt inherited `use_cache=False` from the checkpoint.
Growing per-turn latency led to interrupting that attempt and preserving its
partial trace in `mini_o3_run`. The comparison run, `mini_o3_cached_run`, enables
the inference KV cache. This is explicit in each recorded request and metadata.
The eight completed uncached first-example responses exactly match the cached
replay. Those eight calls took 384.46 seconds uncached and 46.48 seconds cached;
the full cached 12-call episode took 72.57 seconds. A ninth interrupted call
is excluded from the text comparison. These are timings on a shared GPU.

Video-CoM uses the original vision and manipulation helpers with a reconstructed
standalone open-ended instruction. No released standalone evaluation prompt
was found; the paper's dataset-generation prompt is not an inference prompt.
This is therefore an adapted inference test, not an exact reproduction of
the authors' evaluation. It uses `FIND_SEGMENT`, `FIND_FRAME`, `SPATIAL_ZOOM`,
five calls and 512 new tokens per call. Greedy decoding is our comparison choice.
The multiple-choice fallback is replaced by an open-ended final-answer request;
training-only dummy calls after a final answer are omitted.

## Video sampling

The requested 16 distinct frames at stride 2 require at least 31 source frames.
A seeded uniform valid start is selected anywhere in each chosen view's full
recording; indices are `start, start+2, ..., start+30`. The manifest records
every index and source hash. Six recordings qualify. Examples 1, 4, 5 and 10
have 22, 24, 24 and 25 frames and are skipped under the strict-stride policy.
No frames are repeated or padded to conceal the shortage.

Source recordings are ordered PNG directories, located through
`/hdd2/ahmedaly/echo_dicom_index.json`; acquisition FPS is unavailable. A nominal
2-FPS FFV1 container enables the author's segment/frame annotation helper, which
encodes its output with mp4v. Nominal time is a processing convention, not a
claim about cardiac timing. The prompt says acquisition timing is unavailable.
The initial 16 images are passed as an explicit video-frame list at effective
nominal FPS 1.0. The processor was verified to retain all 16 frames (temporal
grid length 8 with temporal patch size 2). Subsequent tools can request more
evidence from the same full recording. Released segment code returns 16 frames.

## Commands and provenance

Run from `/home/mashrafimonon/EchoSonarVideo`:

```bash
bash scripts/run_visual_tool_baseline_cuda.sh --model chain_of_focus --out-dir build/visual_baselines_20260922/chain_of_focus_run
bash scripts/run_visual_tool_baseline_cuda.sh --model mini_o3 --out-dir build/visual_baselines_20260922/mini_o3_cached_run
PYTHONPATH=packages .venv_deepeyes_cuda/bin/python -m eval.prepare_video_com_inputs --source-sample build/visual_baselines_20260922/source_sample.jsonl --dicom-index /hdd2/ahmedaly/echo_dicom_index.json --upstream build/visual_baselines_20260922/video_com --out-dir build/visual_baselines_20260922/video_inputs --seed 1
bash scripts/run_visual_tool_baseline_cuda.sh --model video_com --video-manifest build/visual_baselines_20260922/video_inputs/manifest.json --out-dir build/visual_baselines_20260922/video_com_run
```

Chain-of-Focus executed at `eebd374`; Video-CoM preparation was added at
`2c097aa`; Mini-o3's cache correction executed at `eb7b761`. Exact Git commits,
runner source hashes, commands and model revisions are retained in metadata.
The report/audit implementation and real tool regression checks were added at
`41627ce`. Raw logs, predictions, images and checkpoints remain ignored.

## Results and validation

Chain-of-Focus completed all ten attempts in 137.61 seconds: eight completed
answers, 32 model calls, 22 delivered follow-up observations, three invalid
bounding-box observations, two episodes ending at the six-call limit, no token
truncation and no `addCriterion`. These are execution counts, not accuracy.
The original loop's zero error statuses do not erase its two incomplete episodes.

Mini-o3 completed ten attempts in 947.69 seconds: four completed answers,
81 model calls and 71 delivered crops. Five examples exhausted the twelve-call
limit; example 6 generated 8,192 tokens on its final call without a complete
answer. There were 41 repeated identical responses across episodes, no
`addCriterion`, and no runtime exceptions. Its first example's cache and prompt
checks are described above.

Video-CoM completed six eligible attempts in 132.06 seconds: five final answers,
16 model calls, six delivered `FIND_FRAME` observations and one delivered
`SPATIAL_ZOOM` crop. Example 8 made one frame selection, then repeated text
about missing patient identity across four 512-token truncations and exhausted
the five-call limit. Three follow-ups therefore contained only text, not tool
media. No `FIND_SEGMENT` request occurred. There were no runtime exceptions or
`addCriterion` markers. Four selected recordings were skipped for insufficient
frames. Every attempted first call retained 16 frames, independently checked
in both the recorded frame list and temporal grid. It executed at `41627ce`.

All three requested runs are complete; no baseline inference process remains
running. W&B remains offline. No clinical metrics were calculated.

The initial runner metadata's `errors` field counts all non-success statuses,
including turn limits and skipped inputs. Use the report's audited completion,
runtime-error and skip counts rather than interpreting that field as crashes.

[Local combined comparison and all three reports](../../build/visual_baselines_20260922/report.md).
[Local Video-CoM report with all sampled frames and tool observations](../../build/visual_baselines_20260922/video_com_run/report.md).
[Local Chain-of-Focus report with exact prompts, frames and turns](../../build/visual_baselines_20260922/chain_of_focus_run/report.md).
[Local Mini-o3 report with exact prompts, crops and turns](../../build/visual_baselines_20260922/mini_o3_cached_run/report.md).
Reports verify image hashes and decoding, and retain full requests and raw text.
All ten focused tests passed on iCardio, covering original crop behavior,
Mini-o3 recursive cropping and safe error handling, stride sampling, real
Video-CoM frame selection/cropping, and existing DeepEyes tool behavior.
The source trees, shared environments and unrelated GPU processes were preserved.
