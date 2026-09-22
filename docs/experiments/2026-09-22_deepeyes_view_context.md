# DeepEyes view-context comparison, 2026-09-22

Completed on iCardio: ten complete answers, ten delivered crops, no upstream
errors or truncation; total episode time 192.47 seconds. The user requested
telling the model both the view
of the supplied image and that it is a frame from the corresponding video.
This is an explicit prompt ablation relative to the no-view run, not an
unchanged reproduction of the authors' prompt.

## Fixed inputs and inference

The same ten distinct held-out studies/questions and exact PNG bytes as the
previous runs are reused. Selection seed 1, two questions per type, one random
frame from one random view per study. Training pool: 5,061 studies. Held-out
pool: 1,215 studies / 31,209 rows. None of the ten selected studies is in the
training pool. SHA256 identities are checked in the comparison report.

Model: `ChenShawn/DeepEyes-7B`, revision
`3b31e4b342592d14cb12bb8434a6ea55e9cd78c8`, at
`/home/mashrafimonon/EchoSonarVideo/checkpoints/deepeyes_7b`.
Upstream: `external/DeepEyes`, revision
`11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1`, actual `eval/eval_hrbench.py`
function. Runner commit: `adafc32`. Baseline runner commit: `7558b38`.

Both iCardio runs use shared NVIDIA RTX A6000 GPU 1, Transformers 4.57.3,
PyTorch 2.5.1+cu121, BF16, SDPA, greedy decoding, seed 1, 8,192 tokens per
call, and the upstream 11-call limit. Stop strings: `<|im_end|>` and
`</tool_call>`. The original HRBench image resizing and crop loop are retained.
The local Transformers client is an explicit backend difference from upstream
vLLM, identical across this prompt comparison. Multiple-choice options are
omitted, model tool literals use `ast.literal_eval`, and the adapter supplies
the pinned script's missing `copy` import. No training is performed.

## Prompt change

The system prompt and tool instructions stay unchanged. After the original
question, the runner adds:

```text
This image is a frame from an echocardiography video.
View: {actual recorded view label}
```

Only one frame is supplied; no other frames, motion information, reference
answers, frame indices or other study details are added. The view label comes
from that frame's existing sample metadata. Saved `question` retains the
original dataset question; `prompt_question` records the exact extended text.
This context remains in history when a crop is returned.

## Exact command

From `/home/mashrafimonon/EchoSonarVideo`:

```bash
bash scripts/run_deepeyes_tools_cuda.sh \
  --include-view-context \
  --sample-jsonl build/deepeyes_icardio_20260922/inputs/sample.jsonl \
  --revision-file build/deepeyes_icardio_20260922/inputs/model_revision.txt \
  --out-dir build/deepeyes_icardio_view_context_20260922/hrbench
```

The launcher defaults to GPU 1, `.venv_deepeyes_cuda`, offline Hugging Face,
and offline W&B. There is no online W&B link. The existing local video dataset
is available; exact input PNGs from the preceding run were reused to avoid
changing frame selection.

## Validation and artifacts

All 14 focused inference tests passed on iCardio. The real pinned HRBench
function was tested with context off/on: expected stop settings, real crop
pixels in the next turn, supplied view context, unchanged question field and
absence of the reference answer. The first recorded GPU request was separately
checked for the actual video-frame sentence and `View: Subcostal Standard`.

The [baseline report](../../build/deepeyes_icardio_20260922/report.md) contains
exact prompts, frames, crops and raw turns. The [new comparison report](../../build/deepeyes_icardio_view_context_20260922/report.md)
shows both answers for each identical frame, exact user/system prompts and all
raw turns. All 31 output/image files passed SHA256 verification; images decode
and every local report link resolves. Raw clinical artifacts
remain ignored by Git. Completion counts measure execution and output format,
not clinical accuracy; shared-GPU timing is not a throughput benchmark.


## Results and remaining output issues

| Measure | No view context | With view context |
| --- | --- | --- |
| Complete tagged answers | 10 | 10 |
| Crop observations delivered | 12 | 10 |
| Examples containing `addCriterion` | 0 | 1 |
| Upstream errors | 0 | 0 |
| Examples with a truncated generation | 0 | 0 |
| Final answers followed by a tool request | 8 | 9 |
| Responses ending with >100 whitespace characters | 1 | 0 |
| Total episode seconds | 280.34 | 192.47 |

Example 6, model call 2, emitted `addCriterion` once after its completed
`</answer>`, followed by an empty `<tool_response>` block. This was model text,
not an environment observation. The published HRBench stop list stops at
`</tool_call>` and end-of-message, not at `</answer>`. The view-context run
retains that contract. Adding a final-answer stop would be a separate protocol
change; it was not applied here.

Nine final answers still include a trailing tool request which the original
loop does not deliver to another model call. The marker's return demonstrates
that the tool-boundary stop does not universally eliminate malformed output.
All exact text remains visible in the report/JSON; answer extraction alone is
not evidence of clean interaction or clinical correctness.

Example 7's metadata view is the literal string `none`; the prompt preserves
that value. The other nine images have named view labels. No missing view was
inferred or invented. The model still receives only one frame, so statements
about function or an entire study are not validated by this experiment.
