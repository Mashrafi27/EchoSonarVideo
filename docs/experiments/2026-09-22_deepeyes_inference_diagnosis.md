# DeepEyes repetitive-generation diagnosis, 2026-09-22

Direct Transformers inference reproduced the repeated tool JSON and
`addCriterion` seen in vLLM. This establishes that the behavior is not specific
to vLLM. It does not establish why the released checkpoint produces that
particular identifier. The marker is generated text, absent from our prompts
and upstream tool code; repeated JSON within one response is not repeated tool
execution.
The Transformers and original vLLM outputs match exactly through the first
closing tool tag; both continue generating unwanted text after that boundary.

## Controlled reference

Job 209370 ran on one MI210 on `auh7-1b-gpu-259`. Its Transformers stage used
the exact saved first request from example 4 of run 209288, including the same
frame and system/user text. The model remained `ChenShawn/DeepEyes-7B`, revision
`3b31e4b342592d14cb12bb8434a6ea55e9cd78c8`, at
`/vast/users/mohammad.yaqub/project/EchoSonarVideo/checkpoints/deepeyes_7b`.
Configuration: BF16, Transformers 4.57.3, PyTorch 2.9.1+git8907517, SDPA,
greedy decoding, seed 1, a 512-token diagnostic cap, and the checkpoint EOS.
It received 512 input tokens and produced 396 tokens, ending at EOS. Generation
took 22.2 seconds; processor/model loading plus generation took 38.2 seconds.

The output contains one `addCriterion`, repeated tool objects, and a claim
to have inspected a zoom despite this being only the first model call. No crop
was delivered during this isolated first-turn diagnostic. Full token IDs,
rendered prompt, image tensor hash, generation configuration and output are in
the [local diagnostic output](../../build/deepeyes_inference_check_20260922/transformers_gpu.json).

The subsequent vLLM stage failed before generation: calling `/tokenize` first
populated the sender's image cache without supplying that image to the engine,
which raised `Expected a cached item for mm_hash`. The whole job therefore
has exit 1, despite the completed Transformers result. The diagnostic now calls
the tokenizer endpoint only after generation. This new diagnostic failure did
not occur in the original ten-example run and does not explain its repetition.

## Completed AMD protocol comparison

The author's pinned V* demo stops only at `<|im_end|>`. Its loop waits for a
whole response, parses the first tool block and sends the entire response,
including repetition, back with the crop. The published
[HRBench demo](https://github.com/Visual-Agent/DeepEyes/blob/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/eval/eval_hrbench.py)
also stops at `</tool_call>`, uses 8,192 tokens per call instead of 10,240, and
resizes the initial frame. The API omits a matched stop string from returned
content; the upstream parser supports the resulting tool block without its
closing tag. A missing `copy` import in that pinned script is supplied by our
adapter, preserving the upstream file itself.

Job **209392 completed successfully in 3m13s** before an attempted cancellation
when moving work to iCardio. It first replayed the same saved request twice, changing only the tool stop string, then
ran the actual HRBench function on the same ten-study sample as 209288:
seed 1, two examples per question type, ten distinct held-out studies, no overlap
with 5,061 training studies. Held-out pool: 31,209 rows / 1,215 studies. No
training, reference answers or view labels are added to prompts.

Exact launch from the AMD repository root:

```bash
sbatch --parsable --nodelist=auh7-1b-gpu-259 \
  --export=ALL,ECHO_CHECK_SKIP_TRANSFORMERS=1,ECHO_CHECK_RUN_HRBENCH=1 \
  scripts/run_deepeyes_check.sbatch
```

Source HEAD: `55af99e249356af82d4c2b08b0651ae210fc810b`, with uncommitted
diagnostic and protocol-adapter changes. Model and upstream pins are unchanged.
The HRBench run records exact source snapshots, SHA256 values, runtime versions,
CLI, settings, traces and summaries under
`build/deepeyes_baseline_20260922/check_209392/hrbench/` on AMD. W&B is offline;
there is no online link. All ten examples returned complete tagged answers;
12 crop observations were delivered. There were zero upstream errors, zero
truncations, and zero examples containing `addCriterion`. The 10 episodes took
112.61 seconds of inference. The controlled vLLM replay (512-token cap) returned
512 tokens with the old stop list, including one `addCriterion`; adding the
tool boundary stopped at 112 tokens with no marker and an identical prefix.
These counts verify
execution and output format, not clinical correctness. Eight final responses contain an answer followed by another tool request.
The upstream loop terminates after seeing the answer, so these trailing requests
are not delivered in another model turn. The second full-report answer also
ends with 1,621 whitespace characters. Thus output structure is still imperfect.

The [local report with frames and crops](../../build/deepeyes_hrbench_20260922/report.md)
is complete. The downloader verified 33 output/image files; image decoding and
all report links passed. Raw outputs remain ignored by Git.

## iCardio cross-check

The same ten exact input PNGs were copied to iCardio for a controlled comparison;
the local video dataset was available, but no frames were reselected. Frame
SHA256 values and membership in the same 1,215 held-out studies, excluding all
5,061 training studies, were verified. No view labels or references are prompted.

Code commit `7558b38` was pushed and fast-forward pulled into
`/home/mashrafimonon/EchoSonarVideo`. An isolated `.venv_deepeyes_cuda` inherits
read-only packages from the existing environment; PyTorch 2.5.1+cu121,
Transformers 4.57.3, BF16, SDPA, greedy decoding, seed 1. The pinned checkpoint
is at `checkpoints/deepeyes_7b`. Inference uses shared RTX A6000 GPU 1. The
Transformers client applies stop strings during generation and omits the
matched stop text to match the API response contract. The original HRBench
function still owns prompts, resizing, cropping and turn handling. This is a
backend adaptation, not the authors' complete vLLM environment.

The NVIDIA same-request control has a 512-token diagnostic cap. With the V*
stop list it generated 512 tokens and `addCriterion`; with the extra tool
boundary it generated 122 tokens, without that marker. Outputs match exactly
up to the first `</tool_call>` boundary. Generation took 23.74 and 5.59 seconds,
respectively. This reproduces the behavior outside AMD and isolates the effect
of stopping for that first turn. It does not explain why that identifier was
learned. The full ten-example HRBench cross-check completed in 280.34 seconds of
episode time: all ten returned complete tagged answers, with 12 delivered crops,
zero `addCriterion`, zero truncations and zero upstream errors. The second
full-report question was longest: 123.3 seconds across three calls, including
a 2,866-token final response. Eight final responses included trailing tool
requests after the answer, as in the AMD batch. The second full-report answer
ends with 1,345 whitespace characters. These are retained in exact outputs
and explicitly counted where the Markdown display trims trailing whitespace.

The [iCardio report](../../build/deepeyes_icardio_20260922/report.md) includes
both exact prompts, all ten original and resized frames, crops, references,
answers and raw turns. All 33 output/image files passed checksum validation,
with successful image decoding and link checks.

Exact batch command, from the iCardio repository root:

```bash
bash scripts/run_deepeyes_tools_cuda.sh \
  --sample-jsonl build/deepeyes_icardio_20260922/inputs/sample.jsonl \
  --revision-file build/deepeyes_icardio_20260922/inputs/model_revision.txt \
  --out-dir build/deepeyes_icardio_20260922/hrbench
```

The launcher selects GPU 1 unless `CUDA_VISIBLE_DEVICES` is explicitly supplied.
W&B remains offline, with no online link. The bounded control was run first
using the retained operator script
`build/deepeyes_icardio_20260922/run_control.py`; its request is the exact saved
first turn from example 4 of the earlier V* batch.

[Exact prompts and settings](../../build/deepeyes_icardio_20260922/prompts.md) ·
[Controlled outputs](../../build/deepeyes_icardio_20260922/control.json).

## Other diagnostics and validation

GPU job 209359 failed before inference on node 190 (`No HIP GPUs are available`).
CPU-only probe 209362 read the legacy `/dev/kfd` mapping on nodes 185/212/234;
no permissions or scheduler configuration were changed. Node 219 has an active
maintenance reservation and cannot accept new jobs.

CPU job 209371 used the same saved request, FP32 and a 256-token diagnostic cap.
It was cancelled at 15m35s after GPU Transformers reproduced the issue. It did
not produce a final generation result and must not be cited as confirmation.

All 13 focused baseline/tool-loop tests passed on AMD. The added test calls
the real pinned HRBench function with a fake server, verifies its stop list and
8,192-token limit, checks actual crop pixels on the next call, and confirms the
reference is never prompted. Python syntax, shell syntax and diff whitespace
checks passed. Real AMD HRBench GPU validation passed as recorded above. The same 13 tests
also passed on iCardio before its GPU run.
