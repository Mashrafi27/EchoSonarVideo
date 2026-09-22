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

## Protocol comparison prepared

The author's pinned V* demo stops only at `<|im_end|>`. Its loop waits for a
whole response, parses the first tool block and sends the entire response,
including repetition, back with the crop. The published
[HRBench demo](https://github.com/Visual-Agent/DeepEyes/blob/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/eval/eval_hrbench.py)
also stops at `</tool_call>`, uses 8,192 tokens per call instead of 10,240, and
resizes the initial frame. The API omits a matched stop string from returned
content; the upstream parser supports the resulting tool block without its
closing tag. A missing `copy` import in that pinned script is supplied by our
adapter, preserving the upstream file itself.

Job **209392 is queued; its inference results are not yet available**. It first
replays the same saved request twice, changing only the tool stop string, then
runs the actual HRBench function on the same ten-study sample as 209288:
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
there is no online link. The local watcher will produce the image report in
`build/deepeyes_hrbench_20260922/` after completion, with checksum, image and
link validation. Its [status file](../../build/deepeyes_hrbench_20260922/watcher_status.json)
records monitoring progress. Generated outputs remain ignored by Git.

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
checks passed. Real HRBench GPU validation remains pending.
