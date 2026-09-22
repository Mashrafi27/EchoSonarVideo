# DeepEyes original tool inference, 2026-09-22

Slurm job **209288** completed with exit 0 on `auh7-1b-gpu-259`, using one
AMD MI210. Total job time was **18m25s**. Ten held-out questions were processed
through the pinned DeepEyes V* inference function. Eight produced complete
tagged answers; eight received at least one zoom observation. This establishes
tool execution, not answer correctness or clinical performance.

Subsequent trace inspection found `addCriterion` in **all ten examples**, with
repeated tool objects/tags and excessive whitespace across the batch. The eight
extractable answers must not be interpreted as eight clean interactions. The
loop feeds entire raw responses, including repetition, into later turns. The
initial cause remains unverified. Basic checkpoint EOS settings agree with
the configured stop token; the bundled tokenizer and processor templates
render the same ordinary role boundaries. These checks do not establish
full runtime equivalence or rule out serving problems.

## Results

| Measure | Result |
| --- | --- |
| Questions / distinct held-out studies | 10 / 10 |
| Questions containing a tool-request opening tag | 10 |
| Responses containing a tool-request opening tag | 20 |
| Questions receiving a crop on a later model call | 8 |
| Crop observations delivered | 10 |
| Complete tagged answers | 8 |
| Upstream error statuses | 1 |
| Examples with length-limited generation | 1 |
| Sum of per-example inference time | 1,036.5 seconds |

The original loop parses only the first tool block in each response. Request
counts include one malformed block. A tool call alongside a final answer can
be processed without sending its crop back to the model; request and delivered
observation counts therefore differ.

Both full-report examples had format failures. Example 6 produced long runs
of blank lines, hit the 10,240-token generation limit, received three crops,
then emitted repeated opening tool tags. Its first argument block was empty
and parsing failed. Example 8 generated a report and a closing answer tag
without an opening answer tag. The upstream loop reported success because it
checks for the closing tag; our complete-answer count excludes this result.

## Model, source and protocol

- Model: `ChenShawn/DeepEyes-7B`, revision
  `3b31e4b342592d14cb12bb8434a6ea55e9cd78c8`; AMD checkpoint path
  `/vast/users/mohammad.yaqub/project/EchoSonarVideo/checkpoints/deepeyes_7b`.
- Repository HEAD: `55af99e249356af82d4c2b08b0651ae210fc810b`, with uncommitted
  inference adapter and launcher changes. Runner SHA256:
  `2ab0cdb6ed37db101045e782ec8d346bcbc323b00476ba4250b618afd36bbad7`.
  Exact runner and upstream source snapshots remain in the run's `source/`.
- [Upstream V* function](https://github.com/Visual-Agent/DeepEyes/blob/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/eval/eval_vstar.py):
  commit `11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1`, source SHA256
  `d646c2255be051cb8da3e968aec5be4eadef32ff4cd2a013ef11da9333f5e3e2`.
- Preserved: system/tool prompt, user suffix, original PIL crop and resize,
  conversation loop, temperature 0, 10,240 tokens per call, 11-call ceiling,
  and stop string `<|im_end|>`. No forced tool call was added.
- Two adapter changes: omit multiple-choice options for open-ended questions;
  use `ast.literal_eval` instead of Python `eval`. Serving uses our AMD runtime,
  so this is not a bit-for-bit reproduction of the authors' environment.

Runtime: ROCm 7.0.2, vLLM `0.17.0+rocm700`, PyTorch `2.9.1+git8907517`,
Transformers `4.57.3`, OpenAI client `2.24.0`, Pillow `12.3.0`. Local vLLM API,
BF16, eager mode, generation attention `TRITON_ATTN`, vision attention
`TORCH_SDPA`, no prefix caching, seed 1, native 128,000-token context and
checkpoint image pixel limits. Launcher: [run_deepeyes_tools.sbatch](../../scripts/run_deepeyes_tools.sbatch).

## Dataset and exact command

Sample seed **1**; two examples each of `abnormality_list`,
`structure_description`, `abnormality_classification`, `full_report`, and
`conclusion`. Each uses one random view and one random actual clip frame from
a distinct study in `build/eval.jsonl` (31,209 rows / 1,215 studies). None overlap
the 5,061 training studies in `build/rl.jsonl`; the original single-example
study is excluded. Sample SHA256:
`fb34a0d7f64e6dca4512523bd2a6e3be284d09f679273eaded01894fcee5af02`.

From `/vast/users/mohammad.yaqub/project/EchoSonarVideo`:

```bash
sbatch --parsable --nodelist=auh7-1b-gpu-259 \
  scripts/run_deepeyes_tools.sbatch \
  --sample-jsonl build/deepeyes_baseline_20260922/batch_preflight/sample.jsonl
```

The launcher invokes `python -m eval.run_deepeyes_tools` against its local
vLLM server. For this run the API URL was `http://127.0.0.1:34207/v1` and output
directory was `build/deepeyes_baseline_20260922/tools_209288`.

The prior direct-answer batch 209270 was cancelled after the user requested
the original tool protocol. Initial tool launcher attempt 209283 failed before
inference because the general vLLM CLI imported benchmark/AITER dependencies.
Launching `vllm.entrypoints.openai.api_server` directly, with the runtime's
binary directory on PATH, resolved startup for 209288.

## Artifacts and validation

The [local image report](../../build/deepeyes_original_tools_20260922/report.md)
contains all ten inputs, delivered crops, questions, references, complete
answers where present, tool arguments, and raw model outputs. It removes only
trailing whitespace from display, with counts; exact output remains in the
adjacent `outputs.json`. Nineteen unique PNG files cover ten input images and
ten crop observations, with one repeated crop deduplicated. All downloaded
checksums, image decoding, ten report sections and 22 local links were verified.
These generated files are ignored by Git and exist in this workspace.

Full AMD artifacts:
`/vast/users/mohammad.yaqub/project/EchoSonarVideo/build/deepeyes_baseline_20260922/tools_209288/`.
W&B was offline: `wandb/offline-run-20260922_113939-4xzxpgp8/` under that
directory. No online W&B link exists.

All **12** focused baseline and tool-loop tests passed on AMD. Tests exercise
the actual pinned upstream function with a fake server, verify crop bytes and
prompt/decoding parameters, reject executable tool arguments, and check the
original 11-call retry limit. Python syntax, shell syntax and whitespace
checks passed. Actual GPU inference and tool observations are captured above.
