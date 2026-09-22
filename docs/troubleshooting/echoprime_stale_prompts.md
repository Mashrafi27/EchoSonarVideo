# EchoPrime tool descriptions absent from stored prompts

On 2026-09-18, inspecting a Weave prompt exposed a mismatch between the source
system prompt and the serialized parquet inputs. All 5,061 training rows and
1,215 validation rows contained the older instruction requesting `<answer>`
tags and no tool descriptions. The source constant already described tools,
but neither the dataset nor the agent rebuilt the stored system messages.

Full run `196227` was deliberately cancelled after 10 updates (59m21s), with
user authorization to repair and restart. Its positive rewards and nonzero
gradients established answer-reward learning only. The missing instructions
are a concrete setup error; repairing them does not guarantee spontaneous
tool use by the SFT-initialized policy.

## Repair and prevention

`packages/echoprime_track/prompts.py` now holds the shared prompt and validation
contract. Tool examples are valid JSON, include valid frame and bbox values,
and explain how to continue after a tool result. Final answers are plain text
after `</think>`, consistent with the SFT answer layout.

`python -m echoprime_track.check_grpo_prompts <train.parquet> <val.parquet>
--repair --archive-dir <prior-run>/input_artifacts --tokenizer <checkpoint>
--max-prompt-length 3584` changes only system-message content. It retains exact
original files in SHA-256-named directories, preserves Arrow schema and row
order, verifies the parquet round-trip, and checks every rendered prompt with
the same `<think>\n` prefix as rollout. Run the same command without `--repair`
to validate. Both AMD launchers run this preflight; the tool agent also rejects
stale prompts before generation.

The repaired inputs retained all rows. Maximum primed prompt lengths were
3,526 tokens for train and 3,521 for validation, below the 3,584 limit.

| Artifact | Original SHA-256 | Repaired SHA-256 |
| --- | --- | --- |
| Train | `5a9c5a370b4858d9a1be736c0a03d269f3c36d74f4bac46ad76e1aa0a049bc28` | `38c0152bd2a00e9e8d1ce711b625a984e932fe8449460ce545472cf4db06adf5` |
| Validation | `e9bb095645c2e10737402cfb8cb54bc70c4403beacce8b033d2c992fcc1f9a50` | `ac6709798ffc22ace998f7a8677ae2263548dd097c732b0117b26a1a6d29357b` |

## Interpreting turn counts

The custom loop reports `num_turns = 2 + turn`, where `turn` is the zero-based
generation-loop index. A value of 2 means one generation without continuation;
3 means a second generation after a tool result. It is a historical counter,
not a count of assistant messages or proof of successful tool execution.
Inspect actual tool calls and results when evaluating tool use.
