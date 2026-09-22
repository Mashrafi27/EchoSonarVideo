# Past work: plan superseded on 2026-09-22

This is the exact prior plan, retained for provenance. Its queue and run-status
claims describe an earlier snapshot and are not current verification.

# PLAN

Living document. Where things stand and what happens next, revisit and edit this rather than
letting it go stale — see SPEC.md for the reference facts this plan is built on.

# Current state (2026-09-18)

The SFT-init full run failed before its first update because Triton could not
read `_lora_shrink_kernel.source` from its scratch cache. The reason that file
was missing is unconfirmed. Per-process compiler caches and the larger response
budget passed the two-update gate, including weight sync and a nonzero GRPO
gradient; see [the incident notes](docs/troubleshooting/echoprime_triton_cache.md).

`<think>\n` priming is present and the installed vLLM token path preserves it.
Markdown also occurs inside Darya's training thinking blocks, so Markdown alone
does not establish that thinking was skipped. Response truncation needs to be
checked before interpreting missing `</think>` as a prompt failure.

Weave 0.53.9 is installed in the AMD training environment. Both AMD launchers
enable verl's rollout tracing, and the custom agent loop traces its complete
output so decoded responses are available.

The replacement full run `196227` was intentionally stopped after 10 updates
because its stored prompts lacked tool instructions. Nonzero gradients had
confirmed answer-reward learning, but not the intended tool-training setup.

# Immediate next steps

All 5,061 training and 1,215 validation system messages are repaired. Exact
originals are archived under the stopped run's `input_artifacts` directory.
The longest repaired primed prompt is 3,526 tokens (limit 3,584). The launchers
and tool agent now reject stale prompts. GPU validation `196280` is queued;
full restart `196288` depends on its successful completion and an additional
launch gate checking actual rollout prompts and learning signal. See the
[restart record](docs/experiments/2026-09-18_echoprime_tool_prompt_repair.md).

1. Check queued validation and the gated full restart. If either fails, inspect
   the failing gate before resubmitting. Once verification finishes, remove
   generated diagnostic artifacts; retain user-requested W&B/Weave tracking.
2. Check subsequent updates for cache errors, truncation, sustained nonzero
   gradients, reward differences within GRPO groups, and tool calls.
   Tool-use learning is unverified.
3. Inspect the first saved checkpoint and held-out evaluation. A successful
   startup or positive training reward does not establish clinical quality.

# Longer-term

- Validate eval pipeline against EchoSonar-R's reported Table 1 numbers (Qwen3-VL track).
- Once echoprime_track run has checkpoints: eval with `packages/eval/echoprime/eval_checkpoint_vllm.py`, compare against EchoSonar-R Table 3 (use `packages/eval/darya_report_bridge.py` for per-section scoring).
