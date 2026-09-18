# SPEC

## Tracks
- **echo_grpo**: Qwen3-VL-2B, agentic, `echo` tool (select_view/select_frames/zoom). Config: `packages/verl_bridge/configs/echo_grpo.yaml`.
- **echoprime_track**: frozen EchoPrime encoder (mvit_v2_s, 34.6M params, 512-dim out) → trainable projector → Qwen3-8B-text. Tools: `select_frames` and `zoom` only (`select_view` retired — all views shown upfront in prompt). Initialized from Darya's SFT checkpoint (`checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503`). Config: `packages/verl_bridge/configs/echoprime_grpo.yaml`. Agent: `EchoPrimeToolAgentLoop` (`echoprime_tool_agent_loop.py`), registered as `echoprime_tool_agent`.

## Data
- `build/rl.jsonl`: train pool, 128,215 rows / 5,061 studies. No internal split.
- `build/eval.jsonl`: held-out test, 31,209 rows / 1,215 studies. Zero overlap with train. Same set EchoSonar-R reports Table 1 on.
- Question-type mix: structure_description 44.9%, abnormality_classification 43.4%, abnormality_list 3.9%, conclusion 3.9%, full_report 3.7%.
- Disease mapping: `packages/eval/diseases.py`. Their Table 1/2/3 numbers: `packages/eval/echosonar_r.py`.
- One epoch = one QA pair per study, all studies (`--one-per-study --shuffle-seed N`).

## GRPO config
LoRA r=64/alpha=128 on attn+MLP only (not projector — projector trains fully). lr=1e-6, KL loss coef 0.001. Reward: `packages/verl_bridge/reward.py` → `packages/data_core/reward/score.py`.

## Reward function (echoprime_track, current)
Multiplicative format gating matching EchoSonar-R:

    r_fmt  = 1 if ≥1 balanced <think>…</think> block AND all <tool_call> blocks are valid JSON {name, arguments}, else 0
    r_cor  = score_outcome(...)   # 0/1 for yesno, IoU∈[0,1] for set
    r_tool = tool_bonus_coef × r_cor   if tool_calls ≥ 1, else 0
    r_len  = min(0, (L − L_min) / L_min)   where L_min = 200 × (tool_calls + 1)
    reward = r_fmt × (r_cor + r_tool) + r_len

`r_fmt` gates both outcome and tool bonus: wrong format → zero reward.
`r_tool` gives partial bonus for set-IoU too (proportional to correctness), not just yesno=1.
`r_len` is an overshort penalty scaled by turns taken, never a positive bonus.
`tool_bonus_coef` defaults to 0.0 (can be annealed up via `extra_info` per-row).

**Answer extraction** (`extract_answer`): prefers `<answer>…</answer>` tags; falls back to text after the last `</think>`, stripping any trailing `<tool_call>` blocks.

**Verifiable question types**: `abnormality_classification` (yesno, exact match) and `abnormality_list` (set, IoU). Structure description / conclusion / full_report excluded from GRPO parquet (unverifiable at this stage).

## echoprime_track format spec
- **System prompt**: `"You are a medical imaging assistant specialized in echocardiography analysis."` — matches Darya's SFT data exactly. We append tool schema descriptions to guide tool use.
- **Training format** (Darya's SFT): `<think>\n{thinking}\n</think>\n{answer}` — NO `<answer>` tags. The `<think>\n` prefix was always masked (never predicted) in SFT, so the model has zero probability of generating it on its own.
- **Critical inference fix**: `EchoPrimeToolAgentLoop` appends `<think>\n` to the prompt after `apply_chat_template(add_generation_prompt=True)` to prime the assistant turn. Without this, the model skips to the answer in markdown format and `r_fmt = 0` on every episode.
- **Multi-turn format**: think→`<tool_call>{json}</tool_call>`→tool_response (turns 1…N), then think→answer (final). Up to MAX_TOOL_TURNS=8 tool rounds.
- **vLLM stop strings**: `</tool_call>` and `</answer>` (with `include_stop_str_in_output=True`) — the SFT checkpoint cannot reliably emit EOS after structural tags.

## Token limits (echoprime_track)
8 views × 393 clip tokens = 3144 clip items per prompt; 8 views × ~4 DETR detections ≈ 32 detr items.
vLLM `limit_mm_per_prompt: {clip: 4000, detr: 100}` (default 999 would reject real prompts).
`max_prompt_length=3584`, `max_model_len=4608`, `max_response_length=1024`.

## grpo-2b-video-0910-1319 (echo_grpo, Qwen3-VL-2B) — checkpoints deleted, merged models + eval kept
Reached step 272. Eval: 200-episode capped sample (`--per-type 40`), steps 50/170/250, via served vLLM. Artifact: [Echo Reading Room](https://claude.ai/code/artifact/7d99b23f-07f9-4bf4-b1fc-ab6e025531e1).

| step | train reward | BAcc | macro F1 | answered | max_turns |
|---|---|---|---|---|---|
| 50 | ~0.15 | 0.048 | 0.086 | 92/200 | 48/200 |
| 170 | ~0.44 | 0.249 | 0.364 | 90/200 | 77/200 |
| 250 | 0.957 | 0.161 | 0.227 | 95/200 | 84/200 |

Reward diverged by question type over training (structure_description up, classification/conclusion down) despite pooled reward climbing to 0.957 — see reward mechanics below for the likely cause.

## Reward function (echo_grpo track, old additive formula — kept for reference)
`total_reward = 1.0*outcome + 0.2*format + tool_bonus`. `outcome` depends on `reward_key.kind`:
- `yesno` (abnormality_classification): exact match, `parse_yes_no(pred) == target` → 1.0/0.0. Not gameable.
- `set` (abnormality_list): F1 of `finding_set(pred)` vs gold set.
- `text` (structure_description/conclusion/full_report): tries in order —
  1. `score_gold_value`: substring containment of each gold-dict label in pred text.
  2. per-section score (`score_by_section`, EchoSonar-R's `r_cor^report`), each section scored via (3).
  3. `_score_text` = `0.5*judge + 0.5*entity_F1`; judge is `NullJudge` (always `None`) in current config, so this reduces to **pure entity_F1**.
  - `entity_F1` = F1 of keyword-stem matches (regex: `dilat|reduced|abnormal|severe|moderate|mild|regurgitat|stenos|hypertroph|impaired|akinet|hypokinet|effusion|thromb|normal`) between pred and gold text. No real content/image grounding required — "normal" is itself one of the 15 keywords, and most structures are normal in most studies.
`format` = 0.5 (has `<think>`) + 0.5 (has `<answer>` or valid `<tool_call>`), capped 1.0.

## grpo-echoprime-real-0913-2052 (echoprime_track, no-SFT ablation) — paused at step 180
`train_batch_size=8`, `rollout.n=2`, 3 epochs (~1896 steps total). Used old additive reward formula and no-SFT cold start. Superseded by the SFT-init tool track below.

Full 1,215-study eval, step 150 (`build/echoprime_eval_step150_full1215.jsonl`, `build/echosonar_r_table1_comparison.md`): mean reward 0.340. Classification: F1 0.0 on all 11 diseases, BAcc 49.8 (~chance), 90-95% unparsable (format not learned yet, not a semantic failure).

## echoprime_track SFT-init tool run — pending launch (2026-09-18)
Smoke job 195792 passed (2 steps, 2x MI210, ~179s/step). Zero reward during smoke: root cause was `<think>` not being generated — fixed by priming the assistant turn with `<think>\n` in the agent loop. Parquet rebuild required before real run (new system prompt with tool schemas). Configuration: `train_batch_size=8`, `rollout.n=2`, 3 epochs (~1896 steps), `max_response_length=1024`, 2x MI210, 126h wall time.

Key changes from the no-SFT ablation:
- Initialized from Darya's SFT checkpoint (not cold start)
- Tools: select_frames and zoom (not select_view — all views upfront)
- New multiplicative reward: `r_fmt × (r_cor + r_tool) + r_len`
- `<think>\n` priming in agent loop (critical: model never predicted this token in SFT)

## Infra
Served vLLM works both tracks. `resume_mode=auto` works but checkpoint-load OOMs at `gpu_memory_utilization=0.46` on resume (plain `torch.load`, no map_location) — use `0.30` for resume launches. Ray-under-load timeout on resume: `RAY_raylet_start_wait_time_s=180`.
