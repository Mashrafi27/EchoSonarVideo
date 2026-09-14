# SPEC

## Tracks
- **echo_grpo**: Qwen3-VL-2B, agentic, `echo` tool (select_view/select_frames/zoom). Config: `packages/verl_bridge/configs/echo_grpo.yaml`.
- **echoprime_track**: frozen EchoPrime encoder (mvit_v2_s, 34.6M params, 512-dim out) → trainable projector → Qwen3-8B-text, no tools, no-SFT-cold-start ablation. Config: `packages/verl_bridge/configs/echoprime_grpo.yaml`.

## Data
- `build/rl.jsonl`: train pool, 128,215 rows / 5,061 studies. No internal split.
- `build/eval.jsonl`: held-out test, 31,209 rows / 1,215 studies. Zero overlap with train. Same set EchoSonar-R reports Table 1 on.
- Question-type mix: structure_description 44.9%, abnormality_classification 43.4%, abnormality_list 3.9%, conclusion 3.9%, full_report 3.7%.
- Disease mapping: `packages/eval/diseases.py`. Their Table 1/2/3 numbers: `packages/eval/echosonar_r.py`.
- One epoch = one QA pair per study, all studies (`--one-per-study --shuffle-seed N`).

## GRPO config
LoRA r=64/alpha=128 on attn+MLP only (not projector). lr=1e-6, KL loss on (coef 0.001). Reward: `packages/verl_bridge/reward.py` → `packages/data_core/reward/score.py` (pooled, not EchoSonar-R's exact reward).

## grpo-2b-video-0910-1319 (echo_grpo, Qwen3-VL-2B) — checkpoints deleted, merged models + eval kept
Reached step 272. Eval: 200-episode capped sample (`--per-type 40`), steps 50/170/250, via served vLLM. Artifact: [Echo Reading Room](https://claude.ai/code/artifact/7d99b23f-07f9-4bf4-b1fc-ab6e025531e1).

| step | train reward | BAcc | macro F1 | answered | max_turns |
|---|---|---|---|---|---|
| 50 | ~0.15 | 0.048 | 0.086 | 92/200 | 48/200 |
| 170 | ~0.44 | 0.249 | 0.364 | 90/200 | 77/200 |
| 250 | 0.957 | 0.161 | 0.227 | 95/200 | 84/200 |

Reward diverged by question type over training (structure_description up, classification/conclusion down) despite pooled reward climbing to 0.957 — see reward mechanics below for the likely cause.

## Reward function (`packages/data_core/reward/score.py`)
`total_reward = 1.0*outcome + 0.2*format + tool_bonus`. `outcome` depends on `reward_key.kind`:
- `yesno` (abnormality_classification): exact match, `parse_yes_no(pred) == target` → 1.0/0.0. Not gameable.
- `set` (abnormality_list): F1 of `finding_set(pred)` vs gold set.
- `text` (structure_description/conclusion/full_report): tries in order —
  1. `score_gold_value`: substring containment of each gold-dict label in pred text.
  2. per-section score (`score_by_section`, EchoSonar-R's `r_cor^report`), each section scored via (3).
  3. `_score_text` = `0.5*judge + 0.5*entity_F1`; judge is `NullJudge` (always `None`) in current config, so this reduces to **pure entity_F1**.
  - `entity_F1` = F1 of keyword-stem matches (regex: `dilat|reduced|abnormal|severe|moderate|mild|regurgitat|stenos|hypertroph|impaired|akinet|hypokinet|effusion|thromb|normal`) between pred and gold text. No real content/image grounding required — "normal" is itself one of the 15 keywords, and most structures are normal in most studies.
`format` = 0.5 (has `<think>`) + 0.5 (has `<answer>` or valid `<tool_call>`), capped 1.0.

## grpo-echoprime-real-0913-2052 (echoprime_track, no-SFT ablation) — active, paused at step 180
`train_batch_size=8`, `rollout.n=2`, 3 epochs (~1896 steps total). Resumable (`resume_mode=auto`).

Full 1,215-study eval, step 150 (`build/echoprime_eval_step150_full1215.jsonl`, `build/echosonar_r_table1_comparison.md`): mean reward 0.340. Classification: F1 0.0 on all 11 diseases, BAcc 49.8 (~chance), 90-95% unparsable (format not learned yet, not a semantic failure). Too early (8% through) to know if it hits the same reward-hacking pattern as the 2B run — same question-type mix, so same risk applies.

## Infra
Served vLLM works both tracks. `resume_mode=auto` works but checkpoint-load OOMs at `gpu_memory_utilization=0.46` on resume (plain `torch.load`, no map_location) — use `0.30` for resume launches. Ray-under-load timeout on resume: `RAY_raylet_start_wait_time_s=180`.
