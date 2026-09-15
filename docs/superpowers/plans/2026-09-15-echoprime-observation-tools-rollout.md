# EchoPrime dual-modality observation + select_frames/zoom tools + multi-turn rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `packages/echoprime_track/`'s GRPO rollout actually match Darya's SFT observation format (full per-view clip + DETR token grids, real projector weights) and add `select_frames`/`zoom` as tools that index into that same precomputed grid, via a new multi-turn agent loop.

**Architecture:** Three parallel representations of "the model" all currently hardcode a single 512-dim pooled embedding + one `VIEW_TOKEN`: the HF class (`modeling.py`, used for the FSDP actor's training forward/backward), the vLLM-native class (`vllm_model.py`, used for rollout generation), and the SFT-init script (`save_sft_init_checkpoint.py`). All three move to two modalities (clip 768-dim, detr 256-dim) with two placeholder tokens and two projectors, matching `report_generation/sft_thinking/model.py::EchoVLM` exactly, with real weights copied from Darya's checkpoint. A new pure-function module (`grid.py`) indexes the precomputed `(393, 768)` per-view grid for `select_frames`/`zoom` (no live encoder inference). A new agent loop (`echoprime_tool_agent_loop.py`) drives multi-turn generation, accumulating prompt tokens and embeddings across tool calls the same way verl's own `ToolAgentLoop` does for images (same `request_id`, full-sequence resend each `generate()` call).

**Tech Stack:** PyTorch, HuggingFace `transformers`, vLLM (custom-registered model), upstream `verl` v0.7.1 (pinned submodule, `external/verl`), h5py, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md`

## Global Constraints

- No live EchoPrime encoder inference at rollout time — `select_frames`/`zoom` only index into the already-cached `(393, 768)` grid (spec §2).
- `select_frames`/`zoom` are stateless: `view` is a required argument on every call, no `select_view` tool (spec §2).
- Each `select_frames`/`zoom` call resolves to exactly 1 temporal group (49 tokens, fewer for `zoom` after spatial subselection) — user-confirmed cap (spec §2, "Token budget" decision).
- `packages/echoprime_track/dataset.py`'s `EchoPrimeSFTDataset`/`EchoPrimeCollator`/`train_sft.py` are pre-existing SFT-training code with **no active launcher referencing them** (verified this session via repo-wide grep — no `.sbatch`/`.sh` references `train_sft.py`) — out of scope, left untouched. `dataset.py::SYSTEM_PROMPT` IS actively imported elsewhere and must keep working; `VIEW_TOKEN` stays defined in `modeling.py` as a legacy constant so `dataset.py`'s existing import doesn't break, even though the new splice path (Task 2) doesn't use it.
- `packages/echoprime_track/save_cold_start_checkpoint.py` (the older no-SFT-init ablation entrypoint) is untouched — out of scope.
- Reward changes (`data_core/reward/score.py`, `verl_bridge/reward.py`, canonical-finding taxonomy, DETR-grounding tool bonus) are a **separate plan** (`docs/superpowers/plans/2026-09-15-echoprime-verifiable-reward.md`) — not touched here, except that Task 7 threads a per-view `dicom_uuid` into `extra_info` so that plan has what it needs later.
- Every model/code change in this plan must keep `EchoPrimeQwen3ForCausalLM.from_cold_start` and `EchoPrimeQwen3ForCausalLMVLLM`'s weight names loadable from the SAME checkpoint directory produced by `save_sft_init_checkpoint.py` (Task 3) — the HF class trains it, the vLLM class serves it; a name mismatch between them silently breaks weight sync (this exact class of bug already bit this project once, per `echoprime_grpo.yaml`'s comment about `"projector.0.base_layer"`).

---

## Task 1: Verify Darya's h5 caches actually cover this project's study pool

**Files:**
- Create: `packages/echoprime_track/coverage.py`
- Test: `packages/echoprime_track/tests/test_coverage.py`
- Create: `scripts/check_darya_cache_coverage.py`

**Interfaces:**
- Produces: `coverage.py::missing_dicom_uuids(needed: set[str], available: set[str]) -> set[str]` — pure set difference, used by later tasks' own coverage assertions if needed, and by the standalone script.

This is a standalone risk-verification step (spec §1). **Update, ruled during execution:** the real check found ~40% of needed (study, view) pairs missing from Darya's clip-token h5 on both splits, uniformly across every study. Verified this is NOT a blocker — Darya's own SFT (`report_generation/sft_thinking/dataset.py::__getitem__`) already trains on exactly this partial-coverage pattern: it writes the view-name text unconditionally but only appends clip-token placeholders `if dicom_uuid in self.clip_h5`, silently omitting the block otherwise. So there is no "pass" gate before Task 7 — the script reports coverage as an informational metric, and Task 7's view-block construction (§ below) must replicate Darya's same skip-if-missing guard, not require full coverage. See spec §1's updated "Risk verified" paragraph for the full reasoning.

- [ ] **Step 1: Write the failing test for the pure coverage-diff function**

```python
# packages/echoprime_track/tests/test_coverage.py
from echoprime_track.coverage import missing_dicom_uuids


def test_missing_dicom_uuids_reports_gap():
    needed = {"di-0001", "di-0002", "di-0003"}
    available = {"di-0001", "di-0003"}
    assert missing_dicom_uuids(needed, available) == {"di-0002"}


def test_missing_dicom_uuids_full_coverage():
    needed = {"di-0001", "di-0002"}
    available = {"di-0001", "di-0002", "di-0099"}
    assert missing_dicom_uuids(needed, available) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/echoprime_track && python -m pytest tests/test_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'echoprime_track.coverage'`

- [ ] **Step 3: Write the minimal implementation**

```python
# packages/echoprime_track/coverage.py
"""Pure set-difference helper for checking whether Darya's h5 caches
(report_generation/data/clip_tokens_{train,test}.h5, {train,test}_detections.h5,
keyed by dicom_uuid) actually cover this project's own train/eval dicom pool
before any training-facing code trusts them (spec: docs/superpowers/specs/
2026-09-15-echoprime-tool-grpo-design.md, section 1's coverage risk note).
"""


def missing_dicom_uuids(needed: set, available: set) -> set:
    return needed - available
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/echoprime_track && python -m pytest tests/test_coverage.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the real coverage-check script (not unit-tested — needs the real 50GB+ h5 files and the real study pool; run manually, once, before Task 7)**

```python
# scripts/check_darya_cache_coverage.py
"""Run once, manually, before trusting Darya's h5 caches for training (see
docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md section 1).

Usage:
    python scripts/check_darya_cache_coverage.py \
        --frames-jsonl build/frames.jsonl \
        --clip-h5 /vast/users/mohammad.yaqub/report_generation/data/clip_tokens_train.h5 \
        --detr-h5 /vast/users/mohammad.yaqub/report_generation/data/train_detections.h5

Run once for the train split and once for the test split (swap in
clip_tokens_test.h5 / test_detections.h5 and the matching frames file).
"""
import argparse
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
from echoprime_track.coverage import missing_dicom_uuids


def _needed_dicom_uuids(frames_jsonl: str) -> set:
    needed = set()
    with open(frames_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for dc in rec["dicoms"]:
                needed.add(dc["dicom_uuid"])
    return needed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-jsonl", required=True)
    ap.add_argument("--clip-h5", required=True)
    ap.add_argument("--detr-h5", required=True)
    args = ap.parse_args(argv)

    needed = _needed_dicom_uuids(args.frames_jsonl)
    print(f"[coverage] {len(needed)} dicom_uuids needed from {args.frames_jsonl}", flush=True)

    with h5py.File(args.clip_h5, "r") as f:
        clip_keys = set(f.keys())
    missing_clip = missing_dicom_uuids(needed, clip_keys)
    print(f"[coverage] clip h5: {len(clip_keys)} keys, "
          f"{len(missing_clip)}/{len(needed)} needed dicoms MISSING", flush=True)
    if missing_clip:
        print(f"[coverage] example missing clip dicom_uuids: {list(missing_clip)[:10]}", flush=True)

    with h5py.File(args.detr_h5, "r") as f:
        detr_keys = set(f.keys())
    missing_detr = missing_dicom_uuids(needed, detr_keys)
    print(f"[coverage] detr h5: {len(detr_keys)} keys, "
          f"{len(missing_detr)}/{len(needed)} needed dicoms MISSING "
          "(fine to have gaps here -- absent detections just means no "
          "'Structure features:' block for that view, not a hard failure)",
          flush=True)

    print(f"[coverage] clip-token coverage: {len(needed) - len(missing_clip)}/{len(needed)} "
          f"({100 * (len(needed) - len(missing_clip)) / len(needed):.1f}%) -- informational "
          "only. Partial coverage is EXPECTED: Darya's own SFT (report_generation/sft_thinking/"
          "dataset.py::__getitem__) already tolerates a missing dicom_uuid by silently omitting "
          "that view's clip-token block (keeps the view-name text line) -- Task 7's view-block "
          "construction must replicate that same guard. This is not a pass/fail gate.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `--frames-jsonl`/a `build/frames.jsonl`-shaped file does not exist in this repo (confirmed during Task 1's real execution) — this project's actual study/dicom source is `build/rl.jsonl` / `build/eval.jsonl`, whose records carry `overview.views[]` (`{"view", "frame", "frame_count"}`, dicom_uuid parsed from the `frame` path's `di-<uuid>_<View>` parent directory, per `packages/data_core/data/views.py::parse_clip_dirname`'s convention) rather than a `dicoms` list. Rename the CLI arg to `--source-jsonl` and adjust `_needed_dicom_uuids` accordingly before writing this script for real — don't build against the brief's literal `--frames-jsonl`/`rec["dicoms"]` shape without checking, since it doesn't match this repo.

- [ ] **Step 6: Run the real coverage check against both splits and record the result**

```bash
python scripts/check_darya_cache_coverage.py \
    --source-jsonl build/rl.jsonl \
    --clip-h5 /vast/users/mohammad.yaqub/report_generation/data/clip_tokens_train.h5 \
    --detr-h5 /vast/users/mohammad.yaqub/report_generation/data/train_detections.h5
python scripts/check_darya_cache_coverage.py \
    --source-jsonl build/eval.jsonl \
    --clip-h5 /vast/users/mohammad.yaqub/report_generation/data/clip_tokens_test.h5 \
    --detr-h5 /vast/users/mohammad.yaqub/report_generation/data/test_detections_merged.h5
```

(Note the eval-split DETR file is `test_detections_merged.h5`, not `test_detections.h5` — confirmed against the real file on disk during Task 1.)

Expected: both print a coverage percentage (real numbers found this session: ~40% train, ~41% eval, uniform across studies) — record it in the ledger for later reference, but there is nothing to "pass" here; proceed to Task 7 regardless, with its view-block construction built to skip missing dicoms gracefully.

- [ ] **Step 7: Commit**

```bash
git add packages/echoprime_track/coverage.py packages/echoprime_track/tests/test_coverage.py \
        scripts/check_darya_cache_coverage.py
git commit -m "$(cat <<'EOF'
add dicom_uuid coverage check between our study pool and Darya's h5 caches

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Restructure `modeling.py` for dual clip/detr projectors + generalized splice

**Files:**
- Modify: `packages/echoprime_track/modeling.py`
- Test: `packages/echoprime_track/tests/test_modeling_splice.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `EchoPrimeQwen3Config(clip_embed_dim=768, detr_embed_dim=256, clip_token, detr_token, clip_token_id, detr_token_id, ...)`; `EchoPrimeQwen3ForCausalLM.clip_projector`, `.detr_projector` (both `nn.Sequential(nn.LayerNorm, nn.Linear)`); `EchoPrimeQwen3ForCausalLM._splice_positions` (staticmethod, pure-tensor, unit-tested); `.forward(input_ids, attention_mask=None, clip_embeddings=None, clip_counts=None, detr_embeddings=None, detr_counts=None, labels=None, past_key_values=None, inputs_embeds=None, **kwargs)`. Task 3, Task 4, Task 7, Task 8 all consume these exact names.

- [ ] **Step 1: Write the failing test for `_splice_positions` (pure-tensor, no real LM needed)**

```python
# packages/echoprime_track/tests/test_modeling_splice.py
import torch

from echoprime_track.modeling import EchoPrimeQwen3ForCausalLM


def test_splice_positions_single_example():
    # 1 example, 5 tokens, token_id=99 at positions [1, 3]
    input_ids = torch.tensor([[0, 99, 0, 99, 0]])
    inputs_embeds = torch.zeros(1, 5, 4)
    projected = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0, 2.0]])
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected, counts=[2])
    assert torch.equal(out[0, 1], projected[0])
    assert torch.equal(out[0, 3], projected[1])
    assert torch.equal(out[0, 0], torch.zeros(4))  # untouched


def test_splice_positions_batch_with_zero_count_row():
    # row 0 has 1 placeholder, row 1 has none
    input_ids = torch.tensor([[99, 0, 0], [0, 0, 0]])
    inputs_embeds = torch.zeros(2, 3, 4)
    projected = torch.tensor([[9.0, 9.0, 9.0, 9.0]])
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected, counts=[1, 0])
    assert torch.equal(out[0, 0], projected[0])
    assert torch.equal(out[1], torch.zeros(3, 4))


def test_splice_positions_count_mismatch_asserts():
    input_ids = torch.tensor([[99, 99, 0]])  # 2 placeholders
    inputs_embeds = torch.zeros(1, 3, 4)
    projected = torch.zeros(1, 4)
    try:
        EchoPrimeQwen3ForCausalLM._splice_positions(
            inputs_embeds, input_ids, token_id=99, projected=projected, counts=[1])
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "placeholders" in str(e)


def test_splice_positions_accepts_tensor_counts():
    # verl's extract_multi_modal_inputs concatenates per-example (1,) tensors this way
    input_ids = torch.tensor([[99, 0]])
    inputs_embeds = torch.zeros(1, 2, 4)
    projected = torch.ones(1, 4)
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected,
        counts=torch.tensor([[1]]))
    assert torch.equal(out[0, 0], torch.ones(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/echoprime_track && python -m pytest tests/test_modeling_splice.py -v`
Expected: FAIL — `_splice_positions` doesn't exist yet (AttributeError).

- [ ] **Step 3: Rewrite the config class's constructor to carry dual-modality fields**

In `packages/echoprime_track/modeling.py`, replace the `VIEW_TOKEN`/`video_embed_dim`/`view_token`/`view_token_id` constructor args with clip/detr equivalents, keeping `VIEW_TOKEN` defined at module level (legacy, see Global Constraints) but no longer threaded through the config:

```python
VIEW_TOKEN = "<|view_embed|>"  # legacy -- only echoprime_track/dataset.py's SFT collator
                                # (no active launcher references it, see plan's Global
                                # Constraints) still imports this; unused by the splice below.
CLIP_TOKEN = "<|clip_embed|>"
DETR_TOKEN = "<|detr_embed|>"


class EchoPrimeQwen3Config(PretrainedConfig):
    model_type = "echoprime_qwen3"

    def __init__(self, text_model_name: str = "Qwen/Qwen3-8B",
                 clip_embed_dim: int = 768, detr_embed_dim: int = 256,
                 clip_token: str = CLIP_TOKEN, detr_token: str = DETR_TOKEN,
                 clip_token_id: int = None, detr_token_id: int = None,
                 text_config=None, **kwargs):
        self.text_model_name = text_model_name
        self.clip_embed_dim = clip_embed_dim
        self.detr_embed_dim = detr_embed_dim
        self.clip_token = clip_token
        self.detr_token = detr_token
        self.clip_token_id = clip_token_id
        self.detr_token_id = detr_token_id
        # (everything below this line is UNCHANGED from the current file -- the
        # text_config handling, the eos/pad/bos/tie_word_embeddings sync loop, and the
        # super().__init__(**kwargs) call. Do not touch those, only the block above.)
```

Do not touch `__getattr__` — it stays exactly as-is.

- [ ] **Step 4: Rewrite `EchoPrimeQwen3ForCausalLM.__init__` for dual projectors**

```python
    def __init__(self, config: EchoPrimeQwen3Config):
        super().__init__(config)
        self.lm = AutoModelForCausalLM.from_config(config.text_config)
        hidden = self.lm.config.hidden_size
        # LayerNorm, Linear, GELU, Linear -- matching report_generation/sft_thinking/
        # model.py::EchoVLM's clip_projector/detr_projector layer-for-layer (verified against
        # the real file directly, model.py:67-79 -- an earlier version of this plan said 2
        # layers, based on truncated grep output; corrected during Task 2's review).
        # save_sft_init_checkpoint.py (Task 3) loads Darya's real trained weights into these
        # via a strict load_state_dict, so the shapes/layer types must match hers exactly or
        # that load raises RuntimeError: Unexpected key(s) in state_dict.
        self.clip_projector = nn.Sequential(
            nn.LayerNorm(config.clip_embed_dim),
            nn.Linear(config.clip_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.detr_projector = nn.Sequential(
            nn.LayerNorm(config.detr_embed_dim),
            nn.Linear(config.detr_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
```

- [ ] **Step 5: Update `from_cold_start` to cast both projectors, not the old single one**

Replace the single `model.projector.to(dtype)` line near the end of `from_cold_start` with:

```python
        model.projector.to(dtype)
```
becomes
```python
        model.clip_projector.to(dtype)
        model.detr_projector.to(dtype)
```

Everything else in `from_cold_start` (loading the real Qwen3-8B weights, re-pointing `model.config.text_config`, re-syncing eos/pad/bos/tie_word_embeddings) is unchanged.

- [ ] **Step 6: Replace `_splice_view_embeddings` with `_splice_positions` (staticmethod) + `_splice_vision_embeddings` (instance method)**

```python
    @staticmethod
    def _splice_positions(inputs_embeds: torch.Tensor, input_ids: torch.Tensor,
                           token_id: int, projected: torch.Tensor, counts) -> torch.Tensor:
        """Returns `inputs_embeds` with each `token_id` position (in order, per example)
        replaced by the next row of `projected`. `projected` is (sum(counts), H) -- unpadded,
        concatenated across the batch in order, same convention the old
        `_splice_view_embeddings` used for a single modality, generalized to be called once
        per modality (clip, detr) instead of baking one modality in."""
        if torch.is_tensor(counts):
            counts = counts.flatten().tolist()
        offset = 0
        for b in range(input_ids.shape[0]):
            n = counts[b]
            if n == 0:
                continue
            positions = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
            assert len(positions) == n, (
                f"row {b}: {len(positions)} placeholders for token_id={token_id} but "
                f"{n} embeddings -- prompt construction and the embeddings passed into "
                f"forward() disagree on this example's count")
            inputs_embeds[b, positions] = projected[offset:offset + n]
            offset += n
        return inputs_embeds

    def _splice_vision_embeddings(self, input_ids: torch.Tensor,
                                   clip_embeddings, clip_counts,
                                   detr_embeddings, detr_counts) -> torch.Tensor:
        """inputs_embeds with clip- and detr-token positions replaced by their projected
        embeddings. Either modality may be absent (e.g. a study with no DETR detections at
        all) -- `clip_embeddings`/`detr_embeddings` None or empty just skips that splice."""
        # .clone() required, not optional: see the original _splice_view_embeddings'
        # docstring for why (in-place indexed assignment into a leaf-adjacent
        # requires-grad tensor errors during backward without it).
        inputs_embeds = self.lm.get_input_embeddings()(input_ids).clone()
        if clip_embeddings is not None and clip_embeddings.numel() > 0:
            clip_proj = self.clip_projector(clip_embeddings.to(inputs_embeds.dtype))
            inputs_embeds = self._splice_positions(
                inputs_embeds, input_ids, self.config.clip_token_id, clip_proj, clip_counts)
        if detr_embeddings is not None and detr_embeddings.numel() > 0:
            detr_proj = self.detr_projector(detr_embeddings.to(inputs_embeds.dtype))
            inputs_embeds = self._splice_positions(
                inputs_embeds, input_ids, self.config.detr_token_id, detr_proj, detr_counts)
        return inputs_embeds
```

- [ ] **Step 7: Update `forward` and `prepare_inputs_for_generation` for the new params**

```python
    def forward(self, input_ids, attention_mask=None,
                clip_embeddings=None, clip_counts=None,
                detr_embeddings=None, detr_counts=None,
                labels=None, past_key_values=None, inputs_embeds=None, **kwargs):
        """`clip_embeddings is not None or detr_embeddings is not None` is the actual branch
        condition (mirrors the old `view_embeddings is not None` check) -- see the original
        docstring for why `inputs_embeds` is a named-but-discarded parameter."""
        if clip_embeddings is not None or detr_embeddings is not None:
            inputs_embeds = self._splice_vision_embeddings(
                input_ids, clip_embeddings, clip_counts, detr_embeddings, detr_counts)
            return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            labels=labels, past_key_values=past_key_values, **kwargs)
        return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        past_key_values=past_key_values, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                       clip_embeddings=None, clip_counts=None,
                                       detr_embeddings=None, detr_counts=None, **kwargs):
        model_inputs = self.lm.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs)
        if past_key_values is None:
            model_inputs["clip_embeddings"] = clip_embeddings
            model_inputs["clip_counts"] = clip_counts
            model_inputs["detr_embeddings"] = detr_embeddings
            model_inputs["detr_counts"] = detr_counts
        return model_inputs
```

- [ ] **Step 8: Run the splice tests to verify they pass**

Run: `cd packages/echoprime_track && python -m pytest tests/test_modeling_splice.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Run the full existing test suite for this package to catch anything the rename broke**

Run: `cd packages/echoprime_track && python -m pytest -v`
Expected: PASS. If anything outside `test_modeling_splice.py` references `EchoPrimeQwen3ForCausalLM`'s old `projector`/`_splice_view_embeddings`/`view_embeddings` names, fix that reference now (it's a consumer this plan's later tasks haven't reached yet) rather than deferring.

- [ ] **Step 10: Commit**

```bash
git add packages/echoprime_track/modeling.py packages/echoprime_track/tests/test_modeling_splice.py
git commit -m "$(cat <<'EOF'
restructure EchoPrimeQwen3ForCausalLM for dual clip/detr projectors

Replaces the single pooled-512-dim VIEW_TOKEN splice with two independently
projected modalities (clip 768-dim, detr 256-dim), matching Darya's real SFT
observation format instead of a format the checkpoint was never trained on.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Load Darya's real projector weights in `save_sft_init_checkpoint.py`

**Files:**
- Modify: `scripts/save_sft_init_checkpoint.py`
- Test: `scripts/tests/test_save_sft_init_checkpoint.py` (new file — this script package has no `tests/` dir yet)

**Interfaces:**
- Consumes: `EchoPrimeQwen3Config`, `EchoPrimeQwen3ForCausalLM`, `CLIP_TOKEN`, `DETR_TOKEN` from Task 2's `modeling.py`.
- Produces: a checkpoint dir whose `clip_projector`/`detr_projector` weights are Darya's real trained ones, not random — consumed at training-launch time by `echoprime_grpo.yaml`'s `actor_rollout_ref.model.path`.

- [ ] **Step 1: Write the failing test for the new `_load_darya_projectors` helper, using tiny fake projector checkpoints (no real 8B model, no real Darya checkpoint needed)**

```python
# scripts/tests/test_save_sft_init_checkpoint.py
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from save_sft_init_checkpoint import load_darya_projectors


def _toy_projector(in_dim, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))


def test_load_darya_projectors_copies_real_weights(tmp_path):
    # Simulate EchoVLM.load_projectors's save format:
    # <save_dir>/clip_projector.pt, <save_dir>/detr_projector.pt
    src_clip = _toy_projector(768, 16)
    src_detr = _toy_projector(256, 16)
    torch.save(src_clip.state_dict(), tmp_path / "clip_projector.pt")
    torch.save(src_detr.state_dict(), tmp_path / "detr_projector.pt")

    dst_clip = _toy_projector(768, 16)
    dst_detr = _toy_projector(256, 16)
    # dst starts randomly initialized -- must differ from src before loading
    assert not torch.equal(dst_clip[1].weight, src_clip[1].weight)

    load_darya_projectors(dst_clip, dst_detr, str(tmp_path))

    assert torch.equal(dst_clip[1].weight, src_clip[1].weight)
    assert torch.equal(dst_detr[1].weight, src_detr[1].weight)


def test_load_darya_projectors_missing_file_raises(tmp_path):
    dst_clip = _toy_projector(768, 16)
    dst_detr = _toy_projector(256, 16)
    try:
        load_darya_projectors(dst_clip, dst_detr, str(tmp_path))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest tests/test_save_sft_init_checkpoint.py -v`
Expected: FAIL — `load_darya_projectors` doesn't exist yet.

- [ ] **Step 3: Add `load_darya_projectors` and wire it into `main`**

In `scripts/save_sft_init_checkpoint.py`, add the helper function and call it after `from_cold_start`, before saving. Also add both special tokens (not just `VIEW_TOKEN`) and set both token ids on the config:

```python
def load_darya_projectors(clip_projector: torch.nn.Module, detr_projector: torch.nn.Module,
                           darya_checkpoint_dir: str) -> None:
    """Loads report_generation/sft_thinking/model.py::EchoVLM.save_projectors's saved format
    (clip_projector.pt / detr_projector.pt, plain state_dicts) into our own projector modules,
    in place. Raises FileNotFoundError if either file is missing -- a silent skip here would
    leave the projector randomly initialized with no signal that Darya's real weights never
    loaded (the exact bug this plan's Task 3 exists to fix)."""
    clip_path = os.path.join(darya_checkpoint_dir, "clip_projector.pt")
    detr_path = os.path.join(darya_checkpoint_dir, "detr_projector.pt")
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"no clip_projector.pt at {darya_checkpoint_dir}")
    if not os.path.exists(detr_path):
        raise FileNotFoundError(f"no detr_projector.pt at {darya_checkpoint_dir}")
    clip_projector.load_state_dict(torch.load(clip_path, map_location="cpu"))
    detr_projector.load_state_dict(torch.load(detr_path, map_location="cpu"))
```

And in `main`, replace the `tok.add_special_tokens({"additional_special_tokens": [VIEW_TOKEN]})` /
`view_token_id = ...` block with:

```python
from echoprime_track.modeling import (
    EchoPrimeQwen3Config,
    EchoPrimeQwen3ForCausalLM,
    CLIP_TOKEN,
    DETR_TOKEN,
)
...
    tok.add_special_tokens({"additional_special_tokens": [CLIP_TOKEN, DETR_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    clip_token_id = tok.convert_tokens_to_ids(CLIP_TOKEN)
    detr_token_id = tok.convert_tokens_to_ids(DETR_TOKEN)
    print(f"[sft_init] CLIP_TOKEN id={clip_token_id}, DETR_TOKEN id={detr_token_id}, "
          f"total vocab={len(tok)}", flush=True)

    cfg = EchoPrimeQwen3Config(text_model_name=args.sft_llm,
                                clip_token_id=clip_token_id, detr_token_id=detr_token_id)
    model = EchoPrimeQwen3ForCausalLM.from_cold_start(cfg, dtype=torch.bfloat16)
```

Add a new required CLI arg for Darya's projector checkpoint dir, and call the loader right after `from_cold_start`:

```python
    ap.add_argument(
        "--darya-projectors", required=True,
        help="Dir containing Darya's clip_projector.pt/detr_projector.pt "
             "(same checkpoint-1503 dir as --sft-tokenizer, per EchoVLM.save_projectors).")
```

```python
    load_darya_projectors(model.clip_projector, model.detr_projector, args.darya_projectors)
    print(f"[sft_init] loaded real clip/detr projector weights from {args.darya_projectors}",
          flush=True)
```

Update the module's `resize_token_embeddings` call site: it already reads `len(tok)` generically, no change needed there. Update the module docstring's `Usage` example to add `--darya-projectors` pointing at the same `checkpoint-1503` dir.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd scripts && python -m pytest tests/test_save_sft_init_checkpoint.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the real script against Darya's actual checkpoint (manual verification, not a unit test)**

```bash
python scripts/save_sft_init_checkpoint.py \
    --sft-tokenizer /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503 \
    --sft-llm      /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503/llm \
    --darya-projectors /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503 \
    --out-dir      /vast/users/mohammad.yaqub/project/EchoSonarVideo/build/echoprime_sft_init
```

Expected: prints `[sft_init] loaded real clip/detr projector weights from ...` and completes with `[sft_init] Done -> .../build/echoprime_sft_init`. If `clip_projector.pt`/`detr_projector.pt` aren't actually at that path (check with `ls` first — `EchoVLM.save_projectors`'s exact save location in her training script may differ from the eval checkpoint dir), find the right path before proceeding; don't guess.

- [ ] **Step 6: Commit**

```bash
git add scripts/save_sft_init_checkpoint.py scripts/tests/test_save_sft_init_checkpoint.py
git commit -m "$(cat <<'EOF'
load Darya's real clip/detr projector weights into the SFT-init checkpoint

Previously the projector started randomly initialized -- only the LLM half
was ever actually Darya's SFT weights, which combined with the pooled-512
observation format (fixed in modeling.py) meant this track's SFT init barely
transferred anything from her checkpoint.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Restructure `vllm_model.py` for dual clip/detr modalities

**Files:**
- Modify: `packages/echoprime_track/vllm_model.py`
- Test: manual verification only (see Step 1 — this needs a real vLLM install, which only exists in the AMD validated env, not in a fast unit-test environment)

**Interfaces:**
- Consumes: `CLIP_TOKEN`, `DETR_TOKEN` from Task 2's `modeling.py`.
- Produces: `EchoPrimeQwen3ForCausalLMVLLM` registered for two modalities (`"clip"`, `"detr"`) instead of one (`"image"`); loadable from the same checkpoint dir Task 3 produces (`hf_to_vllm_mapper` must still rename `lm.*` → `language_model.*`, and now also needs `clip_projector.*`/`detr_projector.*` to match Task 2's HF-side names exactly — both sides already use the same attribute names, so no new mapper entries needed, but verify this in Step 4).

- [ ] **Step 1: Investigate vLLM's real multi-modal merge API on the AMD box before writing the dual-modality registration**

This project's `vllm_model.py` currently registers exactly one modality (`"image"`). Before writing code for two modalities (`"clip"`, `"detr"`), confirm on the machine where vLLM is actually installed (`.tmp_work/rocm_validation_20260914/env`, per CLAUDE.md's AMD substrate section) how `SupportsMultiModal.embed_multimodal` is called when a model registers more than one modality — specifically:

```bash
source .tmp_work/rocm_validation_20260914/env/bin/activate  # or the project's actual activation path
python -c "
import inspect
from vllm.model_executor.models.interfaces import SupportsMultiModal
print(inspect.getsource(SupportsMultiModal.get_multimodal_embeddings))
"
python -c "
import inspect
import vllm.model_executor.models.utils as u
print(inspect.getsource(u.merge_multimodal_embeddings))
"
```

Read both. Confirm: (a) is `embed_multimodal` (or whatever the current vLLM version actually calls the hook — check the exact method name on `SupportsMultiModal` too, since this repo's existing `vllm_model.py` defines `embed_multimodal` but the interface may have renamed it) called once per forward pass with ALL registered modalities' kwargs present together (e.g. both `clip_embeds` and `detr_embeds` in the same `**kwargs`), or once per modality; (b) does the return value need to be one combined sequence covering every placeholder position across both modalities in prompt order, or a dict/list keyed by modality. Write a one-paragraph note into this task's PR/commit message (or a scratch file, then delete it) recording what you found, then proceed to Step 2 using the CONFIRMED shape, not the guess below — the code in Step 2 is this plan's best-grounded prediction (multiple existing vLLM multi-modal models, e.g. ones combining image+video, register more than one modality this same way), but treat it as a draft to correct against what Step 1 actually shows.

- [ ] **Step 2: Rewrite the processor/dummy-input/model classes for two modalities**

```python
CLIP_EMBED_DIM = 768
DETR_EMBED_DIM = 256


class _EchoPrimeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str, clip=None, detr=None, **kwargs) -> BatchFeature:
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if clip is not None:
            data["clip_embeds"] = clip
        if detr is not None:
            data["detr_embeds"] = detr
        return BatchFeature(data=data)


class EchoPrimeProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"clip": None, "detr": None}

    def get_hf_processor(self, **kwargs: object) -> _EchoPrimeProcessor:
        return _EchoPrimeProcessor(self.get_tokenizer())

    def _get_expected_hidden_size(self) -> int | None:
        return None


class EchoPrimeDummyInputsBuilder(BaseDummyInputsBuilder["EchoPrimeProcessingInfo"]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN
        return CLIP_TOKEN * mm_counts.get("clip", 0) + DETR_TOKEN * mm_counts.get("detr", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options
    ) -> MultiModalDataDict:
        n_clip = mm_counts.get("clip", 0)
        n_detr = mm_counts.get("detr", 0)
        return {
            "clip": torch.zeros(n_clip, 1, CLIP_EMBED_DIM),
            "detr": torch.zeros(n_detr, 1, DETR_EMBED_DIM),
        }


class EchoPrimeMultiModalProcessor(BaseMultiModalProcessor[EchoPrimeProcessingInfo]):
    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            clip_embeds=MultiModalFieldConfig.batched("clip"),
            detr_embeds=MultiModalFieldConfig.batched("detr"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        clip_token_id = hf_config.clip_token_id
        detr_token_id = hf_config.detr_token_id

        def get_clip_replacement(item_idx: int):
            items = mm_items.get_items("clip", (ImageEmbeddingItems, ImageProcessorItems))
            return [clip_token_id] * items.get_feature_size(item_idx)

        def get_detr_replacement(item_idx: int):
            items = mm_items.get_items("detr", (ImageEmbeddingItems, ImageProcessorItems))
            return [detr_token_id] * items.get_feature_size(item_idx)

        return [
            PromptReplacement(modality="clip", target=[clip_token_id],
                               replacement=get_clip_replacement),
            PromptReplacement(modality="detr", target=[detr_token_id],
                               replacement=get_detr_replacement),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    EchoPrimeMultiModalProcessor,
    info=EchoPrimeProcessingInfo,
    dummy_inputs=EchoPrimeDummyInputsBuilder,
)
class EchoPrimeQwen3ForCausalLMVLLM(nn.Module, SupportsMultiModal, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm.model.": "language_model.model.",
            "lm.lm_head.": "language_model.lm_head.",
        }
    )
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping
    embedding_modules = Qwen3ForCausalLM.embedding_modules

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.language_model = Qwen3ForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        hidden = config.text_config.hidden_size
        # 4-layer LayerNorm/Linear/GELU/Linear, matching modeling.py's HF-side projectors and
        # Darya's real EchoVLM shape exactly (report_generation/sft_thinking/model.py:67-79,
        # verified directly -- corrected during Task 2's review, see spec section 1). Index [1]
        # below (dtype read) still lands on the first Linear with this 4-layer shape.
        self.clip_projector = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM), nn.Linear(CLIP_EMBED_DIM, hidden),
            nn.GELU(), nn.Linear(hidden, hidden))
        self.detr_projector = nn.Sequential(
            nn.LayerNorm(DETR_EMBED_DIM), nn.Linear(DETR_EMBED_DIM, hidden),
            nn.GELU(), nn.Linear(hidden, hidden))

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN
        if modality == "clip":
            return CLIP_TOKEN
        if modality == "detr":
            return DETR_TOKEN
        return None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # NOTE: revisit this body against Step 1's findings before trusting it -- written to
        # match the SINGLE-call-with-both-modalities-present hypothesis (matches how llava
        # variants handling multiple modalities in one class are structured). If Step 1 shows
        # vLLM calls this once per modality instead, split into embed_multimodal receiving only
        # one of clip_embeds/detr_embeds per call, matching whatever signature it actually gets.
        results = []
        clip_embeds = kwargs.pop("clip_embeds", None)
        if clip_embeds is not None:
            if isinstance(clip_embeds, list):
                clip_embeds = torch.cat(clip_embeds, dim=0)
            dtype = self.clip_projector[1].weight.dtype
            results.append(self.clip_projector(clip_embeds.to(dtype)))
        detr_embeds = kwargs.pop("detr_embeds", None)
        if detr_embeds is not None:
            if isinstance(detr_embeds, list):
                detr_embeds = torch.cat(detr_embeds, dim=0)
            dtype = self.detr_projector[1].weight.dtype
            results.append(self.detr_projector(detr_embeds.to(dtype)))
        if not results:
            return []
        return torch.cat(results, dim=0) if len(results) > 1 else results[0]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

Note the `LayerNorm`/`Linear` index change from the old single-`nn.Sequential(Linear, GELU, Linear)` (`self.projector[0]`) to the new `nn.Sequential(LayerNorm, Linear)` — `weight.dtype` now reads off index `[1]` (the `Linear`), not `[0]`.

- [ ] **Step 3: Update `register_vllm_model.py` if it references the old single-projector class shape**

Read `packages/echoprime_track/register_vllm_model.py` first — if it only calls `ModelRegistry.register_model(...)` by class reference with no shape-specific logic, no change needed. If it does anything with `projector`/`image` modality names directly, update those references to match Task 2/Step 2 above.

- [ ] **Step 4: Manual verification against a real checkpoint on the AMD box**

This can't be a fast unit test (needs a real vLLM engine + real checkpoint). Run a standalone smoke script that loads the Task 3 checkpoint through vLLM directly (mirroring whatever smoke-test pattern `docs/ROCM_VALIDATION.md` already documents for this track) and generates from a prompt containing both a `CLIP_TOKEN` run and a `DETR_TOKEN` run with matching `clip`/`detr` tensors in `multi_modal_data`. Confirm it doesn't raise a shape/placeholder-count error and produces non-garbage output (compare against the same known-working smoke check `docs/ROCM_VALIDATION.md` already describes for the single-modality version, job 185987).

- [ ] **Step 5: Commit**

```bash
git add packages/echoprime_track/vllm_model.py packages/echoprime_track/register_vllm_model.py
git commit -m "$(cat <<'EOF'
register EchoPrimeQwen3ForCausalLMVLLM for two modalities (clip, detr)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update the verl multi_modal_inputs patch for the new key names

**Files:**
- Modify: `external/verl-hf-rollout-registry.patch`

**Interfaces:**
- Consumes: the fact that Task 8's agent loop will emit `AgentLoopOutput.multi_modal_data` under new keys (defined in this task, consumed by Task 8 — the two tasks must agree on exact key names: `clip_embeddings`, `clip_counts`, `detr_embeddings`, `detr_counts`).

- [ ] **Step 1: Edit the patch's `_compute_multi_modal_inputs` hunk**

In `external/verl-hf-rollout-registry.patch`, change the `diff --git a/verl/experimental/agent_loop/agent_loop.py ...` hunk's added block from:

```python
+        if "view_embeddings" in output.multi_modal_data:
+            return {
+                "view_embeddings": output.multi_modal_data["view_embeddings"],
+                "view_counts": torch.tensor([output.multi_modal_data["view_counts"]]),
+            }
```

to:

```python
+        # EchoSonarVideo patch: this project's echoprime_track agent loop
+        # (packages/echoprime_track/echoprime_tool_agent_loop.py) carries clip/detr features
+        # through as plain tensors, not something a real HF processor would produce -- forward
+        # them straight through to the actor's forward() (modeling.py's EchoPrimeQwen3ForCausalLM
+        # expects exactly these four keys).
+        if "clip_embeddings" in output.multi_modal_data or "detr_embeddings" in output.multi_modal_data:
+            return {
+                "clip_embeddings": output.multi_modal_data.get("clip_embeddings"),
+                "clip_counts": torch.tensor([output.multi_modal_data.get("clip_counts", 0)]),
+                "detr_embeddings": output.multi_modal_data.get("detr_embeddings"),
+                "detr_counts": torch.tensor([output.multi_modal_data.get("detr_counts", 0)]),
+            }
```

- [ ] **Step 2: Re-apply the patch to the pinned submodule and confirm it applies cleanly**

```bash
cd external/verl
git apply --check ../../external/verl-hf-rollout-registry.patch
git apply ../../external/verl-hf-rollout-registry.patch
cd ../..
python scripts/check_train_env.py
```

Expected: `check_train_env.py` passes (it already asserts patches are present, per CLAUDE.md's verl gotchas section — confirm it covers this patch specifically; if it only checks the video-nccl patch, note that as a gap but don't expand this task's scope to fix `check_train_env.py`'s coverage unless it's a one-line addition).

- [ ] **Step 3: Commit**

```bash
git add external/verl-hf-rollout-registry.patch
git commit -m "$(cat <<'EOF'
update verl patch for clip/detr multi_modal_data keys, not view_embeddings

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `grid.py` — pure functions for indexing the precomputed 393-token grid

**Files:**
- Create: `packages/echoprime_track/grid.py`
- Test: `packages/echoprime_track/tests/test_grid.py`

**Interfaces:**
- Produces: `grid.py::N_TEMPORAL_GROUPS = 8`, `TOKENS_PER_GROUP = 49`, `FRAMES_PER_GROUP = 2`; `resolve_temporal_group(frame_indices: list[int]) -> int`; `group_token_slice(group: int) -> slice`; `spatial_subset(group: int, bbox, grid_side: int = 7) -> list[int]` (returns absolute token indices into the 393-token grid, already restricted to the given bbox within that group). Consumed by Task 8's agent loop.

- [ ] **Step 1: Write the failing tests**

```python
# packages/echoprime_track/tests/test_grid.py
import pytest

from echoprime_track.grid import (
    resolve_temporal_group, group_token_slice, spatial_subset,
    N_TEMPORAL_GROUPS, TOKENS_PER_GROUP,
)


def test_resolve_temporal_group_maps_frame_pairs():
    # 16 frames -> 8 groups of 2: frame // 2
    assert resolve_temporal_group([0]) == 0
    assert resolve_temporal_group([1]) == 0
    assert resolve_temporal_group([2]) == 1
    assert resolve_temporal_group([15]) == 7


def test_resolve_temporal_group_uses_first_valid_index_when_spanning_groups():
    # spec: cap at 1 group per call -- use the group of the first valid index
    assert resolve_temporal_group([3, 10]) == 1  # frame 3 -> group 1


def test_resolve_temporal_group_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_temporal_group([16])
    with pytest.raises(ValueError):
        resolve_temporal_group([-1])
    with pytest.raises(ValueError):
        resolve_temporal_group([])


def test_group_token_slice_layout():
    # token 0 is the global/CLS token; group 0 is tokens [1, 50), group 7 is [344, 393)
    assert group_token_slice(0) == slice(1, 50)
    assert group_token_slice(7) == slice(344, 393)
    assert (group_token_slice(7).stop - group_token_slice(7).start) == TOKENS_PER_GROUP


def test_group_token_slice_rejects_out_of_range_group():
    with pytest.raises(ValueError):
        group_token_slice(N_TEMPORAL_GROUPS)
    with pytest.raises(ValueError):
        group_token_slice(-1)


def test_spatial_subset_full_bbox_returns_whole_group():
    idxs = spatial_subset(group=0, bbox=(0.0, 0.0, 1.0, 1.0))
    assert idxs == list(range(1, 50))


def test_spatial_subset_partial_bbox_returns_fewer_tokens():
    # top-left quadrant of the 7x7 grid
    idxs = spatial_subset(group=0, bbox=(0.0, 0.0, 0.5, 0.5))
    assert 0 < len(idxs) < 49
    assert all(1 <= i < 50 for i in idxs)


def test_spatial_subset_offsets_by_group():
    idxs_g0 = spatial_subset(group=0, bbox=(0.0, 0.0, 1.0, 1.0))
    idxs_g1 = spatial_subset(group=1, bbox=(0.0, 0.0, 1.0, 1.0))
    assert idxs_g1 == [i + 49 for i in idxs_g0]


def test_spatial_subset_empty_bbox_raises():
    with pytest.raises(ValueError):
        spatial_subset(group=0, bbox=(0.9, 0.9, 0.91, 0.91), grid_side=7)  # no cell center falls inside
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/echoprime_track && python -m pytest tests/test_grid.py -v`
Expected: FAIL — `grid.py` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# packages/echoprime_track/grid.py
"""Pure indexing over the precomputed (393, 768) per-view clip-token grid (Darya's
report_generation/data/clip_tokens_{train,test}.h5, confirmed this session: 393 = 1 global/CLS
token + 8 temporal groups (16 frames -> 8 groups of 2) x 49 spatial tokens (7x7 grid)).
select_frames/zoom (echoprime_tool_agent_loop.py) use these to resolve a model tool call into
absolute token indices -- no live EchoPrime encoder inference, ever (spec: docs/superpowers/
specs/2026-09-15-echoprime-tool-grpo-design.md section 2).
"""
import math

N_TEMPORAL_GROUPS = 8
FRAMES_PER_GROUP = 2
TOKENS_PER_GROUP = 49
GRID_SIDE = 7
N_FRAMES = N_TEMPORAL_GROUPS * FRAMES_PER_GROUP  # 16
GLOBAL_TOKEN_COUNT = 1  # index 0


def resolve_temporal_group(frame_indices: list) -> int:
    """First valid frame index's temporal group (0..7). Capped at 1 group per tool call
    (spec's "Token budget" decision) -- callers with indices spanning multiple groups get the
    first one; the caller (echoprime_tool_agent_loop.py) is responsible for telling the model
    that in the tool response text."""
    valid = [i for i in frame_indices if isinstance(i, int) and 0 <= i < N_FRAMES]
    if not valid:
        raise ValueError(f"no valid frame indices in {frame_indices!r} (need 0..{N_FRAMES - 1})")
    return valid[0] // FRAMES_PER_GROUP


def group_token_slice(group: int) -> slice:
    """Absolute [start, stop) token-index slice into the 393-token grid for one temporal
    group. Group 0 starts at index 1 (index 0 is the global/CLS token)."""
    if not (0 <= group < N_TEMPORAL_GROUPS):
        raise ValueError(f"group {group} out of range 0..{N_TEMPORAL_GROUPS - 1}")
    start = GLOBAL_TOKEN_COUNT + group * TOKENS_PER_GROUP
    return slice(start, start + TOKENS_PER_GROUP)


def spatial_subset(group: int, bbox, grid_side: int = GRID_SIDE) -> list:
    """Absolute token indices within `group`'s 49-token block whose spatial cell CENTER falls
    inside `bbox` (left, top, right, bottom), each in [0, 1] normalized coordinates over the
    view's frame -- same normalized convention as tool_env/bbox.py's callers elsewhere in this
    project, for consistency. Raises ValueError if no cell center falls inside (an empty
    selection is a bad tool call, not silently a no-op observation)."""
    left, top, right, bottom = bbox
    if not (left < right and top < bottom):
        raise ValueError(f"invalid bbox {bbox!r}: left<right and top<bottom required")
    base = group_token_slice(group).start
    out = []
    for row in range(grid_side):
        cy = (row + 0.5) / grid_side
        if not (top <= cy <= bottom):
            continue
        for col in range(grid_side):
            cx = (col + 0.5) / grid_side
            if left <= cx <= right:
                out.append(base + row * grid_side + col)
    if not out:
        raise ValueError(f"bbox {bbox!r} contains no grid cell center in group {group}")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/echoprime_track && python -m pytest tests/test_grid.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/echoprime_track/grid.py packages/echoprime_track/tests/test_grid.py
git commit -m "$(cat <<'EOF'
add grid.py: pure indexing into the precomputed 393-token clip grid

Backs select_frames/zoom -- no live encoder inference, just resolves a
tool call's frame_indices/bbox into absolute token indices already present
in the cached per-view grid.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Darya-format prompts + verifiable-only data scope in parquet generation

**Files:**
- Modify: `packages/echoprime_track/generate_grpo_parquet.py`
- Modify: `packages/echoprime_track/rl_dataset.py`
- Test: `packages/echoprime_track/tests/test_generate_grpo_parquet.py`

**Interfaces:**
- Consumes: `CLIP_TOKEN`, `DETR_TOKEN` from Task 2; `RT_DETR_CLASSES` convention from `report_generation/sft_thinking/dataset.py` (reused, not re-derived).
- Produces: `generate_grpo_parquet.py::build_row(rl_rec, clip_h5, detr_h5) -> dict | None` (returns `None` for a row whose `question_type` isn't in the verifiable set — caller filters those out); the row's `extra_info` carries `dicom_uuids_by_view: dict[str, str]` (view_name -> dicom_uuid), needed by the reward plan's DETR-grounding lookup.

- [ ] **Step 1: Write the failing test for `build_row`'s data-scope filter and new prompt shape**

```python
# packages/echoprime_track/tests/test_generate_grpo_parquet.py
import json

import h5py
import numpy as np
import pytest

from echoprime_track.generate_grpo_parquet import build_row, VERIFIABLE_QUESTION_TYPES


@pytest.fixture
def small_h5s(tmp_path):
    clip_path = tmp_path / "clip.h5"
    detr_path = tmp_path / "detr.h5"
    with h5py.File(clip_path, "w") as f:
        f.create_dataset("dicom-A4C", data=np.zeros((393, 768), dtype=np.float32))
    with h5py.File(detr_path, "w") as f:
        pass  # no detections for this dicom -- must be handled, not required
    return str(clip_path), str(detr_path)


def _rl_rec(question_type="abnormality_classification"):
    return {
        "study_uuid": "study-1",
        "question": "Is there mitral regurgitation?",
        "question_type": question_type,
        "reward_key": {"kind": "yesno", "target": "no", "gold": {}},
        "dicoms_by_view": {"A4C": "dicom-A4C"},
    }


def test_build_row_includes_verifiable_question_types(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec("abnormality_classification"), clip_h5, detr_h5)
    assert row is not None
    assert row["data_source"] == "echoprime_grpo"
    assert "A4C:" in row["prompt"][1]["content"]
    assert row["extra_info"]["dicom_uuids_by_view"] == {"A4C": "dicom-A4C"}


def test_build_row_excludes_unverifiable_question_types(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec("structure_description"), clip_h5, detr_h5)
    assert row is None


def test_verifiable_question_types_is_exactly_two():
    assert VERIFIABLE_QUESTION_TYPES == {"abnormality_classification", "abnormality_list"}


def test_build_row_prompt_has_clip_token_count_matching_grid(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec(), clip_h5, detr_h5)
    from echoprime_track.modeling import CLIP_TOKEN
    assert row["prompt"][1]["content"].count(CLIP_TOKEN) == 393
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/echoprime_track && python -m pytest tests/test_generate_grpo_parquet.py -v`
Expected: FAIL — new `build_row` signature and `VERIFIABLE_QUESTION_TYPES` don't exist yet.

- [ ] **Step 3: Rewrite `build_row`**

```python
# packages/echoprime_track/generate_grpo_parquet.py
"""GRPO parquet builder for the frozen-EchoPrime + Qwen3-8B-text track.

Prompts now match Darya's real SFT observation format (report_generation/sft_thinking/
dataset.py::__getitem__, reused verbatim): per view, "{view_name}:\n" + 393 CLIP_TOKEN
placeholders + (if detections exist) "Structure features:\n" + one DETR_TOKEN per detected
class. Training/eval data is restricted to verifiable question types only (spec: docs/
superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md section 4) -- structure_description/
conclusion/full_report fall back to unverifiable entity_F1 for reward and are excluded here,
at generation time, not filtered later.
"""
import json

from echoprime_track.dataset import SYSTEM_PROMPT
from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN

_DATA_SOURCE = "echoprime_grpo"
_AGENT_NAME = "echoprime_tool_agent"  # Task 8 registers the new multi-turn loop under this name

VERIFIABLE_QUESTION_TYPES = {"abnormality_classification", "abnormality_list"}

# Same 7 RT-DETR structure classes report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES
# uses -- kept in sync manually (small, stable, not worth a cross-repo import).
RT_DETR_CLASSES = {
    0: "Left Ventricle", 1: "Left Atrium", 2: "Right Atrium", 3: "Right Ventricle",
    4: "Mitral Valve", 5: "Tricuspid Valve", 6: "LVOT Area",
}
NUM_FRAMES = 16


def _detr_class_ids_present(detr_h5, dicom_uuid: str) -> list:
    if dicom_uuid not in detr_h5:
        return []
    grp = detr_h5[dicom_uuid]
    classes = set()
    for frame_idx in range(NUM_FRAMES):
        key = f"frame_{frame_idx}"
        if key not in grp:
            continue
        for cls_id in grp[key]["classes"][:]:
            classes.add(int(cls_id))
    return sorted(classes)


def _view_block(view_name: str, dicom_uuid: str, clip_h5, detr_h5) -> str:
    parts = [f"{view_name}:\n"]
    n_clip = clip_h5[dicom_uuid]["tokens"].shape[0] if dicom_uuid in clip_h5 else 0
    parts.append(CLIP_TOKEN * n_clip)
    class_ids = _detr_class_ids_present(detr_h5, dicom_uuid)
    if class_ids:
        parts.append("\nStructure features:\n")
        parts.append(DETR_TOKEN * len(class_ids))
    return "".join(parts)


def build_row(rl_rec: dict, clip_h5, detr_h5) -> dict | None:
    """Returns None for a row whose question_type isn't verifiable (spec section 4) --
    caller drops those rather than writing them to the parquet at all."""
    if rl_rec["question_type"] not in VERIFIABLE_QUESTION_TYPES:
        return None

    dicoms_by_view = rl_rec["dicoms_by_view"]  # {view_name: dicom_uuid}, from build/rl.jsonl
    blocks = [_view_block(view, dicom_uuid, clip_h5, detr_h5)
              for view, dicom_uuid in dicoms_by_view.items()]
    user_turn = "\n".join(blocks) + f"\n{rl_rec['question']}"

    return {
        "data_source": _DATA_SOURCE,
        "agent_name": _AGENT_NAME,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_turn}],
        "reward_model": {"ground_truth": json.dumps(rl_rec["reward_key"]), "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {"study_uuid": rl_rec["study_uuid"],
                        "question_type": rl_rec["question_type"],
                        "dicom_uuids_by_view": dicoms_by_view,
                        "need_tools_kwargs": False},
    }


def write_parquet(rows: list, path: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)
```

Note: this assumes `build/rl.jsonl` records carry a `dicoms_by_view` field (`{view_name: dicom_uuid}`). Confirmed this session: they don't — `rl.jsonl`/`eval.jsonl` rows carry `overview.views[]` (`{"view", "frame", "frame_count"}`), one entry per raw acquisition (a study can have multiple dicoms for the same `view_name`, e.g. 2-3 duplicate A4C clips), dicom_uuid parsed from the `frame` path per `packages/data_core/data/views.py::parse_clip_dirname`'s convention. Build `dicoms_by_view` by grouping `overview.views` by `view_name` and picking one dicom per group — **prefer one already present in Darya's clip h5** when a group has multiple candidates (open the h5 once, check membership before picking), falling back to any candidate (e.g. the first) when none of a view's candidates are covered. This is a strict improvement over Darya's own `_select_dicoms` (`report_generation/sft_thinking/dataset.py`), which picks `random.choice(group)` with no coverage awareness — preferring a covered candidate only ever increases effective coverage, never deviates from what her SFT checkpoint already tolerates (a missing dicom → that view's clip-token block is silently empty, spec §1). `_view_block` (Step 3 above) already handles a picked-but-uncovered dicom correctly (`n_clip = ... if dicom_uuid in clip_h5 else 0`) — no change needed there.

- [ ] **Step 4: Update `rl_dataset.py::EchoPrimeGRPODataset.__getitem__` to build the same prompt shape and load clip/detr features from the h5 caches instead of the old pooled-embedding `.pt` cache**

Read the current `__getitem__` (continues past what's shown in this plan's research) before editing — it needs to: open `clip_h5`/`detr_h5` (lazy, same fork-safe pattern `report_generation/sft_thinking/dataset.py::EchoVQAThinkingDataset.clip_h5`/`detr_h5` properties use), rebuild the identical per-view prompt text `build_row` already baked into the parquet's `prompt` column (or — simpler and less duplicative — just tokenize `row["prompt"]` directly via `apply_chat_template`, since Task 7/Step 3 already put the fully-formed text there, avoiding rebuilding it twice from raw h5 lookups). Prefer the second approach: it removes the current file's duplicated prompt-construction logic entirely, which is itself a small simplification.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/echoprime_track && python -m pytest tests/test_generate_grpo_parquet.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Generate a real (small) parquet to sanity-check row counts, using the coverage numbers Task 1 already recorded (train ~40%, eval ~41% — informational, not a gate, per spec §1)**

Generate the real train/val parquets (whatever CLI this project already exposes for that — check for an existing `build_grpo_parquet.sh`-equivalent for this track, or write one following the same `--study-list`/`--shuffle-seed` convention CLAUDE.md's data-pipeline section requires; never rely on `--limit` alone). Record the actual row/study counts once filtered to `VERIFIABLE_QUESTION_TYPES` — don't assume SPEC.md's question-type-mix percentages hold exactly (spec §4).

- [ ] **Step 7: Commit**

```bash
git add packages/echoprime_track/generate_grpo_parquet.py packages/echoprime_track/rl_dataset.py \
        packages/echoprime_track/tests/test_generate_grpo_parquet.py
git commit -m "$(cat <<'EOF'
switch parquet generation to Darya-format prompts, verifiable-only scope

Restricts training/eval data to abnormality_classification and
abnormality_list (deterministic outcome scoring) and rebuilds each view's
prompt block as full clip-token + DETR-structure-token placeholders instead
of one pooled VIEW_TOKEN, matching what the SFT checkpoint was actually
trained on.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: New multi-turn agent loop with `select_frames`/`zoom`

**Files:**
- Create: `packages/echoprime_track/echoprime_tool_agent_loop.py`
- Test: `packages/echoprime_track/tests/test_echoprime_tool_agent_loop.py` (pure-logic pieces only — tool-call resolution, prompt-splicing text construction; the actual `async run()` against a real vLLM server is manual verification, same as Task 4)

**Interfaces:**
- Consumes: `grid.py` (Task 6) for frame/bbox resolution; `tool_env.parse.parse_action` (existing, reused) for `<tool_call>`/`<answer>` extraction; `CLIP_TOKEN`, `DETR_TOKEN` from `modeling.py` (Task 2); the `clip_embeddings`/`clip_counts`/`detr_embeddings`/`detr_counts` key names Task 5's patch expects.
- Produces: `EchoPrimeToolAgentLoop`, registered under agent name `"echoprime_tool_agent"` (matches Task 7's `_AGENT_NAME`).

- [ ] **Step 1: Write the failing tests for the pure tool-call-resolution helper (no vLLM/async needed)**

```python
# packages/echoprime_track/tests/test_echoprime_tool_agent_loop.py
import torch

from echoprime_track.echoprime_tool_agent_loop import resolve_tool_call


def _fake_grid():
    # (393, 768) grid where every token's first value is its own index, for easy assertion
    g = torch.zeros(393, 768)
    for i in range(393):
        g[i, 0] = i
    return g


def test_resolve_select_frames_returns_group_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A4C", "frame_indices": [0, 1]}},
        view_grids={"A4C": grid})
    assert tokens.shape == (49, 768)
    assert tokens[0, 0].item() == 1  # group 0 starts at absolute index 1
    assert "A4C" in text


def test_resolve_zoom_returns_spatial_subset():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "zoom", "arguments": {"view": "A4C", "bbox": [0.0, 0.0, 0.5, 0.5],
                                        "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert 0 < tokens.shape[0] < 49
    assert tokens.shape[1] == 768


def test_resolve_unknown_view_returns_error_text_no_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A2C", "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown view" in text


def test_resolve_unknown_tool_name_returns_error():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_view", "arguments": {"view": "A4C"}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown tool" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/echoprime_track && python -m pytest tests/test_echoprime_tool_agent_loop.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Write `resolve_tool_call` and the surrounding agent loop**

```python
# packages/echoprime_track/echoprime_tool_agent_loop.py
"""Multi-turn AgentLoop for the frozen-EchoPrime + Qwen3-8B-text GRPO track, adding
select_frames/zoom on top of the turn-0 all-views observation (echoprime_agent_loop.py's
single-turn loop stays as-is for anything not using tools). Mirrors the SAME accumulate-and-
resend pattern verl's own ToolAgentLoop (external/verl/verl/experimental/agent_loop/
tool_agent_loop.py) uses for images: one request_id for the whole trajectory, prompt_ids grows
every turn, and image-modality data grows every turn -- each generate() call resends the FULL
accumulated prompt_ids/embeddings, relying on vLLM's own prefix caching (confirmed this session
by reading tool_agent_loop.py's _handle_generating_state/_handle_processing_tools_state
directly, not assumed).

No real HF processor exists for this text-only model (same reason echoprime_agent_loop.py gives
for not subclassing ToolAgentLoop), so tool-response placeholder text is built by hand, at the
EXACT token count grid.py's resolution produces -- never via apply_chat_template's image-count
inference, which needs a real processor we don't have.
"""
import os
import re
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from echoprime_track.grid import resolve_temporal_group, group_token_slice, spatial_subset
from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN
from tool_env.parse import parse_action

CACHE_DIR = os.environ.get("ECHOPRIME_VIDEO_CACHE_DIR", "build/echoprime_video_cache")
MAX_TOOL_TURNS = 4  # mirrors tool_env/budget.py's cap for the image-based track; revisit if
                     # that module's actual constant differs once checked directly.

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def resolve_tool_call(call: dict, view_grids: dict):
    """`call` = {"name": ..., "arguments": {...}} (tool_env.parse.parse_action's shape).
    `view_grids` = {view_name: (393, 768) tensor}, this study's cached clip-token grids.

    Returns (tokens, text): `tokens` is a (K, 768) tensor to splice as K new CLIP_TOKEN
    placeholders, or None on any error (unknown view/tool/bad args); `text` is what to show the
    model (either a description of what it now sees, or the error)."""
    name = call.get("name")
    args = call.get("arguments", {})
    view = args.get("view")
    grid = view_grids.get(view)
    if grid is None:
        avail = ", ".join(sorted(view_grids.keys()))
        return None, f"unknown view {view!r}; available: {avail}"

    if name == "select_frames":
        try:
            group = resolve_temporal_group(args.get("frame_indices", []))
        except ValueError as e:
            return None, str(e)
        tokens = grid[group_token_slice(group)]
        return tokens, f"{view}: frames from temporal group {group}"

    if name == "zoom":
        try:
            group = resolve_temporal_group(args.get("frame_indices", []))
            idxs = spatial_subset(group, tuple(args.get("bbox", ())))
        except ValueError as e:
            return None, str(e)
        tokens = grid[idxs]
        return tokens, f"{view}: zoom on {len(idxs)} patches in temporal group {group}"

    return None, f"unknown tool {name!r}; available: select_frames, zoom"


@register("echoprime_tool_agent")
class EchoPrimeToolAgentLoop(AgentLoopBase):
    """Turn 0 = Darya-format all-views prompt (already baked into the parquet's `prompt`
    column by generate_grpo_parquet.py -- this loop just tokenizes it, same as
    EchoPrimeAgentLoop). Then up to MAX_TOOL_TURNS rounds of: parse a <tool_call>, resolve it
    against the study's cached grid, splice the result in as new CLIP_TOKEN placeholders,
    continue generation. Stops early on <answer> or MAX_TOOL_TURNS, same shape as
    tool_env/env.py's turn budget for the image-based track."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        study_uuid = kwargs["extra_info"]["study_uuid"]
        dicom_uuids_by_view = kwargs["extra_info"]["dicom_uuids_by_view"]

        cache = torch.load(os.path.join(CACHE_DIR, f"{study_uuid}.pt"))
        # cache["grids"]: {view_name: (393, 768) tensor} -- built by whichever Task 7/Step 6
        # rebuild of build_video_cache.py now caches the raw grid per view, keyed by view_name
        # (not dicom_uuid) to match dicom_uuids_by_view's keys and this study's prompt text.
        view_grids = cache["grids"]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        clip_token_id = self.tokenizer.convert_tokens_to_ids(CLIP_TOKEN)
        detr_token_id = self.tokenizer.convert_tokens_to_ids(DETR_TOKEN)

        clip_chunks = []  # list of (K, 768) tensors, in prompt order
        detr_chunks = []  # populated from the turn-0 cache the same way (left as an exercise
                           # of Task 7's prompt construction -- this loop only needs to know
                           # the INITIAL count, read from prompt_ids, to seed clip_chunks
                           # correctly before any tool call runs)
        n_clip_turn0 = sum(1 for t in prompt_ids if t == clip_token_id)
        n_detr_turn0 = sum(1 for t in prompt_ids if t == detr_token_id)
        # Sanity check mirrors echoprime_agent_loop.py's existing assertion pattern.
        assert n_clip_turn0 == sum(g.shape[0] for g in view_grids.values()), (
            f"study {study_uuid}: prompt has {n_clip_turn0} {CLIP_TOKEN!r} placeholders but "
            f"the cached grids sum to {sum(g.shape[0] for g in view_grids.values())} -- "
            "prompt construction (generate_grpo_parquet.py) and the cache disagree")
        clip_chunks.append(torch.cat(list(view_grids.values()), dim=0))
        if n_detr_turn0:
            detr_chunks.append(cache["detr_features"])  # (n_detr_turn0, 256), same convention

        sampling_params = dict(sampling_params)
        logit_bias = dict(sampling_params.get("logit_bias") or {})
        logit_bias[clip_token_id] = -100
        logit_bias[detr_token_id] = -100
        sampling_params["logit_bias"] = logit_bias

        request_id = uuid4().hex
        response_ids_all = []
        metrics = {}

        for turn in range(MAX_TOOL_TURNS + 1):
            clip_tensor = torch.cat(clip_chunks, dim=0).unsqueeze(1)  # (N, 1, 768)
            with simple_timer(f"generate_turn{turn}", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=clip_tensor,
                )
            new_ids = output.token_ids
            response_ids_all += new_ids
            prompt_ids = prompt_ids + new_ids
            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

            parsed = parse_action(text)
            if parsed.answer is not None or not parsed.calls or turn == MAX_TOOL_TURNS:
                break

            call = parsed.calls[0]  # spec: 1 tool call resolved per turn
            tokens, tool_text = resolve_tool_call(call, view_grids)
            if tokens is None:
                tool_response = f"<tool_response>{tool_text}</tool_response>"
                tool_ids = self.tokenizer(tool_response, add_special_tokens=False)["input_ids"]
            else:
                clip_chunks.append(tokens)
                placeholder = CLIP_TOKEN * tokens.shape[0]
                tool_response = f"<tool_response>{tool_text}\n{placeholder}</tool_response>"
                tool_ids = self.tokenizer(tool_response, add_special_tokens=False)["input_ids"]

            response_ids_all += tool_ids
            prompt_ids = prompt_ids + tool_ids

        response_mask = [1] * len(response_ids_all)
        clip_counts = int(torch.cat(clip_chunks, dim=0).shape[0])
        detr_counts = int(torch.cat(detr_chunks, dim=0).shape[0]) if detr_chunks else 0

        result = AgentLoopOutput(
            prompt_ids=prompt_ids[: len(prompt_ids) - len(response_ids_all)],
            response_ids=response_ids_all[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=None,
            multi_modal_data={
                "clip_embeddings": torch.cat(clip_chunks, dim=0),
                "clip_counts": clip_counts,
                "detr_embeddings": torch.cat(detr_chunks, dim=0) if detr_chunks else None,
                "detr_counts": detr_counts,
            },
            num_turns=2 + turn,
            metrics=metrics,
            extra_fields={},
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result
```

This is the plan's single riskiest file — it depends on (a) Task 4/Step 1's confirmed vLLM merge semantics, (b) `build_video_cache.py` actually producing a `cache["grids"]`/`cache["detr_features"]` shape this loop assumes (not yet true — flagged inline above; if Task 7 didn't already extend the cache builder for this, do it now as part of this step, following `build_video_cache.py`'s existing per-study `.pt` file convention), and (c) `self.server_manager.generate` genuinely tolerating a growing `image_data` tensor across repeated calls under the same `request_id` the same way it tolerates a growing image list in `ToolAgentLoop` — confirmed for the STOCK image-list path, not yet confirmed for our tensor-embedding path specifically. Step 5 below is the check for (c).

- [ ] **Step 4: Run the pure-logic tests to verify they pass**

Run: `cd packages/echoprime_track && python -m pytest tests/test_echoprime_tool_agent_loop.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Manual smoke test against a real served vLLM instance on the AMD box**

Write a minimal standalone script (not committed — throwaway per CLAUDE.md's smoke-test rule) that calls `server_manager.generate` twice with the same `request_id`, the second call's `image_data` being the first call's tensor with extra rows appended, and confirms no error and a sane continuation. This is the direct test of assumption (c) above — if it fails, the fallback is a fresh `request_id` per turn (loses vLLM prefix-cache reuse but is still correct), which only requires changing `request_id=request_id` to `request_id=uuid4().hex` per turn in Step 3's loop.

- [ ] **Step 6: Register the new agent loop and MAX_TOOL_TURNS's real value**

Before committing, open `packages/tool_env/budget.py` and confirm the actual per-episode tool-call cap the image-based track uses; set `MAX_TOOL_TURNS` in Step 3's code to match (or to a deliberately different value, with a one-line comment saying why it diverges) — don't leave the placeholder `4` from Step 3 unexamined.

- [ ] **Step 7: Commit**

```bash
git add packages/echoprime_track/echoprime_tool_agent_loop.py \
        packages/echoprime_track/tests/test_echoprime_tool_agent_loop.py
git commit -m "$(cat <<'EOF'
add EchoPrimeToolAgentLoop: multi-turn select_frames/zoom rollout

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Config wiring

**Files:**
- Modify: `packages/verl_bridge/configs/echoprime_grpo.yaml`
- Modify: `packages/verl_bridge/configs/echoprime_agent_loop.yaml`

**Interfaces:**
- Consumes: `"echoprime_tool_agent"` (Task 8's registered name), `build/echoprime_sft_init` (Task 3's output dir).

- [ ] **Step 1: Point `model.path` at the real SFT-init checkpoint**

In `echoprime_grpo.yaml`, change:
```yaml
    path: build/echoprime_cold_start_checkpoint
```
to:
```yaml
    path: build/echoprime_sft_init
```

- [ ] **Step 2: Read `echoprime_agent_loop.yaml` and add the new agent loop's registration entry alongside the existing `echoprime_agent` one, without removing it (both loops stay usable — `echoprime_agent` for anything still using the old single-turn path, `echoprime_tool_agent` for the new one)**

Match whatever schema the existing entry already uses (likely a list of `{name, path}` or similar — read the file first, don't guess the schema).

- [ ] **Step 3: Update `trainer.experiment_name` to reflect the new run**

```yaml
  experiment_name: grpo-echoprime-qwen3-8b-tools-v1
```

- [ ] **Step 4: Commit**

```bash
git add packages/verl_bridge/configs/echoprime_grpo.yaml packages/verl_bridge/configs/echoprime_agent_loop.yaml
git commit -m "$(cat <<'EOF'
wire config to the real SFT-init checkpoint and the new tool agent loop

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (observation format) → Tasks 2, 3, 4, 7. §2 (tools) → Task 6, 8. §3 (agent loop) → Task 8. §4 (data scope) → Task 7. §5 (reward) → deliberately NOT here, separate plan (`2026-09-15-echoprime-verifiable-reward.md`), Task 7 only threads `dicom_uuids_by_view` through for it.
- **Known real risk carried forward, not hidden:** Task 8/Step 3's assumption that `cache["grids"]`/`cache["detr_features"]` exist in the shape assumed requires `build_video_cache.py` to be extended (not just reused) — this plan didn't give that its own task because the exact current shape of `build_video_cache.py`'s output needs to be read first (it currently produces `cache["embeddings"]`, the OLD pooled 512-dim format, per `echoprime_agent_loop.py`'s existing `cache["embeddings"]` read) — treat extending `build_video_cache.py` to also cache the raw `(393, 768)` grid (copied straight from Darya's h5, not recomputed) and per-study DETR features as a required part of Task 7 (parquet/cache generation), added when that task is actually executed, not deferred silently.
- **Type consistency check:** `clip_embeddings`/`clip_counts`/`detr_embeddings`/`detr_counts` names are used identically across Task 2 (`modeling.py::forward`), Task 4 (`vllm_model.py`'s `clip_embeds`/`detr_embeds` kwarg names — note the deliberate naming difference: HF-side params are `clip_embeddings`/`detr_embeddings`, vLLM multimodal kwargs are `clip_embeds`/`detr_embeds`, matching each side's own existing convention (`view_embeddings` vs `image_embeds` did the same in the original code) — not a typo), Task 5 (patch), and Task 8 (agent loop output). Confirmed consistent.
