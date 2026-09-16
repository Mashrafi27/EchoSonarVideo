# EchoPrime verifiable-only reward: canonical taxonomy + per-kind tool bonus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current open free-text `abnormality_list` outcome scoring with Darya's closed 11-canonical-finding taxonomy + IoU, and replace the flat `tool_bonus` with a per-kind rule (yesno: gated on correctness; set: scaled by real DETR bbox-overlap grounding), plus per-question-type reward normalization before GRPO batch pooling.

**Architecture:** `data_core.reward` gains two new pure-logic modules (`findings.py` for the canonical taxonomy, `grounding.py` for bbox IoU against DETR detections) that `score.py` and `verl_bridge/reward.py` wire together. The DETR-grounding path is the one place this plan crosses `score.py`'s previously-pure, no-I/O contract — isolated to `verl_bridge/reward.py` (which already does process-level state, e.g. the episode counter) rather than leaking into `data_core.reward.score`, which stays pure and independently testable.

**Tech Stack:** Python, h5py, PIL (frame dimensions, offline only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md` (section 5)

## Global Constraints

- This plan assumes Task 7 of `docs/superpowers/plans/2026-09-15-echoprime-observation-tools-rollout.md` has landed — specifically that parquet rows carry `extra_info.dicom_uuids_by_view` and are restricted to `abnormality_classification`/`abnormality_list` question types. Do not start this plan's Task 4 before that.
- `data_core.reward.score` must stay pure/model-free/no-I/O for every kind EXCEPT where this plan explicitly says otherwise (the `set`-kind DETR grounding term) — and even there, the I/O (opening the DETR h5, reading `boxes_xyxy`) happens in `verl_bridge/reward.py`, with `data_core.reward.grounding` taking already-loaded numpy arrays as plain function arguments, never a file path. This keeps `data_core.reward` unit-testable without any real data file on disk.
- `text` kind (`structure_description`/`conclusion`/`full_report`) reward code is UNCHANGED — not touched by this plan (spec §5, item 6: moot once the data-scope restriction lands, kept only for any other caller of `total_reward`).
- No change to `r_fmt`/format scoring, and no length penalty — both explicit user decisions (spec §5, items 3 and 4).

---

## Task 1: Canonical-11-finding taxonomy replaces open free-text `set`-kind scoring

**Files:**
- Create: `packages/data_core/reward/findings.py`
- Modify: `packages/data_core/reward/score.py` (`score_set`)
- Modify: `packages/data_core/data/builders.py` (`_reward_key`'s `set` branch)
- Test: `packages/data_core/tests/test_findings.py`
- Modify: `packages/data_core/tests/test_reward_score.py` (`test_score_set` — old open-text assertions no longer hold)
- Modify: `packages/data_core/tests/test_builders.py` if it exercises `_reward_key`'s `set` branch (check first)

**Interfaces:**
- Produces: `findings.py::CANONICAL_FINDINGS: set[str]`, `extract_canonical_findings(text: str) -> set[str]`, `iou(a: set, b: set) -> float`. Consumed by Task 2 (finding→DETR-class mapping keys off `CANONICAL_FINDINGS`) and Task 4 (`score_set`'s new IoU-based scoring).

- [x] **Step 1: Write the failing tests for the ported taxonomy**

```python
# packages/data_core/tests/test_findings.py
from data_core.reward.findings import CANONICAL_FINDINGS, extract_canonical_findings, iou


def test_canonical_findings_has_eleven_entries():
    assert len(CANONICAL_FINDINGS) == 11


def test_extract_canonical_findings_matches_aliases():
    text = "- TR\n- mitral regurgitation\n- LV enlargement"
    found = extract_canonical_findings(text)
    assert found == {"tricuspid regurgitation", "mitral valve regurgitation",
                      "left ventricular enlargement"}


def test_extract_canonical_findings_no_abnormalities_returns_empty():
    assert extract_canonical_findings("No significant abnormalities identified.") == set()


def test_extract_canonical_findings_ignores_unrecognized_bullets():
    # a bullet that isn't one of the 11 canonical findings (or an alias) is dropped, not
    # kept as an open free-text entry -- this is the whole point of the closed taxonomy
    text = "- some made up finding\n- aortic stenosis"
    assert extract_canonical_findings(text) == {"aortic stenosis"}


def test_iou_edges():
    assert iou(set(), set()) == 1.0
    assert iou({"a"}, set()) == 0.0
    assert iou(set(), {"a"}) == 0.0
    assert iou({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(iou({"a", "b"}, {"a", "c"}) - 1 / 3) < 1e-9
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd packages/data_core && python -m pytest tests/test_findings.py -v`
Expected: FAIL — module doesn't exist.

- [x] **Step 3: Write `findings.py`, porting `report_generation/grpo/rewards.py`'s taxonomy verbatim**

```python
# packages/data_core/reward/findings.py
"""Closed 11-canonical-finding taxonomy for `abnormality_list` outcome scoring, ported
verbatim from report_generation/grpo/rewards.py (Darya's own GRPO reward) rather than this
project's previous open free-text bullet extraction (data_core.data.answers.finding_set) --
a closed vocabulary + alias table is meaningfully harder to game than raw substring presence
(spec: docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md section 5, item 2).

Also the same 11 categories packages/eval/diseases.py::disease_of already uses at question
level (its own rule table is QUESTION-text keyword rules; this one matches FINDING-text
inside a list answer -- different vocabulary distribution, kept as a separate table
deliberately, not merged).
"""
import re

CANONICAL_FINDINGS = {
    "tricuspid regurgitation",
    "mitral valve regurgitation",
    "left atrial enlargement",
    "aortic regurgitation",
    "left ventricular systolic function",
    "mitral valve calcification",
    "left ventricular enlargement",
    "aortic stenosis",
    "right atrial enlargement",
    "bicuspid aortic valve",
    "right ventricular enlargement",
}

_ALIASES = {
    "tricuspid regurgitation": ["tricuspid regurgitation", "tr"],
    "mitral valve regurgitation": ["mitral valve regurgitation", "mitral regurgitation", "mr"],
    "left atrial enlargement": ["left atrial enlargement", "la enlargement", "left atrium enlargement"],
    "aortic regurgitation": ["aortic regurgitation", "ar", "aortic insufficiency"],
    "left ventricular systolic function": [
        "left ventricular systolic function", "lv systolic function",
        "left ventricular systolic dysfunction", "lv systolic dysfunction",
    ],
    "mitral valve calcification": ["mitral valve calcification", "mitral calcification"],
    "left ventricular enlargement": ["left ventricular enlargement", "lv enlargement"],
    "aortic stenosis": ["aortic stenosis", "as"],
    "right atrial enlargement": ["right atrial enlargement", "ra enlargement", "ra dilation", "right atrial dilation"],
    "bicuspid aortic valve": ["bicuspid aortic valve", "bav"],
    "right ventricular enlargement": [
        "right ventricular enlargement", "rv enlargement", "rv dilation", "right ventricular dilation",
    ],
}

_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in _ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical

_NO_ABNORMALITY_RE = re.compile(r"no\s+(significant\s+)?abnormalit", re.I)


def extract_canonical_findings(text: str) -> set:
    """Free-text list answer -> set of canonical finding names. Bullets that don't match any
    known alias are DROPPED (not kept as open free text) -- the closed vocabulary is the
    entire point (see module docstring)."""
    text_lower = (text or "").lower()
    if _NO_ABNORMALITY_RE.search(text_lower):
        return set()

    text_lower = re.sub(r"the following abnormalities are identified:?\s*", "", text_lower)
    text_lower = re.sub(r"\d+\.\s*", "", text_lower)
    items = re.split(r"[-•\n;,]+", text_lower)

    found = set()
    for item in items:
        cleaned = item.strip().rstrip(".")
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned in _ALIAS_TO_CANONICAL:
            found.add(_ALIAS_TO_CANONICAL[cleaned])
            continue
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
            if len(alias) <= 3:
                continue
            if alias in cleaned or cleaned in alias:
                found.add(canonical)
                break
    return found


def iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd packages/data_core && python -m pytest tests/test_findings.py -v`
Expected: PASS (5 tests)

- [x] **Step 5: Wire `score_set` to use the new taxonomy + IoU instead of open free-text F1**

In `packages/data_core/reward/score.py`, replace:

```python
def score_set(pred_answer: str, target: list) -> float:
    return f1(finding_set(pred_answer or ""), set(target or []))
```

with:

```python
def score_set(pred_answer: str, target: list) -> float:
    from data_core.reward.findings import extract_canonical_findings, iou
    return iou(extract_canonical_findings(pred_answer or ""), set(target or []))
```

(local import to avoid a module-level circular-import risk between `score.py` and the new `findings.py` if `findings.py` ever needs anything from `score.py` later — check whether the rest of the file already does local imports for similar reasons, e.g. `sections.py`'s import, and match that convention; if `score.py` only ever does top-level imports, move this to the top instead for consistency.)

- [x] **Step 6: Update `packages/data_core/tests/test_reward_score.py::test_score_set`**

The old test (`ans = "- mitral regurgitation\n- lv dilation"`, expecting a partial-F1 result against `["mitral regurgitation"]`) exercised the OLD open-text behavior. Replace with:

```python
def test_score_set():
    ans = "- mitral valve regurgitation\n- left ventricular enlargement"
    assert score_set(ans, ["mitral valve regurgitation", "left ventricular enlargement"]) == 1.0
    assert score_set("No significant abnormalities.", []) == 1.0
    assert 0.0 < score_set(ans, ["mitral valve regurgitation"]) < 1.0
```

- [x] **Step 7: Update `_reward_key`'s `set` branch in `builders.py` to build gold targets from the same taxonomy**

```python
elif qtype == "abnormality_list":
    from data_core.reward.findings import extract_canonical_findings
    kind, target = "set", sorted(extract_canonical_findings(answer))
```

Check `packages/data_core/tests/test_builders.py` for any existing assertion on `_reward_key`'s `set`-kind output shape and update it the same way if present.

- [x] **Step 8: Run the full `data_core` test suite**

Run: `cd packages/data_core && python -m pytest -v`
Expected: PASS. Investigate and fix anything broken by the `score_set`/`_reward_key` signature-preserving-but-behavior-changing edit (should be limited to the two test files already touched, but verify).

- [x] **Step 9: Commit**

```bash
git add packages/data_core/reward/findings.py packages/data_core/reward/score.py \
        packages/data_core/data/builders.py packages/data_core/tests/test_findings.py \
        packages/data_core/tests/test_reward_score.py
git commit -m "$(cat <<'EOF'
switch abnormality_list outcome scoring to closed canonical-finding taxonomy

Replaces open free-text bullet extraction (arbitrary substring match) with
Darya's 11-canonical-finding taxonomy + alias table + IoU, ported from
report_generation/grpo/rewards.py -- meaningfully harder to game.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Finding -> RT-DETR class mapping

**Files:**
- Modify: `packages/data_core/reward/findings.py`
- Test: `packages/data_core/tests/test_findings.py`

**Interfaces:**
- Produces: `findings.py::FINDING_TO_DETR_CLASS: dict[str, int | None]` (keys = every name in `CANONICAL_FINDINGS`; value = RT-DETR class id 0-6, or `None` for findings RT-DETR has no class for). Consumed by Task 4's grounding wiring.

- [x] **Step 1: Write the failing tests**

```python
# append to packages/data_core/tests/test_findings.py
from data_core.reward.findings import FINDING_TO_DETR_CLASS, CANONICAL_FINDINGS


def test_finding_to_detr_class_covers_every_canonical_finding():
    assert set(FINDING_TO_DETR_CLASS.keys()) == CANONICAL_FINDINGS


def test_finding_to_detr_class_maps_covered_structures():
    assert FINDING_TO_DETR_CLASS["left ventricular enlargement"] == 0
    assert FINDING_TO_DETR_CLASS["right ventricular enlargement"] == 3
    assert FINDING_TO_DETR_CLASS["mitral valve calcification"] == 4
    assert FINDING_TO_DETR_CLASS["mitral valve regurgitation"] == 4
    assert FINDING_TO_DETR_CLASS["tricuspid regurgitation"] == 5
    assert FINDING_TO_DETR_CLASS["left atrial enlargement"] == 1
    assert FINDING_TO_DETR_CLASS["right atrial enlargement"] == 2
    assert FINDING_TO_DETR_CLASS["left ventricular systolic function"] == 0


def test_finding_to_detr_class_none_for_uncovered_aortic_valve_findings():
    # RT-DETR's 7 classes (report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES) have
    # no aortic valve class at all
    assert FINDING_TO_DETR_CLASS["aortic regurgitation"] is None
    assert FINDING_TO_DETR_CLASS["aortic stenosis"] is None
    assert FINDING_TO_DETR_CLASS["bicuspid aortic valve"] is None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd packages/data_core && python -m pytest tests/test_findings.py -v -k detr_class`
Expected: FAIL — `FINDING_TO_DETR_CLASS` doesn't exist yet.

- [x] **Step 3: Add the mapping to `findings.py`**

```python
# RT-DETR classes, matching report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES
# exactly (0=Left Ventricle, 1=Left Atrium, 2=Right Atrium, 3=Right Ventricle,
# 4=Mitral Valve, 5=Tricuspid Valve, 6=LVOT Area). None = no RT-DETR class covers this
# finding's structure at all (e.g. every aortic-valve finding -- RT-DETR has no aortic valve
# class) -- these contribute nothing to the grounding average (Task 4), never penalized.
FINDING_TO_DETR_CLASS = {
    "left ventricular enlargement": 0,
    "left ventricular systolic function": 0,
    "left atrial enlargement": 1,
    "right atrial enlargement": 2,
    "right ventricular enlargement": 3,
    "mitral valve regurgitation": 4,
    "mitral valve calcification": 4,
    "tricuspid regurgitation": 5,
    "aortic regurgitation": None,
    "aortic stenosis": None,
    "bicuspid aortic valve": None,
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd packages/data_core && python -m pytest tests/test_findings.py -v`
Expected: PASS (all tests in the file)

- [x] **Step 5: Commit**

```bash
git add packages/data_core/reward/findings.py packages/data_core/tests/test_findings.py
git commit -m "$(cat <<'EOF'
add finding -> RT-DETR class mapping for the grounding reward term

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: DETR bbox IoU grounding utility

**Files:**
- Create: `packages/data_core/reward/grounding.py`
- Create: `scripts/build_frame_dims.py`
- Test: `packages/data_core/tests/test_grounding.py`

**Interfaces:**
- Produces: `grounding.py::bbox_iou(a: tuple, b: tuple) -> float` (pure, both boxes same coordinate space); `normalize_detr_box(box_xyxy: tuple, frame_width: int, frame_height: int) -> tuple` (pixel xyxy -> normalized `[0,1]` left/top/right/bottom, matching `grid.py`'s bbox convention from the other plan); `ground_tool_calls(tool_calls: list, dicom_uuid: str, detr_class_id: int, detr_boxes_by_frame: dict, frame_dims: tuple) -> float` (returns the best IoU across all `zoom`/`select_frames` calls touching this dicom's frames — see Step 3 for the "select_frames = whole-frame bbox" convention). Consumed by Task 4.

DETR's `boxes_xyxy` are real pixel coordinates (confirmed this session by direct inspection: values up to ~374, not `[0,1]`) — frame pixel dimensions aren't stored in the h5 or in `frames.jsonl`, only derivable from the actual PNG files on disk (`frame_paths`). Precompute once, offline (Step 5), rather than opening PNGs during reward computation.

- [ ] **Step 1: Write the failing tests for the pure IoU math**

```python
# packages/data_core/tests/test_grounding.py
from data_core.reward.grounding import bbox_iou, normalize_detr_box, ground_tool_calls


def test_bbox_iou_identical_boxes():
    assert bbox_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0


def test_bbox_iou_disjoint_boxes():
    assert bbox_iou((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)) == 0.0


def test_bbox_iou_partial_overlap():
    # two unit-area boxes overlapping in a 0.5x0.5 region: intersection 0.25, union 1.75
    val = bbox_iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 1.5, 1.5))
    assert abs(val - 0.25 / 1.75) < 1e-9


def test_normalize_detr_box():
    # pixel box (100, 100, 200, 300) in a 400x400 frame -> (0.25, 0.25, 0.5, 0.75)
    out = normalize_detr_box((100.0, 100.0, 200.0, 300.0), frame_width=400, frame_height=400)
    assert out == (0.25, 0.25, 0.5, 0.75)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/data_core && python -m pytest tests/test_grounding.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write `bbox_iou`, `normalize_detr_box`, and `ground_tool_calls`**

```python
# packages/data_core/reward/grounding.py
"""Bbox-overlap grounding between a model's zoom/select_frames tool calls and RT-DETR-detected
structure boxes, for the abnormality_list tool_bonus (spec: docs/superpowers/specs/
2026-09-15-echoprime-tool-grpo-design.md section 5, item "set"). Pure math only -- no h5/file
I/O here (verl_bridge/reward.py, which already does process-level state, owns opening the h5
and the frame_dims lookup table and passes plain arrays/tuples in).

DETR's boxes_xyxy are real pixel coordinates (confirmed this session by direct inspection:
sample values up to ~374, not [0,1]) -- normalize_detr_box needs the frame's real pixel
dimensions, which aren't in the h5 or in frames.jsonl, only derivable from the actual PNG on
disk. scripts/build_frame_dims.py precomputes these once, offline.
"""


def bbox_iou(a: tuple, b: tuple) -> float:
    """Both boxes (left, top, right, bottom), same coordinate space (normalized [0,1] here)."""
    al, at, ar, ab = a
    bl, bt, br, bb = b
    inter_l = max(al, bl)
    inter_t = max(at, bt)
    inter_r = min(ar, br)
    inter_b = min(ab, bb)
    inter_w = max(0.0, inter_r - inter_l)
    inter_h = max(0.0, inter_b - inter_t)
    inter = inter_w * inter_h
    area_a = max(0.0, ar - al) * max(0.0, ab - at)
    area_b = max(0.0, br - bl) * max(0.0, bb - bt)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_detr_box(box_xyxy: tuple, frame_width: int, frame_height: int) -> tuple:
    left, top, right, bottom = box_xyxy
    return (left / frame_width, top / frame_height, right / frame_width, bottom / frame_height)


def ground_tool_calls(tool_calls: list, view: str, detr_boxes_norm: list) -> float:
    """`tool_calls`: list of {"name": ..., "arguments": {...}} dicts (tool_env.parse.parse_action's
    shape) already filtered to calls referencing `view`. `detr_boxes_norm`: list of already-
    normalized (left, top, right, bottom) DETR boxes for the relevant class, across whichever
    frames matter for this call. Returns the BEST (max) IoU across every call x box pair --
    "did the model ever point at roughly the right place", not an average (a model that zooms
    once correctly after one bad guess shouldn't be penalized for the bad guess).

    select_frames calls have no bbox (only frame_indices) -- treated as the whole-frame box
    (0, 0, 1, 1), same normalized convention grid.py uses elsewhere in this project. This gives
    select_frames calls touching the right view partial (low) credit even without spatial
    precision, while zoom calls that actually bound the structure score much higher."""
    if not tool_calls or not detr_boxes_norm:
        return 0.0
    best = 0.0
    for call in tool_calls:
        if call.get("name") == "zoom":
            bbox = tuple(call.get("arguments", {}).get("bbox", ()))
            if len(bbox) != 4:
                continue
        elif call.get("name") == "select_frames":
            bbox = (0.0, 0.0, 1.0, 1.0)
        else:
            continue
        for detr_box in detr_boxes_norm:
            best = max(best, bbox_iou(bbox, detr_box))
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/data_core && python -m pytest tests/test_grounding.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the offline frame-dimensions precompute script**

```python
# scripts/build_frame_dims.py
"""Run once, offline, before Task 4 of docs/superpowers/plans/
2026-09-15-echoprime-verifiable-reward.md needs to normalize DETR pixel boxes. Reads one PNG
per dicom (the first frame_path) to get (width, height) -- DETR boxes are pixel-space and this
dimension isn't stored anywhere else (checked this session: not in frames.jsonl, not in the
detections h5).

Usage:
    python scripts/build_frame_dims.py --frames-jsonl build/frames.jsonl \
        --out build/frame_dims.json
"""
import argparse
import json

from PIL import Image


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    dims = {}
    with open(args.frames_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for dc in rec["dicoms"]:
                dicom_uuid = dc["dicom_uuid"]
                if dicom_uuid in dims:
                    continue
                paths = dc.get("frame_paths") or []
                if not paths:
                    continue
                with Image.open(paths[0]) as img:
                    dims[dicom_uuid] = list(img.size)  # [width, height]

    with open(args.out, "w") as fh:
        json.dump(dims, fh)
    print(f"[frame_dims] wrote {len(dims)} entries to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the real script against both splits' frame sources**

```bash
python scripts/build_frame_dims.py --frames-jsonl <train frames source> --out build/frame_dims_train.json
python scripts/build_frame_dims.py --frames-jsonl <test frames source> --out build/frame_dims_test.json
```

Expected: prints a count roughly matching Task 1's earlier `[coverage]` needed-dicom count (from the other plan) for each split. Spot-check a couple of entries look like plausible echo frame dimensions (hundreds of pixels per side, not e.g. `[1,1]` or `[0,0]`).

- [ ] **Step 7: Commit**

```bash
git add packages/data_core/reward/grounding.py packages/data_core/tests/test_grounding.py \
        scripts/build_frame_dims.py
git commit -m "$(cat <<'EOF'
add DETR bbox IoU grounding utility + offline frame-dimension precompute

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Per-kind `tool_bonus` in `total_reward` / `compute_score`

**Files:**
- Modify: `packages/data_core/reward/score.py` (`total_reward`)
- Modify: `packages/verl_bridge/reward.py` (`compute_score`)
- Test: `packages/data_core/tests/test_reward_total.py`
- Test: `packages/verl_bridge/tests/test_reward.py`

**Interfaces:**
- Consumes: `data_core.reward.findings.FINDING_TO_DETR_CLASS` (Task 2), `data_core.reward.grounding.ground_tool_calls`/`normalize_detr_box` (Task 3), `extra_info.dicom_uuids_by_view` (from the other plan's Task 7).
- Produces: `total_reward(reward_key, completion, *, tool_calls=0, tool_bonus_coef=0.0, grounding_iou=None, ...)` — new `grounding_iou` param (a plain float, already computed by the caller; `data_core.reward.score` stays I/O-free by taking the number, not computing it).

- [ ] **Step 1: Write the failing tests for `total_reward`'s new per-kind `tool_bonus`**

```python
# add to packages/data_core/tests/test_reward_total.py
from data_core.reward.score import total_reward


def test_tool_bonus_yesno_gated_on_correct_answer():
    rk = {"kind": "yesno", "target": "yes", "gold": {}}
    correct = "<think>t</think>\n<tool_call>{}</tool_call>\n<answer>Yes.</answer>"
    wrong = "<think>t</think>\n<tool_call>{}</tool_call>\n<answer>No.</answer>"
    r_correct = total_reward(rk, correct, tool_calls=1, tool_bonus_coef=0.5)
    r_wrong = total_reward(rk, wrong, tool_calls=1, tool_bonus_coef=0.5)
    assert r_correct["tool_bonus"] == 0.5   # coef * outcome(1.0)
    assert r_wrong["tool_bonus"] == 0.0     # coef * outcome(0.0)


def test_tool_bonus_set_scaled_by_grounding_iou():
    rk = {"kind": "set", "target": ["left ventricular enlargement"], "gold": {}}
    sol = "<think>t</think>\n<tool_call>{}</tool_call>\n<answer>- left ventricular enlargement</answer>"
    r = total_reward(rk, sol, tool_calls=1, tool_bonus_coef=0.5, grounding_iou=0.8)
    assert abs(r["tool_bonus"] - 0.4) < 1e-9  # coef * grounding_iou


def test_tool_bonus_set_defaults_to_zero_without_grounding_iou():
    rk = {"kind": "set", "target": [], "gold": {}}
    sol = "<think>t</think>\n<answer>No significant abnormalities.</answer>"
    r = total_reward(rk, sol, tool_calls=0, tool_bonus_coef=0.5)
    assert r["tool_bonus"] == 0.0


def test_tool_bonus_text_kind_unchanged_flat():
    rk = {"kind": "text", "target": "x", "gold": {}}
    sol = "<think>t</think>\n<tool_call>{}</tool_call>\n<answer>x</answer>"
    r = total_reward(rk, sol, tool_calls=1, tool_bonus_coef=0.3)
    assert r["tool_bonus"] == 0.3  # unchanged flat behavior, spec section 5 item 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/data_core && python -m pytest tests/test_reward_total.py -v -k tool_bonus`
Expected: FAIL — `grounding_iou` param doesn't exist, old flat behavior still active.

- [ ] **Step 3: Rewrite `total_reward`'s tool_bonus logic**

```python
def total_reward(reward_key: dict, completion: str, *, tool_calls: int = 0,
                 tool_bonus_coef: float = 0.0, outcome_weight: float = 1.0,
                 format_weight: float = 0.2, question: str = "",
                 grounding_iou: float = None,
                 judge: JudgeClient = NullJudge()) -> dict:
    answer = extract_answer(completion)
    outcome = score_outcome(reward_key, answer, question=question, judge=judge) if answer else 0.0
    fmt = score_format(completion)

    kind = reward_key.get("kind")
    if tool_calls < 1:
        tool_bonus = 0.0
    elif kind == "yesno":
        tool_bonus = tool_bonus_coef * outcome
    elif kind == "set":
        tool_bonus = tool_bonus_coef * (grounding_iou or 0.0)
    else:
        tool_bonus = tool_bonus_coef  # text kind: unchanged flat behavior (spec section 5 item 6)

    reward = outcome_weight * outcome + format_weight * fmt + tool_bonus
    return {"reward": reward, "outcome": outcome, "format": fmt, "tool_bonus": tool_bonus}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/data_core && python -m pytest tests/test_reward_total.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones — check none of them assumed the old flat-always behavior; fix any that did to match the new per-kind rule, same way Task 1/Step 6 did for `test_score_set`)

- [ ] **Step 5: Write the failing test for `verl_bridge/reward.py`'s grounding wiring**

```python
# add to packages/verl_bridge/tests/test_reward.py
from verl_bridge.reward import compute_score, _compute_grounding_iou


def test_compute_grounding_iou_uses_extra_info(monkeypatch):
    # Stub out the h5/frame_dims lookups so this test needs no real data files.
    def fake_boxes(dicom_uuid, detr_class_id):
        return [(100.0, 100.0, 200.0, 200.0)]  # pixel xyxy

    def fake_dims(dicom_uuid):
        return (400, 400)

    monkeypatch.setattr("verl_bridge.reward._detr_boxes_for", fake_boxes)
    monkeypatch.setattr("verl_bridge.reward._frame_dims_for", fake_dims)

    sol = ('<think>t</think>\n'
           '<tool_call>{"name": "zoom", "arguments": {"view": "A4C", '
           '"bbox": [0.25, 0.25, 0.5, 0.5], "frame_indices": [0]}}</tool_call>\n'
           '<answer>- left ventricular enlargement</answer>')
    iou = _compute_grounding_iou(
        sol, finding_names=["left ventricular enlargement"],
        dicom_uuids_by_view={"A4C": "dicom-1"})
    assert iou == 1.0  # (100,100,200,200) in a 400x400 frame == (0.25,0.25,0.5,0.5) exactly


def test_compute_score_set_kind_applies_grounding(monkeypatch):
    def fake_boxes(dicom_uuid, detr_class_id):
        return [(100.0, 100.0, 200.0, 200.0)]

    def fake_dims(dicom_uuid):
        return (400, 400)

    monkeypatch.setattr("verl_bridge.reward._detr_boxes_for", fake_boxes)
    monkeypatch.setattr("verl_bridge.reward._frame_dims_for", fake_dims)

    rk = {"kind": "set", "target": ["left ventricular enlargement"], "gold": {}}
    sol = ('<think>t</think>\n'
           '<tool_call>{"name": "zoom", "arguments": {"view": "A4C", '
           '"bbox": [0.25, 0.25, 0.5, 0.5], "frame_indices": [0]}}</tool_call>\n'
           '<answer>- left ventricular enlargement</answer>')
    r = compute_score("echoprime_grpo", sol, rk,
                       extra_info={"tool_bonus_coef": 0.5,
                                   "dicom_uuids_by_view": {"A4C": "dicom-1"}})
    # outcome(1.0) + format(1.0*0.2) + tool_bonus(0.5*1.0) = 1.7
    assert abs(r - 1.7) < 1e-9
```

- [ ] **Step 6: Run to verify it fails**

Run: `cd packages/verl_bridge && python -m pytest tests/test_reward.py -v -k grounding`
Expected: FAIL — `_compute_grounding_iou`, `_detr_boxes_for`, `_frame_dims_for` don't exist yet.

- [ ] **Step 7: Write the h5/frame_dims-backed helpers and wire them into `compute_score`**

```python
# packages/verl_bridge/reward.py additions
import os
import h5py
from data_core.reward.findings import FINDING_TO_DETR_CLASS
from data_core.reward.grounding import ground_tool_calls, normalize_detr_box
from tool_env.parse import parse_action

_DETR_H5_PATH = os.environ.get(
    "ECHO_DETR_H5", "/vast/users/mohammad.yaqub/report_generation/data/train_detections.h5")
_FRAME_DIMS_PATH = os.environ.get("ECHO_FRAME_DIMS", "build/frame_dims_train.json")

_detr_h5 = None
_frame_dims = None


def _load_detr_h5():
    global _detr_h5
    if _detr_h5 is None:
        _detr_h5 = h5py.File(_DETR_H5_PATH, "r")
    return _detr_h5


def _load_frame_dims():
    global _frame_dims
    if _frame_dims is None:
        import json
        with open(_FRAME_DIMS_PATH) as fh:
            _frame_dims = json.load(fh)
    return _frame_dims


def _detr_boxes_for(dicom_uuid: str, detr_class_id: int) -> list:
    """Pixel-space boxes_xyxy for one dicom/class, across all 16 frames, deduped by rounding
    (multiple frames often detect the same structure at nearly the same location -- keeping
    every one is wasted work in ground_tool_calls's O(calls * boxes) loop, not a correctness
    issue, so dedup is a plain performance choice, not required for correctness)."""
    h5 = _load_detr_h5()
    if dicom_uuid not in h5:
        return []
    grp = h5[dicom_uuid]
    boxes = []
    for frame_idx in range(16):
        key = f"frame_{frame_idx}"
        if key not in grp:
            continue
        classes = grp[key]["classes"][:]
        xyxy = grp[key]["boxes_xyxy"][:]
        for cls_id, box in zip(classes, xyxy):
            if int(cls_id) == detr_class_id:
                boxes.append(tuple(float(v) for v in box))
    return boxes


def _frame_dims_for(dicom_uuid: str):
    dims = _load_frame_dims()
    if dicom_uuid not in dims:
        return None
    w, h = dims[dicom_uuid]
    return (w, h)


def _compute_grounding_iou(solution_str: str, finding_names: list,
                            dicom_uuids_by_view: dict) -> float:
    """Best grounding IoU across every (finding, tool call) pair -- mirrors ground_tool_calls's
    own "best, not average" reasoning (docstring there). Returns 0.0 if nothing groundable
    (no mapped finding, no tool calls, no matching DETR detections)."""
    if not finding_names or not dicom_uuids_by_view:
        return 0.0
    parsed = parse_action(solution_str)
    best = 0.0
    for finding in finding_names:
        detr_class_id = FINDING_TO_DETR_CLASS.get(finding)
        if detr_class_id is None:
            continue
        for view, dicom_uuid in dicom_uuids_by_view.items():
            dims = _frame_dims_for(dicom_uuid)
            if dims is None:
                continue
            w, h = dims
            pixel_boxes = _detr_boxes_for(dicom_uuid, detr_class_id)
            if not pixel_boxes:
                continue
            norm_boxes = [normalize_detr_box(b, w, h) for b in pixel_boxes]
            view_calls = [c for c in parsed.calls if c.get("arguments", {}).get("view") == view]
            best = max(best, ground_tool_calls(view_calls, view, norm_boxes))
    return best
```

Update `compute_score` to compute `grounding_iou` for `set`-kind rows and pass it through:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    reward_key = _parse_reward_key(ground_truth)
    info = extra_info or {}
    n_tool_calls = _count_tool_calls(solution_str)

    grounding_iou = None
    if reward_key.get("kind") == "set":
        grounding_iou = _compute_grounding_iou(
            solution_str,
            finding_names=reward_key.get("target") or [],
            dicom_uuids_by_view=info.get("dicom_uuids_by_view") or {})

    result = total_reward(
        reward_key,
        solution_str or "",
        tool_calls=n_tool_calls,
        tool_bonus_coef=float(info.get("tool_bonus_coef", 0.0)),
        grounding_iou=grounding_iou,
    )
    # (unchanged episode-counter/logging block below)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd packages/verl_bridge && python -m pytest tests/test_reward.py -v`
Expected: PASS (all tests, including pre-existing ones — the old `test_compute_score_applies_annealed_bonus` test uses `kind: "yesno"`, which now goes through the `tool_bonus_coef * outcome` path instead of always-flat; recompute its expected value: `rk = {"kind": "yesno", "target": "yes", ...}`, `sol` answers "Yes." correctly, so `outcome=1.0`, `tool_bonus = 0.1 * 1.0 = 0.1` — same numeric result as before by coincidence (the test's `target`/answer already agree), so it should still pass unchanged; verify this rather than assume).

- [ ] **Step 9: Set `ECHO_DETR_H5`/`ECHO_FRAME_DIMS` env vars in the training launch script**

Find wherever `run_grpo_echoprime_amd.sbatch` (or its successor once Task 9 of the other plan updates config) sets other `ECHO_*` env vars (e.g. `ECHO_PREPROCESSED_DIR`) and add:

```bash
export ECHO_DETR_H5=/vast/users/mohammad.yaqub/report_generation/data/train_detections.h5
export ECHO_FRAME_DIMS=build/frame_dims_train.json
```

For the val/eval run, these should point at the test-split files instead — check whether this project's val launch already uses a separate env-var set or the same sbatch with an override, and match that pattern.

- [ ] **Step 10: Commit**

```bash
git add packages/data_core/reward/score.py packages/verl_bridge/reward.py \
        packages/data_core/tests/test_reward_total.py packages/verl_bridge/tests/test_reward.py
git commit -m "$(cat <<'EOF'
per-kind tool_bonus: DeepEyes-style gate for yesno, DETR IoU grounding for set

yesno: tool_bonus = coef * outcome (only rewarded alongside a correct answer).
set (abnormality_list): tool_bonus = coef * best grounding IoU between the
model's zoom/select_frames tool calls and the DETR-detected box for each
finding's structure. text: unchanged flat bonus (its outcome scoring is
still the unreliable entity_F1 path, not a fair gate to condition on yet).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Per-question-type reward normalization

**Files:**
- Create: `packages/data_core/reward/normalize.py`
- Test: `packages/data_core/tests/test_normalize.py`
- Investigation only (no file changes committed as code) into where this plugs into verl's batch reward pipeline

**Interfaces:**
- Produces: `normalize.py::normalize_by_type(rewards: list[float], types: list[str]) -> list[float]` — z-scores each reward within its own `types` group (same-length lists, paired by index); a group of size 1 (or all-identical rewards) returns 0.0 for that group rather than dividing by a zero stddev.

- [ ] **Step 1: Write the failing tests for the pure normalization function**

```python
# packages/data_core/tests/test_normalize.py
from data_core.reward.normalize import normalize_by_type


def test_normalize_by_type_z_scores_within_group():
    rewards = [1.0, 2.0, 3.0, 10.0, 20.0]
    types = ["a", "a", "a", "b", "b"]
    out = normalize_by_type(rewards, types)
    # group "a": mean=2, std=sqrt(2/3); group "b": mean=15, std=5
    import math
    std_a = math.sqrt(((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3)
    assert abs(out[0] - (1 - 2) / std_a) < 1e-9
    assert abs(out[3] - (10 - 15) / 5) < 1e-9


def test_normalize_by_type_zero_variance_group_returns_zero():
    rewards = [5.0, 5.0, 5.0]
    types = ["a", "a", "a"]
    assert normalize_by_type(rewards, types) == [0.0, 0.0, 0.0]


def test_normalize_by_type_single_item_group_returns_zero():
    rewards = [1.0, 2.0, 3.0]
    types = ["a", "b", "c"]
    assert normalize_by_type(rewards, types) == [0.0, 0.0, 0.0]


def test_normalize_by_type_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize_by_type([1.0, 2.0], ["a"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/data_core && python -m pytest tests/test_normalize.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write the implementation**

```python
# packages/data_core/reward/normalize.py
"""Per-question-type reward normalization -- targets the exact symptom already measured in
the grpo-2b-video-0910-1319 run (SPEC.md): pooled reward climbed to 0.957 while
structure_description reward rose and classification/conclusion reward fell, because one
gameable channel's gains swamped the pooled number. Z-scoring within each question type before
GRPO batch pooling stops that (spec: docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-
design.md section 5, "Also:" paragraph).
"""
from collections import defaultdict


def normalize_by_type(rewards: list, types: list) -> list:
    if len(rewards) != len(types):
        raise ValueError(f"rewards ({len(rewards)}) and types ({len(types)}) length mismatch")

    by_type = defaultdict(list)
    for i, t in enumerate(types):
        by_type[t].append(i)

    stats = {}
    for t, idxs in by_type.items():
        vals = [rewards[i] for i in idxs]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var ** 0.5
        stats[t] = (mean, std)

    out = [0.0] * len(rewards)
    for i, (r, t) in enumerate(zip(rewards, types)):
        mean, std = stats[t]
        out[i] = (r - mean) / std if std > 0 else 0.0
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/data_core && python -m pytest tests/test_normalize.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit the pure function**

```bash
git add packages/data_core/reward/normalize.py packages/data_core/tests/test_normalize.py
git commit -m "$(cat <<'EOF'
add per-question-type reward z-score normalization

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Investigate the real integration point in verl's batch reward pipeline before wiring this in**

`normalize_by_type` needs a batch of (reward, question_type) pairs, not one example at a time like `compute_score` gets — this is a genuinely different point in verl's pipeline than anything touched so far in either plan. Before writing integration code, read `external/verl/verl/workers/reward_manager/` (whichever `RewardManager` subclass `echoprime_grpo.yaml`'s `custom_reward_function` config actually routes through — check `algorithm.adv_estimator: grpo`'s own per-prompt-group normalization first, since GRPO already normalizes advantages within a prompt group; per-TYPE normalization across different prompts is a different axis and may need a custom `reward_manager` class, not just a `compute_score` change) to find where per-batch reward tensors are assembled, and confirm: (a) whether `extra_info`/`question_type` is available at that point, (b) whether overriding the reward manager is configurable via yaml (`reward_model.reward_manager: <name>`) the way `custom_reward_function` already is. Write findings as a short note (not a plan step) and treat wiring this in as a follow-up task once the integration point is confirmed — don't guess at verl internals here the way this plan's other tasks avoided guessing at vLLM internals (see the sibling plan's Task 4/Step 1 and Task 8/Step 5 for the same pattern).

---

## Self-Review Notes

- **Spec coverage:** §5 items 1 (yesno unchanged) → no task needed, already true. Item 2 (set outcome, canonical taxonomy) → Task 1. Item "set" tool_bonus (DETR grounding) → Tasks 2-4. Item 3 (format unchanged) → no task, already true. Item 4 (no length penalty) → no task, deliberately absent. Item 5 (no NLI) → no task, deliberately absent. Item 6 (text unchanged) → Task 4/Step 3 explicitly preserves this. "Also:" (per-type normalization) → Task 5.
- **Placeholder scan:** Task 5/Step 6 is an investigation step, not a placeholder — it has a concrete deliverable (a findings note) and concrete files to read, same pattern the sibling plan uses for its own genuine unknowns (vLLM's multi-modal merge API, vLLM embedding-tensor continuation). Not deferring the actual wiring further than that one open question warrants.
- **Type consistency check:** `grounding_iou` flows from `verl_bridge/reward.py::_compute_grounding_iou` → `compute_score`'s `total_reward(..., grounding_iou=...)` call → `data_core/reward/score.py::total_reward`'s new parameter — same name throughout, confirmed. `FINDING_TO_DETR_CLASS`/`ground_tool_calls`/`normalize_detr_box` names match between Tasks 2, 3, and 4's usage.
