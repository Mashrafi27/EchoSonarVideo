# Phase 2 — Echo Agent Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline, unit-tested **Echo Agent Environment** — a disk-backed frame server exposing the three echo tools (`select_view`, `select_frames`, `zoom`) plus an initial-observation builder and budget guardrails — as a standalone package (`echo_env/`) with zero dependency on the DeepEyes runtime.

**Architecture:** `echo_env/` is a new package **sibling to `echo_rl/`**. It reuses `echo_rl.data` (study indexing, frame selection, view normalization) for all pure logic and adds only what the runtime needs: PIL-based frame loading/cropping, bbox math, per-episode budget, and a framework-agnostic `EchoEnv` orchestrator whose `step(action_string) -> (Observation, reward, done, info)` mirrors DeepEyes' `ToolBase.execute` contract at the **domain level** (returns a rich `Observation`, not a token dict). A single pure function `to_deepeyes_obs()` is the only seam to DeepEyes; the live `ToolBase` subclass, import-registration, and the video-plumbing patch are **explicitly deferred to Phase 3** (they require the DeepEyes/torch runtime to be meaningful). PIL is isolated behind a `FrameLoader` so ~90% of tests run without touching pixels.

**Tech Stack:** Python 3, `Pillow` (new dep — the only addition over Phase 1's stdlib-only `echo_rl`), `pytest`. Reuses `echo_rl` in-repo.

## Global Constraints

Copied verbatim from the spec (`docs/superpowers/specs/2026-07-23-echo-deepeyes-rl-training-design.md`). Every task's requirements implicitly include these.

- **Three free-form, model-driven tools** (§4), taught by outcome reward — no segmentation, no phase labels, no structure menu. Tools must accept arbitrary `view_name`/indices/bbox and fail gracefully, never assume valid input.
  1. `select_view(view_name)` → sparse temporal preview: `n_preview_frames` evenly-spaced low-res frames spanning one cardiac cycle.
  2. `select_frames(view_name, [i, j, k, …])` → those specific whole frames at higher (native) resolution.
  3. `zoom(view_name, bbox, frame_indices=[…])` → one bbox applied across a model-chosen set of frames (spatiotemporal crop).
- **Model → tool call format is XML-wrapped JSON**: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (one or more per turn); `<answer>...</answer>` terminates the episode. This matches DeepEyes' dominant/latest convention (`visual_toolbox_v5.py`). Do **not** use ```` ```json ```` fences.
- **Native frame size is 336×336** (verified 2026-07-24, uniform sample). "High-res" = native 336; resolution tiers exist only to *down*scale previews/thumbnails to save context. bbox coordinates are in native-pixel space `[x1, y1, x2, y2]`.
- **bbox guardrails** (port from `visual_toolbox_v5.maybe_resize_bbox`): clamp to `[0,0,w,h]`; require `left < right and top < bottom`; enforce `max(h,w)/min(h,w) <= 100` aspect ratio; enforce min side `28` px (Qwen2.5-VL patch floor), expanding around the bbox center if smaller. Invalid → return `None` (handled as a graceful tool error, never a crash).
- **Study-level identity**: the env operates on one `study_uuid` at a time; frames come from `<preprocessed_dir>/<study_uuid>/di-*_<View>/N.png` (via `echo_rl.data.studies.study_dir` + `index_study`).
- **DRY**: reuse `echo_rl.data.studies` (`Clip`, `index_study`, `study_dir`), `echo_rl.data.frames` (`evenly_spaced`, `midframe`), `echo_rl.data.views` (`canonical_view`, `base_view`, `parse_clip_dirname`). Do **not** re-implement study indexing, numeric frame sorting, or view parsing.
- **Determinism**: any sampling (frame selection) is deterministic given the study + config; no `random` without a seeded generator. `EnvConfig.seed` defaults to 0.
- **Keep `echo_rl` stdlib-only.** PIL enters only in `echo_env`. `echo_rl` must not gain a Pillow import.
- **The env's procedural `step` reward is `0.0`** for both valid and invalid tool calls (matches `visual_toolbox_v5`). Format / tool-use / outcome rewards live in the Phase-3 reward scorer, **not** here.
- **Phase-2 scope boundary:** everything in `echo_env/` is offline and DeepEyes-runtime-free. The `ToolBase` adapter, `verl` import-registration, and the two-file video patch (`parallel_env.py` + `rl_dataset.py`) are **Phase 3** and appear here only as the `INTEGRATION.md` contract (Task 11).

---

## File Structure

All paths relative to repo root `/vast/users/mohammad.yaqub/project/EchoSonarVideo`.

- `echo_env/__init__.py` — package marker, exports `EchoEnv`, `EnvConfig`, `Observation`.
- `echo_env/config.py` — `EnvConfig` dataclass (resolution tiers, frame counts, budget caps, seed).
- `echo_env/observation.py` — `FrameImg`, `Observation` dataclasses (domain-level tool output).
- `echo_env/bbox.py` — pure bbox clamp/validate/resize (no PIL).
- `echo_env/frames.py` — `FrameLoader` protocol + `PILFrameLoader` (load/downscale/crop); PIL isolated here.
- `echo_env/manifest.py` — `ViewEntry`, `StudyManifest`, `build_manifest` (reuses `index_study`); view-name resolution.
- `echo_env/tools.py` — `select_view`, `select_frames`, `zoom` handlers → `Observation`.
- `echo_env/budget.py` — `Budget` per-episode guardrail tracker.
- `echo_env/parse.py` — `parse_action` (`<tool_call>` + `<answer>` extraction), pure.
- `echo_env/env.py` — `EchoEnv` orchestrator: `reset(study_uuid)`, `step(action_string)`.
- `echo_env/packaging.py` — `to_deepeyes_obs(observation, user_prompt)` (the pure DeepEyes seam).
- `echo_env/INTEGRATION.md` — Phase-3 wiring contract (adapter, registration, video patch).
- `echo_env/tests/conftest.py` — fixture study-dir generator (tiny real PNGs via PIL).
- `echo_env/tests/test_*.py` — one per module.

---

## Task 1: Package scaffold, `EnvConfig`, Pillow dep, fixture generator

**Files:**
- Create: `echo_env/__init__.py`, `echo_env/config.py`, `echo_env/tests/__init__.py`, `echo_env/tests/conftest.py`
- Test: `echo_env/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (foundation).
- Produces:
  - `echo_env.config.EnvConfig` dataclass with fields (all keyword-defaulted):
    `preprocessed_dir: str = ""`, `n_preview_frames: int = 5`, `n_highres_frames: int = 8`,
    `preview_max_side: int = 168`, `highres_max_side: int = 336`, `n_overview_views: int = 19`,
    `min_crop_side: int = 28`, `max_aspect: float = 100.0`, `max_tool_calls: int = 8`,
    `max_frames_per_obs: int = 8`, `max_total_frames: int = 32`, `seed: int = 0`.
    Classmethod `EnvConfig.from_env() -> EnvConfig` reading `ECHO_PREPROCESSED_DIR` (default the Phase-1 value `os.path.join(_PARENT, "preprocessed_data")` where `_PARENT` is two levels up from repo root, matching `echo_rl.config`) for `preprocessed_dir`; all other fields from `int(os.environ.get("ECHO_<NAME>", default))`.
  - `conftest.py` fixture `study_fixture(tmp_path) -> tuple[str, str]` returning `(preprocessed_dir, study_uuid)` — a real on-disk study dir with ≥3 clips of known view/frame-count, each frame a distinct 336×336 PNG (written via PIL) whose pixel content encodes its frame index (so crop/selection tests can assert *which* frame came back). Also a `make_png(path, w, h, marker)` helper.

- [ ] **Step 1: Install Pillow into the Phase-1 venv**

Run: `.venv/bin/pip install Pillow`
Expected: `Successfully installed Pillow-...`. Then verify: `.venv/bin/python -c "from PIL import Image; print(Image.__version__)"` prints a version.

- [ ] **Step 2: Write the failing config test**

Create `echo_env/tests/__init__.py` (empty) and `echo_env/tests/test_config.py`:

```python
import os
from echo_env.config import EnvConfig


def test_defaults():
    c = EnvConfig()
    assert c.n_preview_frames == 5
    assert c.highres_max_side == 336
    assert c.max_tool_calls == 8
    assert c.max_total_frames == 32
    assert c.seed == 0


def test_from_env_reads_preprocessed_dir(monkeypatch):
    monkeypatch.setenv("ECHO_PREPROCESSED_DIR", "/some/where")
    monkeypatch.setenv("ECHO_MAX_TOOL_CALLS", "3")
    c = EnvConfig.from_env()
    assert c.preprocessed_dir == "/some/where"
    assert c.max_tool_calls == 3


def test_from_env_default_preprocessed_dir_endswith(monkeypatch):
    monkeypatch.delenv("ECHO_PREPROCESSED_DIR", raising=False)
    c = EnvConfig.from_env()
    assert c.preprocessed_dir.endswith("preprocessed_data")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest echo_env/tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env'`.

- [ ] **Step 4: Implement `echo_env/__init__.py` and `echo_env/config.py`**

`echo_env/__init__.py`:

```python
from echo_env.config import EnvConfig

__all__ = ["EnvConfig"]
```

`echo_env/config.py`:

```python
import os
from dataclasses import dataclass

# repo root is two levels below the shared data parent, mirroring echo_rl.config
_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class EnvConfig:
    preprocessed_dir: str = ""
    n_preview_frames: int = 5
    n_highres_frames: int = 8
    preview_max_side: int = 168
    highres_max_side: int = 336
    n_overview_views: int = 19
    min_crop_side: int = 28
    max_aspect: float = 100.0
    max_tool_calls: int = 8
    max_frames_per_obs: int = 8
    max_total_frames: int = 32
    seed: int = 0

    @classmethod
    def from_env(cls) -> "EnvConfig":
        def _i(name, default):
            return int(os.environ.get(name, default))
        return cls(
            preprocessed_dir=os.environ.get(
                "ECHO_PREPROCESSED_DIR", os.path.join(_PARENT, "preprocessed_data")),
            n_preview_frames=_i("ECHO_N_PREVIEW_FRAMES", 5),
            n_highres_frames=_i("ECHO_N_HIGHRES_FRAMES", 8),
            preview_max_side=_i("ECHO_PREVIEW_MAX_SIDE", 168),
            highres_max_side=_i("ECHO_HIGHRES_MAX_SIDE", 336),
            n_overview_views=_i("ECHO_N_OVERVIEW_VIEWS", 19),
            min_crop_side=_i("ECHO_MIN_CROP_SIDE", 28),
            max_aspect=float(os.environ.get("ECHO_MAX_ASPECT", 100.0)),
            max_tool_calls=_i("ECHO_MAX_TOOL_CALLS", 8),
            max_frames_per_obs=_i("ECHO_MAX_FRAMES_PER_OBS", 8),
            max_total_frames=_i("ECHO_MAX_TOTAL_FRAMES", 32),
            seed=_i("ECHO_SEED", 0),
        )
```

- [ ] **Step 5: Write the fixture generator `conftest.py`**

`echo_env/tests/conftest.py`:

```python
import os
import pytest
from PIL import Image


def make_png(path: str, w: int = 336, h: int = 336, marker: int = 0) -> None:
    """Write a solid-color PNG whose R channel encodes `marker` (frame index),
    so tests can read back which frame was returned."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (w, h), (marker % 256, 0, 0))
    img.save(path)


# (clip_dirname, n_frames)
_FIXTURE_CLIPS = [
    ("di-AAAA-0001_A4C", 10),
    ("di-BBBB-0002_A4C Zoomed Mitral", 6),
    ("di-CCCC-0003_PLAX", 8),
    ("di-DDDD-0004_PSAX Apex", 4),
]


@pytest.fixture
def study_fixture(tmp_path):
    """Create a real on-disk study and return (preprocessed_dir, study_uuid)."""
    preprocessed_dir = str(tmp_path / "preprocessed_data")
    study_uuid = "st-TEST-0000-0000"
    study_dir = os.path.join(preprocessed_dir, study_uuid)
    for clip_name, n in _FIXTURE_CLIPS:
        for i in range(n):
            make_png(os.path.join(study_dir, clip_name, f"{i}.png"), marker=i)
    return preprocessed_dir, study_uuid
```

- [ ] **Step 6: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_config.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add echo_env/__init__.py echo_env/config.py echo_env/tests/__init__.py echo_env/tests/conftest.py echo_env/tests/test_config.py
git commit -m "feat(env): echo_env scaffold, EnvConfig, PIL fixture generator"
```

---

## Task 2: `Observation` / `FrameImg` dataclasses

**Files:**
- Create: `echo_env/observation.py`
- Test: `echo_env/tests/test_observation.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `echo_env.observation.FrameImg` dataclass: `image` (a PIL `Image` at runtime; typed `object` to avoid a hard PIL import in this module), `view_name: str`, `frame_index: int`, `kind: str` (`"thumbnail"|"preview"|"highres"|"crop"`), `bbox: tuple | None = None`.
  - `echo_env.observation.Observation` dataclass: `tool: str`, `frames: list` (of `FrameImg`), `text: str`, `ok: bool = True`, `error: str | None = None`. Property `.n_frames -> int`. Classmethod `Observation.failure(tool, error) -> Observation` → `ok=False, frames=[], text=error`.

- [ ] **Step 1: Write the failing test**

`echo_env/tests/test_observation.py`:

```python
from echo_env.observation import FrameImg, Observation


def test_frameimg_fields():
    f = FrameImg(image=object(), view_name="A4C", frame_index=3, kind="highres")
    assert f.frame_index == 3
    assert f.bbox is None


def test_observation_counts():
    obs = Observation(tool="select_view", frames=[
        FrameImg(image=object(), view_name="A4C", frame_index=0, kind="preview"),
        FrameImg(image=object(), view_name="A4C", frame_index=5, kind="preview"),
    ], text="ok")
    assert obs.n_frames == 2
    assert obs.ok is True


def test_observation_failure():
    obs = Observation.failure("zoom", "bad bbox")
    assert obs.ok is False
    assert obs.n_frames == 0
    assert obs.error == "bad bbox"
    assert obs.text == "bad bbox"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_env/tests/test_observation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.observation'`.

- [ ] **Step 3: Implement `echo_env/observation.py`**

```python
from dataclasses import dataclass, field


@dataclass
class FrameImg:
    image: object            # PIL.Image.Image at runtime
    view_name: str
    frame_index: int
    kind: str                # "thumbnail" | "preview" | "highres" | "crop"
    bbox: tuple | None = None


@dataclass
class Observation:
    tool: str
    frames: list = field(default_factory=list)
    text: str = ""
    ok: bool = True
    error: str | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @classmethod
    def failure(cls, tool: str, error: str) -> "Observation":
        return cls(tool=tool, frames=[], text=error, ok=False, error=error)
```

- [ ] **Step 4: Run test to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_observation.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/observation.py echo_env/tests/test_observation.py
git commit -m "feat(env): Observation/FrameImg domain dataclasses"
```

---

## Task 3: `bbox.py` — pure clamp / validate / resize

**Files:**
- Create: `echo_env/bbox.py`
- Test: `echo_env/tests/test_bbox.py`

**Interfaces:**
- Consumes: nothing (pure math).
- Produces:
  - `echo_env.bbox.normalize_bbox(bbox, width, height, min_side=28, max_aspect=100.0) -> tuple[int,int,int,int] | None`.
    `bbox` is any 4-sequence `[left, top, right, bottom]` in native pixels. Returns a clamped, min-side-expanded, aspect-validated integer bbox, or `None` if unrecoverably invalid. Ports `visual_toolbox_v5.maybe_resize_bbox` + `validate_bbox` (center-expand when a side `< min_side`).

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_bbox.py`:

```python
from echo_env.bbox import normalize_bbox


def test_valid_bbox_passthrough():
    assert normalize_bbox([10, 20, 100, 200], 336, 336) == (10, 20, 100, 200)


def test_clamp_to_bounds():
    assert normalize_bbox([-5, -5, 500, 500], 336, 336) == (0, 0, 336, 336)


def test_inverted_bbox_rejected():
    assert normalize_bbox([100, 100, 50, 50], 336, 336) is None


def test_tiny_bbox_expanded_to_min_side():
    out = normalize_bbox([160, 160, 170, 170], 336, 336, min_side=28)
    assert out is not None
    left, top, right, bottom = out
    assert (right - left) >= 28 and (bottom - top) >= 28


def test_extreme_aspect_ratio_rejected():
    # 300 wide x 1 tall -> ratio 300 > 100; after min-side expansion still invalid? 
    # a 1px-tall strip expands vertically to >=28, ratio ~ 300/28 ~ 10.7 -> becomes valid.
    # Use a case that stays extreme: width 300, height 2 near an edge cannot expand -> None.
    out = normalize_bbox([0, 0, 300, 2], 300, 2, min_side=28, max_aspect=100.0)
    assert out is None


def test_non_numeric_bbox_rejected():
    assert normalize_bbox(["a", "b", "c", "d"], 336, 336) is None
    assert normalize_bbox([1, 2, 3], 336, 336) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_bbox.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.bbox'`.

- [ ] **Step 3: Implement `echo_env/bbox.py`**

```python
from math import ceil, floor


def _validate(left, top, right, bottom, max_aspect) -> bool:
    if not (left < right and top < bottom):
        return False
    h = bottom - top
    w = right - left
    if max(h, w) / min(h, w) > max_aspect:
        return False
    return True


def normalize_bbox(bbox, width, height, min_side=28, max_aspect=100.0):
    """Clamp to [0,0,width,height], expand sub-min_side sides around the center,
    validate ordering + aspect ratio. Return an int tuple or None."""
    if not hasattr(bbox, "__len__") or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    if not _validate(left, top, right, bottom, max_aspect):
        return None
    h = bottom - top
    w = right - left
    if h < min_side or w < min_side:
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        ratio = min_side / min(h, w)
        half_w = ceil(w * ratio * 0.5)
        half_h = ceil(h * ratio * 0.5)
        left = max(0, floor(cx - half_w))
        right = min(width, ceil(cx + half_w))
        top = max(0, floor(cy - half_h))
        bottom = min(height, ceil(cy + half_h))
        if not _validate(left, top, right, bottom, max_aspect):
            return None
    return (int(left), int(top), int(right), int(bottom))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_bbox.py -q`
Expected: PASS (6 passed). If `test_extreme_aspect_ratio_rejected` surprises you, trace the math — a 300×2 box on a 300×2 image cannot expand vertically (already at the edge) so it stays ratio 150 > 100 → `None`, which is the assertion.

- [ ] **Step 5: Commit**

```bash
git add echo_env/bbox.py echo_env/tests/test_bbox.py
git commit -m "feat(env): pure bbox normalize/clamp/validate"
```

---

## Task 4: `frames.py` — `FrameLoader` + `PILFrameLoader`

**Files:**
- Create: `echo_env/frames.py`
- Test: `echo_env/tests/test_frames.py`

**Interfaces:**
- Consumes: `echo_env.bbox.normalize_bbox` (indirectly, via callers — not here), the `study_fixture` conftest fixture.
- Produces:
  - `echo_env.frames.FrameLoader` — a `typing.Protocol` with `load(path: str) -> object`, `downscale(image, max_side: int) -> object`, `crop(image, bbox) -> object`, `size(image) -> tuple[int,int]`.
  - `echo_env.frames.PILFrameLoader` — concrete impl. `load` opens + `convert("RGB")`; `downscale` returns the image unchanged if `max(w,h) <= max_side`, else `image.resize((w*s, h*s))` preserving aspect where `s = max_side / max(w,h)`; `crop` = `image.crop(tuple(bbox))`; `size` = `image.size`.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_frames.py`:

```python
import os
from echo_env.frames import PILFrameLoader


def test_load_and_size(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "3.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    assert loader.size(img) == (336, 336)
    # marker: R channel encodes the frame index (3)
    assert img.getpixel((0, 0))[0] == 3


def test_downscale_shrinks_large_only(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "0.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    small = loader.downscale(img, 168)
    assert loader.size(small) == (168, 168)
    # already-small stays put
    same = loader.downscale(small, 336)
    assert loader.size(same) == (168, 168)


def test_crop(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "0.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    c = loader.crop(img, (10, 10, 100, 120))
    assert loader.size(c) == (90, 110)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.frames'`.

- [ ] **Step 3: Implement `echo_env/frames.py`**

```python
from typing import Protocol
from PIL import Image


class FrameLoader(Protocol):
    def load(self, path: str) -> object: ...
    def downscale(self, image: object, max_side: int) -> object: ...
    def crop(self, image: object, bbox) -> object: ...
    def size(self, image: object) -> tuple: ...


class PILFrameLoader:
    def load(self, path: str):
        with Image.open(path) as im:
            return im.convert("RGB")

    def size(self, image):
        return image.size

    def downscale(self, image, max_side: int):
        w, h = image.size
        longest = max(w, h)
        if longest <= max_side:
            return image
        s = max_side / float(longest)
        return image.resize((max(1, round(w * s)), max(1, round(h * s))))

    def crop(self, image, bbox):
        return image.crop(tuple(bbox))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_frames.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/frames.py echo_env/tests/test_frames.py
git commit -m "feat(env): FrameLoader protocol + PILFrameLoader"
```

---

## Task 5: `manifest.py` — study manifest + view resolution

**Files:**
- Create: `echo_env/manifest.py`
- Test: `echo_env/tests/test_manifest.py`

**Interfaces:**
- Consumes: `echo_rl.data.studies.index_study`, `echo_rl.data.studies.study_dir`, `echo_rl.data.views.canonical_view`, `echo_rl.data.views.base_view`. (Reuse — do not re-index.)
- Produces:
  - `echo_env.manifest.ViewEntry` dataclass: `view_name: str` (the model-facing name = the clip's view portion, e.g. `"A4C Zoomed Mitral"`), `clip` (an `echo_rl.data.studies.Clip`), and property `frame_count -> int` (delegates to `clip.frame_count`).
  - `echo_env.manifest.StudyManifest`: constructed with `views: list[ViewEntry]`. Methods:
    - `resolve(view_name: str) -> ViewEntry | None` — case-insensitive exact match first; then match on `canonical_view`; then `base_view` fallback (a query of `"A4C"` resolves the plain `"A4C"` clip if present, else the first clip whose base view is `"A4C"` by sorted `view_name`); `None` if no match.
    - `overview(limit: int) -> list[ViewEntry]` — first `limit` views by sorted `view_name` (deterministic).
    - `view_names() -> list[str]`.
  - `echo_env.manifest.build_manifest(preprocessed_dir: str, study_uuid: str) -> StudyManifest` — uses `study_dir` + `index_study`; maps each `Clip.view` to a `ViewEntry`. Propagates `FileNotFoundError` from `index_study` for a missing study (caller handles).

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_manifest.py`:

```python
from echo_env.manifest import build_manifest, StudyManifest, ViewEntry


def test_build_manifest_lists_views(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    names = m.view_names()
    assert "A4C" in names
    assert "A4C Zoomed Mitral" in names
    assert "PLAX" in names


def test_resolve_exact_case_insensitive(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("a4c").view_name == "A4C"
    assert m.resolve("A4C Zoomed Mitral").view_name == "A4C Zoomed Mitral"


def test_resolve_base_fallback_prefers_plain(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    # "A4C" plain clip exists, so a base query returns it, not the zoomed variant
    assert m.resolve("A4C").view_name == "A4C"


def test_resolve_unknown_returns_none(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("A2C") is None


def test_frame_count(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("A4C").frame_count == 10


def test_overview_deterministic_limit(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    ov = m.overview(limit=2)
    assert len(ov) == 2
    assert [v.view_name for v in ov] == sorted(m.view_names())[:2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.manifest'`.

- [ ] **Step 3: Implement `echo_env/manifest.py`**

```python
from dataclasses import dataclass
from echo_rl.data.studies import index_study, study_dir
from echo_rl.data.views import canonical_view, base_view


@dataclass
class ViewEntry:
    view_name: str
    clip: object  # echo_rl.data.studies.Clip

    @property
    def frame_count(self) -> int:
        return self.clip.frame_count


class StudyManifest:
    def __init__(self, views):
        # keep a stable, sorted order
        self.views = sorted(views, key=lambda v: v.view_name)
        self._by_lower = {v.view_name.lower(): v for v in self.views}

    def view_names(self):
        return [v.view_name for v in self.views]

    def overview(self, limit: int):
        return self.views[:limit]

    def resolve(self, view_name: str):
        if not view_name:
            return None
        q = view_name.strip()
        # 1. exact case-insensitive
        hit = self._by_lower.get(q.lower())
        if hit:
            return hit
        # 2. canonical_view match
        qc = canonical_view(q)
        for v in self.views:
            if canonical_view(v.view_name) == qc:
                return v
        # 3. base_view fallback: prefer a clip whose full name IS the base query
        qb = base_view(q)
        base_matches = [v for v in self.views if base_view(v.view_name) == qb]
        for v in base_matches:
            if v.view_name.lower() == qb.lower():
                return v
        return base_matches[0] if base_matches else None


def build_manifest(preprocessed_dir: str, study_uuid: str) -> StudyManifest:
    sdir = study_dir(preprocessed_dir, study_uuid)
    clips = index_study(sdir)
    views = [ViewEntry(view_name=c.view, clip=c) for c in clips]
    return StudyManifest(views)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_manifest.py -q`
Expected: PASS (6 passed). Note: if `canonical_view`/`base_view` behave unexpectedly on the fixture names, read `echo_rl/data/views.py` and adjust the fixture clip names in `conftest.py` to match real conventions rather than bending the resolver.

- [ ] **Step 5: Commit**

```bash
git add echo_env/manifest.py echo_env/tests/test_manifest.py
git commit -m "feat(env): StudyManifest + view-name resolution over echo_rl index"
```

---

## Task 6: `tools.select_view`

**Files:**
- Create: `echo_env/tools.py` (first of three handlers)
- Test: `echo_env/tests/test_tools_select_view.py`

**Interfaces:**
- Consumes: `echo_env.manifest.StudyManifest`, `echo_env.frames.FrameLoader`, `echo_env.config.EnvConfig`, `echo_env.observation.Observation`/`FrameImg`, `echo_rl.data.frames.evenly_spaced`.
- Produces:
  - `echo_env.tools.select_view(manifest, loader, cfg, view_name) -> Observation`. On unknown view → `Observation.failure("select_view", "unknown view '<name>'; available: <sorted names>")`. On success → `n_preview_frames` (or fewer if clip shorter) evenly-spaced frames via `evenly_spaced(frame_count, cfg.n_preview_frames)`, each `loader.load`→`loader.downscale(..., cfg.preview_max_side)`, wrapped as `FrameImg(kind="preview")`; `text` lists view + returned indices.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_tools_select_view.py`:

```python
from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import select_view


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_preview_frames=5, preview_max_side=168)
    m = build_manifest(preprocessed_dir, study_uuid)
    return cfg, m, PILFrameLoader()


def test_select_view_returns_preview_frames(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "A4C")   # 10-frame clip
    assert obs.ok
    assert obs.n_frames == 5
    assert all(f.kind == "preview" for f in obs.frames)
    # downscaled to preview_max_side
    assert loader.size(obs.frames[0].image) == (168, 168)
    # evenly spaced across 10 frames -> deterministic indices
    idxs = [f.frame_index for f in obs.frames]
    assert idxs == sorted(idxs) and idxs[0] == 0 and idxs[-1] == 9


def test_select_view_short_clip(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "PSAX Apex")  # 4 frames < 5 requested
    assert obs.ok
    assert obs.n_frames <= 4


def test_select_view_unknown(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "A2C")
    assert not obs.ok
    assert "unknown view" in obs.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_tools_select_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.tools'`.

- [ ] **Step 3: Implement `echo_env/tools.py` (select_view only for now)**

```python
from echo_env.observation import Observation, FrameImg
from echo_rl.data.frames import evenly_spaced


def _unknown(tool, view_name, manifest):
    avail = ", ".join(manifest.view_names())
    return Observation.failure(tool, f"unknown view '{view_name}'; available: {avail}")


def select_view(manifest, loader, cfg, view_name) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("select_view", view_name, manifest)
    n = entry.frame_count
    idxs = evenly_spaced(n, cfg.n_preview_frames)
    frames = []
    for i in idxs:
        img = loader.downscale(loader.load(entry.clip.frame_path(i)), cfg.preview_max_side)
        frames.append(FrameImg(image=img, view_name=entry.view_name, frame_index=i, kind="preview"))
    text = f"{entry.view_name}: preview frames {idxs} of {n}"
    return Observation(tool="select_view", frames=frames, text=text)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_tools_select_view.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/tools.py echo_env/tests/test_tools_select_view.py
git commit -m "feat(env): select_view tool (sparse temporal preview)"
```

---

## Task 7: `tools.select_frames`

**Files:**
- Modify: `echo_env/tools.py`
- Test: `echo_env/tests/test_tools_select_frames.py`

**Interfaces:**
- Consumes: same as Task 6.
- Produces:
  - `echo_env.tools.select_frames(manifest, loader, cfg, view_name, indices) -> Observation`. `indices` is a list of ints. Unknown view → failure (same message pattern, tool `"select_frames"`). Out-of-range / non-int indices are dropped with a note; if **all** are invalid → failure `"no valid frame indices for <view> (0..N-1)"`. Valid indices are de-duplicated, sorted, capped at `cfg.n_highres_frames`, loaded at native res then `downscale(..., cfg.highres_max_side)` (a no-op at 336), wrapped `FrameImg(kind="highres")`.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_tools_select_frames.py`:

```python
from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import select_frames


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_highres_frames=8, highres_max_side=336)
    return cfg, build_manifest(preprocessed_dir, study_uuid), PILFrameLoader()


def test_select_specific_frames(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [2, 5, 7])
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [2, 5, 7]
    assert all(f.kind == "highres" for f in obs.frames)
    # native resolution preserved
    assert loader.size(obs.frames[0].image) == (336, 336)
    # marker check: frame 2's R channel == 2
    assert obs.frames[0].image.getpixel((0, 0))[0] == 2


def test_dedup_sort_and_drop_out_of_range(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [5, 5, 2, 99])  # 99 out of 0..9
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [2, 5]


def test_cap_at_n_highres(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    cfg.n_highres_frames = 3
    obs = select_frames(m, loader, cfg, "A4C", [0, 1, 2, 3, 4, 5])
    assert obs.n_frames == 3
    assert [f.frame_index for f in obs.frames] == [0, 1, 2]


def test_all_invalid_indices(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [99, -1, 200])
    assert not obs.ok


def test_unknown_view(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A2C", [0])
    assert not obs.ok
    assert "unknown view" in obs.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_tools_select_frames.py -q`
Expected: FAIL — `ImportError: cannot import name 'select_frames'`.

- [ ] **Step 3: Add `select_frames` to `echo_env/tools.py`**

```python
def _clean_indices(indices, n):
    out = []
    for v in indices or []:
        if isinstance(v, bool):        # bool is a subclass of int; reject
            continue
        if isinstance(v, int) and 0 <= v < n:
            out.append(v)
    return sorted(set(out))


def select_frames(manifest, loader, cfg, view_name, indices) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("select_frames", view_name, manifest)
    n = entry.frame_count
    valid = _clean_indices(indices, n)[: cfg.n_highres_frames]
    if not valid:
        return Observation.failure(
            "select_frames", f"no valid frame indices for {entry.view_name} (0..{n-1})")
    frames = []
    for i in valid:
        img = loader.downscale(loader.load(entry.clip.frame_path(i)), cfg.highres_max_side)
        frames.append(FrameImg(image=img, view_name=entry.view_name, frame_index=i, kind="highres"))
    text = f"{entry.view_name}: frames {valid} of {n}"
    return Observation(tool="select_frames", frames=frames, text=text)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_tools_select_frames.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/tools.py echo_env/tests/test_tools_select_frames.py
git commit -m "feat(env): select_frames tool (temporal high-res selection)"
```

---

## Task 8: `tools.zoom`

**Files:**
- Modify: `echo_env/tools.py`
- Test: `echo_env/tests/test_tools_zoom.py`

**Interfaces:**
- Consumes: Task 6/7 plus `echo_env.bbox.normalize_bbox`.
- Produces:
  - `echo_env.tools.zoom(manifest, loader, cfg, view_name, bbox, frame_indices) -> Observation`. Unknown view → failure. `frame_indices` cleaned like `select_frames` (dropped-if-out-of-range, dedup, sorted, capped `cfg.n_highres_frames`); if empty → default to `[midframe(n)]` (via `echo_rl.data.frames.midframe`). `bbox` normalized per-frame at that frame's native size via `normalize_bbox(bbox, w, h, cfg.min_crop_side, cfg.max_aspect)`; invalid bbox → failure `"invalid bbox <bbox>"`. Each chosen frame is loaded at native res, cropped to the normalized bbox, upscaled toward `cfg.highres_max_side` only if the crop's longest side `< cfg.min_crop_side` (never below the patch floor), wrapped `FrameImg(kind="crop", bbox=<normalized>)`.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_tools_zoom.py`:

```python
from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import zoom


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, min_crop_side=28,
                    n_highres_frames=8, highres_max_side=336)
    return cfg, build_manifest(preprocessed_dir, study_uuid), PILFrameLoader()


def test_zoom_single_frame(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[3])
    assert obs.ok
    assert obs.n_frames == 1
    f = obs.frames[0]
    assert f.kind == "crop"
    assert f.frame_index == 3
    assert f.bbox == (40, 40, 200, 240)
    assert loader.size(f.image) == (160, 200)


def test_zoom_multi_frame(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[1, 3, 5])
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [1, 3, 5]


def test_zoom_defaults_to_midframe(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[])
    assert obs.ok
    assert obs.n_frames == 1  # midframe of the 10-frame clip


def test_zoom_invalid_bbox(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [200, 200, 50, 50], frame_indices=[0])
    assert not obs.ok
    assert "invalid bbox" in obs.error


def test_zoom_unknown_view(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A2C", [0, 0, 50, 50], frame_indices=[0])
    assert not obs.ok
    assert "unknown view" in obs.error


def test_zoom_tiny_bbox_expanded(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [160, 160, 168, 168], frame_indices=[0])
    assert obs.ok
    w, h = loader.size(obs.frames[0].image)
    assert min(w, h) >= 28
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_tools_zoom.py -q`
Expected: FAIL — `ImportError: cannot import name 'zoom'`.

- [ ] **Step 3: Add `zoom` to `echo_env/tools.py`**

Add the import at the top of `echo_env/tools.py`:

```python
from echo_env.bbox import normalize_bbox
from echo_rl.data.frames import evenly_spaced, midframe
```

(adjust the existing `from echo_rl.data.frames import evenly_spaced` line to include `midframe`.)

Then:

```python
def zoom(manifest, loader, cfg, view_name, bbox, frame_indices) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("zoom", view_name, manifest)
    n = entry.frame_count
    valid = _clean_indices(frame_indices, n)[: cfg.n_highres_frames]
    if not valid:
        valid = [midframe(n)]
    frames = []
    for i in valid:
        img = loader.load(entry.clip.frame_path(i))
        w, h = loader.size(img)
        nb = normalize_bbox(bbox, w, h, cfg.min_crop_side, cfg.max_aspect)
        if nb is None:
            return Observation.failure("zoom", f"invalid bbox {bbox}")
        crop = loader.crop(img, nb)
        cw, ch = loader.size(crop)
        if max(cw, ch) < cfg.min_crop_side:
            crop = loader.downscale(crop, cfg.min_crop_side)
        frames.append(FrameImg(image=crop, view_name=entry.view_name,
                               frame_index=i, kind="crop", bbox=nb))
    text = f"{entry.view_name}: zoom {frames[0].bbox} on frames {valid} of {n}"
    return Observation(tool="zoom", frames=frames, text=text)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_tools_zoom.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/tools.py echo_env/tests/test_tools_zoom.py
git commit -m "feat(env): zoom tool (spatiotemporal crop)"
```

---

## Task 9: `parse.py` — action parsing

**Files:**
- Create: `echo_env/parse.py`
- Test: `echo_env/tests/test_parse.py`

**Interfaces:**
- Consumes: nothing (pure regex/JSON).
- Produces:
  - `echo_env.parse.ParsedAction` dataclass: `answer: str | None`, `calls: list[dict]` (each `{"name": str, "arguments": dict}`), `errors: list[str]` (malformed-JSON notes).
  - `echo_env.parse.parse_action(action_string: str) -> ParsedAction`. Extract the **last** `<answer>...</answer>` if present (→ `answer`). Extract every `<tool_call>...</tool_call>`, `json.loads` each; well-formed dicts with a `"name"` go to `calls` (missing `"arguments"` defaults to `{}`); malformed ones append a message to `errors` and are skipped. Mirrors `visual_toolbox_v5.extract_action` / `extract_answer`.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_parse.py`:

```python
from echo_env.parse import parse_action


def test_parse_answer():
    p = parse_action("reasoning... <answer>EF is 55%</answer>")
    assert p.answer == "EF is 55%"
    assert p.calls == []


def test_parse_single_tool_call():
    s = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    p = parse_action(s)
    assert p.answer is None
    assert len(p.calls) == 1
    assert p.calls[0]["name"] == "select_view"
    assert p.calls[0]["arguments"]["view_name"] == "A4C"


def test_parse_multiple_tool_calls():
    s = ('<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
         '<tool_call>{"name": "select_view", "arguments": {"view_name": "PLAX"}}</tool_call>')
    p = parse_action(s)
    assert len(p.calls) == 2


def test_parse_malformed_json_recorded_not_raised():
    s = '<tool_call>{"name": "zoom", "arguments": {oops}}</tool_call>'
    p = parse_action(s)
    assert p.calls == []
    assert len(p.errors) == 1


def test_missing_arguments_defaults_empty():
    s = '<tool_call>{"name": "select_view"}</tool_call>'
    p = parse_action(s)
    assert p.calls[0]["arguments"] == {}


def test_last_answer_wins():
    p = parse_action("<answer>first</answer> more <answer>second</answer>")
    assert p.answer == "second"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_parse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.parse'`.

- [ ] **Step 3: Implement `echo_env/parse.py`**

```python
import re
import json
from dataclasses import dataclass, field

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ParsedAction:
    answer: str | None = None
    calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def parse_action(action_string: str) -> ParsedAction:
    result = ParsedAction()
    answers = _ANSWER_RE.findall(action_string or "")
    if answers:
        result.answer = answers[-1].strip()
    for raw in _TOOL_RE.findall(action_string or ""):
        try:
            obj = json.loads(raw.strip())
        except (ValueError, TypeError) as e:
            result.errors.append(f"malformed tool_call JSON: {e}")
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            result.errors.append(f"tool_call missing 'name': {raw.strip()[:80]}")
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        result.calls.append({"name": obj["name"], "arguments": args})
    return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_parse.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_env/parse.py echo_env/tests/test_parse.py
git commit -m "feat(env): parse_action (tool_call + answer extraction)"
```

---

## Task 10: `budget.py` + `env.py` — the `EchoEnv` orchestrator

**Files:**
- Create: `echo_env/budget.py`, `echo_env/env.py`
- Test: `echo_env/tests/test_budget.py`, `echo_env/tests/test_env.py`

**Interfaces:**
- Consumes: all prior tasks (`EnvConfig`, `build_manifest`, `PILFrameLoader`, `parse_action`, the three tool handlers, `Observation`).
- Produces:
  - `echo_env.budget.Budget`: constructed with `cfg`. Fields `tool_calls: int`, `total_frames: int`, `seen: set`. Methods: `can_call() -> bool` (`tool_calls < cfg.max_tool_calls`); `signature(name, arguments) -> str` (stable JSON key for dedup); `seen_before(name, arguments) -> bool`; `register(name, arguments, obs) -> None` (increments `tool_calls`, adds signature, adds `obs.n_frames` to `total_frames`); `frames_left() -> int` (`max(0, cfg.max_total_frames - total_frames)`).
  - `echo_env.env.EchoEnv`: `__init__(cfg, loader=None)` (defaults to `PILFrameLoader()`). `reset(study_uuid) -> Observation` — builds manifest + fresh `Budget`, returns an overview `Observation` (tool `"reset"`): one midframe thumbnail per view (up to `cfg.n_overview_views`), downscaled to `cfg.preview_max_side`, `kind="thumbnail"`; `text` = a listing of `view_name: N frames`. `step(action_string) -> (Observation, float, bool, dict)` mirroring `ToolBase.execute`:
    - `parse_action`; if `answer` present → `("", 0.0, True, {"answer": answer})` (episode ends; matches `visual_toolbox_v5`).
    - No calls and no answer → `(Observation.failure("step", "no tool_call or answer found"), 0.0, False, {...})`.
    - For each parsed call (cap the list at `cfg.max_frames_per_obs` tool calls per turn — DeepEyes uses `max_action_per_turn=3`, we reuse `max_frames_per_obs` as the per-turn cap): reject with a failure Observation if `not budget.can_call()` (budget exhausted → text asks the model to answer); dispatch by name to `select_view`/`select_frames`/`zoom` reading args (`view_name`, `indices`/`frame_indices`, `bbox`); unknown tool name → failure. Truncate returned frames to `budget.frames_left()`. `budget.register(...)`. Combine multiple calls' frames into one merged `Observation` (concatenated frames + joined text). Always `reward=0.0`, `done=False`. `info` carries `tool_calls`, `total_frames`, and any per-call errors.

- [ ] **Step 1: Write failing `budget` tests**

`echo_env/tests/test_budget.py`:

```python
from echo_env.config import EnvConfig
from echo_env.budget import Budget
from echo_env.observation import Observation, FrameImg


def _obs(n):
    return Observation(tool="select_view",
                       frames=[FrameImg(object(), "A4C", i, "preview") for i in range(n)])


def test_can_call_limit():
    b = Budget(EnvConfig(max_tool_calls=2))
    assert b.can_call()
    b.register("select_view", {"view_name": "A4C"}, _obs(3))
    b.register("select_view", {"view_name": "PLAX"}, _obs(3))
    assert not b.can_call()


def test_frames_left():
    b = Budget(EnvConfig(max_total_frames=10))
    b.register("select_view", {"view_name": "A4C"}, _obs(4))
    assert b.frames_left() == 6


def test_dedup_signature():
    b = Budget(EnvConfig())
    assert not b.seen_before("select_view", {"view_name": "A4C"})
    b.register("select_view", {"view_name": "A4C"}, _obs(1))
    assert b.seen_before("select_view", {"view_name": "A4C"})
    assert not b.seen_before("select_view", {"view_name": "PLAX"})
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest echo_env/tests/test_budget.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.budget'`.

- [ ] **Step 3: Implement `echo_env/budget.py`**

```python
import json


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tool_calls = 0
        self.total_frames = 0
        self.seen = set()

    def can_call(self) -> bool:
        return self.tool_calls < self.cfg.max_tool_calls

    def signature(self, name, arguments) -> str:
        return name + ":" + json.dumps(arguments, sort_keys=True, default=str)

    def seen_before(self, name, arguments) -> bool:
        return self.signature(name, arguments) in self.seen

    def register(self, name, arguments, obs) -> None:
        self.tool_calls += 1
        self.seen.add(self.signature(name, arguments))
        self.total_frames += obs.n_frames

    def frames_left(self) -> int:
        return max(0, self.cfg.max_total_frames - self.total_frames)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_budget.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Write failing `env` tests**

`echo_env/tests/test_env.py`:

```python
from echo_env.config import EnvConfig
from echo_env.env import EchoEnv


def _env(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_preview_frames=3,
                    max_tool_calls=2, max_total_frames=8, max_frames_per_obs=3)
    env = EchoEnv(cfg)
    return env, study_uuid


def test_reset_overview_thumbnails(study_fixture):
    env, study_uuid = _env(study_fixture)
    obs = env.reset(study_uuid)
    assert obs.ok
    # one thumbnail per fixture view (4 views)
    assert obs.n_frames == 4
    assert all(f.kind == "thumbnail" for f in obs.frames)


def test_step_select_view(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    obs, reward, done, info = env.step(action)
    assert obs.ok
    assert obs.n_frames == 3
    assert reward == 0.0
    assert done is False
    assert info["tool_calls"] == 1


def test_step_answer_terminates(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step("<answer>Normal LV function</answer>")
    assert done is True
    assert info["answer"] == "Normal LV function"


def test_step_zoom(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = ('<tool_call>{"name": "zoom", "arguments": '
              '{"view_name": "A4C", "bbox": [40,40,200,240], "frame_indices": [3]}}</tool_call>')
    obs, reward, done, info = env.step(action)
    assert obs.ok
    assert obs.frames[0].kind == "crop"


def test_budget_exhaustion_blocks_calls(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    a = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    b = '<tool_call>{"name": "select_view", "arguments": {"view_name": "PLAX"}}</tool_call>'
    c = '<tool_call>{"name": "select_view", "arguments": {"view_name": "PSAX Apex"}}</tool_call>'
    env.step(a)
    env.step(b)              # now tool_calls == 2 == max
    obs, reward, done, info = env.step(c)
    assert not obs.ok        # blocked, asks to answer
    assert info["tool_calls"] == 2


def test_step_no_action(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step("just some prose, no tags")
    assert not obs.ok
    assert done is False


def test_step_unknown_tool(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step(
        '<tool_call>{"name": "teleport", "arguments": {}}</tool_call>')
    assert not obs.ok
```

- [ ] **Step 6: Run to verify fail**

Run: `.venv/bin/pytest echo_env/tests/test_env.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.env'`.

- [ ] **Step 7: Implement `echo_env/env.py`**

```python
from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.observation import Observation, FrameImg
from echo_env.budget import Budget
from echo_env.parse import parse_action
from echo_env.tools import select_view, select_frames, zoom
from echo_rl.data.frames import midframe


class EchoEnv:
    def __init__(self, cfg: EnvConfig, loader=None):
        self.cfg = cfg
        self.loader = loader or PILFrameLoader()
        self.manifest = None
        self.budget = None

    def reset(self, study_uuid: str) -> Observation:
        self.manifest = build_manifest(self.cfg.preprocessed_dir, study_uuid)
        self.budget = Budget(self.cfg)
        frames = []
        lines = []
        for entry in self.manifest.overview(self.cfg.n_overview_views):
            n = entry.frame_count
            i = midframe(n)
            img = self.loader.downscale(
                self.loader.load(entry.clip.frame_path(i)), self.cfg.preview_max_side)
            frames.append(FrameImg(image=img, view_name=entry.view_name,
                                   frame_index=i, kind="thumbnail"))
            lines.append(f"{entry.view_name}: {n} frames")
        text = "Available views:\n" + "\n".join(lines)
        return Observation(tool="reset", frames=frames, text=text)

    def _dispatch(self, name, args):
        if name == "select_view":
            return select_view(self.manifest, self.loader, self.cfg, args.get("view_name"))
        if name == "select_frames":
            return select_frames(self.manifest, self.loader, self.cfg,
                                 args.get("view_name"), args.get("indices", []))
        if name == "zoom":
            return zoom(self.manifest, self.loader, self.cfg, args.get("view_name"),
                        args.get("bbox"), args.get("frame_indices", []))
        return Observation.failure(name or "unknown", f"unknown tool '{name}'")

    def step(self, action_string: str):
        parsed = parse_action(action_string)
        if parsed.answer is not None:
            return "", 0.0, True, {"answer": parsed.answer}
        if not parsed.calls:
            info = {"tool_calls": self.budget.tool_calls,
                    "total_frames": self.budget.total_frames, "errors": parsed.errors}
            return Observation.failure("step", "no tool_call or answer found"), 0.0, False, info

        merged_frames = []
        texts = []
        errors = list(parsed.errors)
        for call in parsed.calls[: self.cfg.max_frames_per_obs]:
            if not self.budget.can_call():
                errors.append("tool budget exhausted; provide <answer>")
                break
            obs = self._dispatch(call["name"], call["arguments"])
            if not obs.ok:
                errors.append(obs.error)
                # a failed call still counts as an attempt
                self.budget.register(call["name"], call["arguments"], obs)
                continue
            keep = obs.frames[: self.budget.frames_left()]
            obs.frames = keep
            self.budget.register(call["name"], call["arguments"], obs)
            merged_frames.extend(keep)
            texts.append(obs.text)

        info = {"tool_calls": self.budget.tool_calls,
                "total_frames": self.budget.total_frames, "errors": errors}
        if not merged_frames:
            msg = "; ".join(errors) or "no frames returned"
            return Observation.failure("step", msg), 0.0, False, info
        return (Observation(tool="step", frames=merged_frames, text="\n".join(texts)),
                0.0, False, info)
```

- [ ] **Step 8: Run env tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_env.py -q`
Expected: PASS (7 passed). Note the budget-exhaustion test: `max_tool_calls=2`, and each `select_view` on the 4-frame-plus clips returns ≤3 frames; the third `step` is blocked at the `can_call()` gate — `obs.ok` is False and `tool_calls` stays 2.

- [ ] **Step 9: Commit**

```bash
git add echo_env/budget.py echo_env/env.py echo_env/tests/test_budget.py echo_env/tests/test_env.py
git commit -m "feat(env): Budget guardrails + EchoEnv orchestrator (reset/step)"
```

---

## Task 11: `packaging.py` (DeepEyes seam) + `INTEGRATION.md` (Phase-3 contract)

**Files:**
- Create: `echo_env/packaging.py`, `echo_env/INTEGRATION.md`
- Modify: `echo_env/__init__.py` (export `EchoEnv`, `Observation`, `to_deepeyes_obs`)
- Test: `echo_env/tests/test_packaging.py`

**Interfaces:**
- Consumes: `echo_env.observation.Observation`/`FrameImg`.
- Produces:
  - `echo_env.packaging.to_deepeyes_obs(observation, user_prompt: str) -> dict` — pure. Builds the DeepEyes multi-turn observation dict exactly as `visual_toolbox_v5` does, one `<tool_response>\n<image>\n</tool_response>\n` block per frame:
    ```python
    {
      "prompt": "<|im_end|>\n<|im_start|>user\n" + tool_response*n + user_prompt
                + "<|im_end|>\n<|im_start|>assistant\n",
      "multi_modal_data": {"image": [f.image for f in observation.frames]},
    }
    ```
    For a failed observation (`ok=False`, no frames) → a text-only dict: `{"prompt": "<|im_end|>\n<|im_start|>user\nError: <error><|im_end|>\n<|im_start|>assistant\n"}` (no `multi_modal_data`), matching `visual_toolbox_v5`'s error path.

- [ ] **Step 1: Write the failing tests**

`echo_env/tests/test_packaging.py`:

```python
from echo_env.observation import Observation, FrameImg
from echo_env.packaging import to_deepeyes_obs


def test_packaging_multi_image():
    obs = Observation(tool="step", frames=[
        FrameImg(image="IMG0", view_name="A4C", frame_index=0, kind="preview"),
        FrameImg(image="IMG1", view_name="A4C", frame_index=5, kind="preview"),
    ], text="ok")
    out = to_deepeyes_obs(obs, user_prompt="Continue.")
    assert out["multi_modal_data"]["image"] == ["IMG0", "IMG1"]
    assert out["prompt"].count("<image>") == 2
    assert out["prompt"].endswith("<|im_start|>assistant\n")
    assert "Continue." in out["prompt"]


def test_packaging_failure_is_text_only():
    obs = Observation.failure("zoom", "invalid bbox [1,2,3,4]")
    out = to_deepeyes_obs(obs, user_prompt="Continue.")
    assert "multi_modal_data" not in out
    assert "Error: invalid bbox" in out["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest echo_env/tests/test_packaging.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_env.packaging'`.

- [ ] **Step 3: Implement `echo_env/packaging.py`**

```python
_HEAD = "<|im_end|>\n<|im_start|>user\n"
_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_TOOL_RESPONSE = "<tool_response>\n<image>\n</tool_response>\n"


def to_deepeyes_obs(observation, user_prompt: str) -> dict:
    if not observation.ok or observation.n_frames == 0:
        return {"prompt": _HEAD + f"Error: {observation.error or observation.text}" + _TAIL}
    tool_response = _TOOL_RESPONSE * observation.n_frames
    return {
        "prompt": _HEAD + tool_response + user_prompt + _TAIL,
        "multi_modal_data": {"image": [f.image for f in observation.frames]},
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest echo_env/tests/test_packaging.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Update `echo_env/__init__.py`**

```python
from echo_env.config import EnvConfig
from echo_env.observation import Observation, FrameImg
from echo_env.env import EchoEnv
from echo_env.packaging import to_deepeyes_obs

__all__ = ["EnvConfig", "Observation", "FrameImg", "EchoEnv", "to_deepeyes_obs"]
```

- [ ] **Step 6: Write `echo_env/INTEGRATION.md` (Phase-3 wiring contract)**

Create `echo_env/INTEGRATION.md` with this content:

```markdown
# Phase-3 Integration Contract — wiring `echo_env` into DeepEyes

`echo_env` is DeepEyes-runtime-free. Phase 3 wires it in via **net-new files** +
a **two-file video patch** to the pinned submodule `external/DeepEyes`
(commit 11d20c6). Nothing below runs offline — it needs the torch/vLLM runtime.

## 1. `ToolBase` adapter (net-new, registered by import)

`external/DeepEyes/verl/workers/agent/envs/echo/echo_env.py`:

    from verl.workers.agent.tool_envs import ToolBase
    from echo_env import EchoEnv, EnvConfig, to_deepeyes_obs

    class EchoToolEnv(ToolBase):
        name = "echo"
        user_prompt = "..."   # echo turn prompt (Phase-3 prompt design)

        def __init__(self, _n, _d, _p, **kw):
            super().__init__(name=self.name)
            self.env = EchoEnv(EnvConfig.from_env())

        def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kw):
            # study_uuid arrives via the dataset row (extra_info / a dedicated column);
            # Phase-3 data-gen must put it where reset() can read it.
            self.env.reset(study_uuid)

        def execute(self, action_string, **kw):
            obs, reward, done, info = self.env.step(action_string)
            if done or not hasattr(obs, "frames"):
                return "", reward, done, info
            return to_deepeyes_obs(obs, self.user_prompt), reward, done, info

Registration = one import line in the launch/entry module (preferred) so upstream
`verl/workers/agent/__init__.py` stays untouched; patch it only if import ordering forces it.
Dataset rows set `env_name="echo"` so `ToolBase.create("echo")` finds this class.

## 2. Video patch (the one irreducible in-tree edit — `[patch]` in spec §8)

`to_deepeyes_obs` currently emits **`<image>`** blocks (multi-image), which already
works through DeepEyes unchanged. TRUE video (cine motion + temporal mRoPE) is a
separate, better-but-harder path requiring:

- `verl/workers/agent/parallel_env.py`: add a `<video>` →
  `<|vision_start|><|video_pad|><|vision_end|>` branch in `_preprocess_multi_modal_inputs`
  (call `processor(videos=...)`) and extend the obs merge-back to append `mm_data['video']`
  + `video_grid_thw`/`second_per_grid_ts`.
- `verl/utils/dataset/rl_dataset.py`: also populate `origin_multi_modal_data["video"]`.

**Decision (2026-07-24):** Phase 2 ships the image-packaging seam; Phase 3 chooses
image-multi-frame vs. true-video per empirical token/quality tradeoff. If video is chosen,
the two edits live as a versioned patch under `external/patches/echo-video-*.patch`,
applied to the pinned submodule at setup (per the "submodule + patch set" decision).
mRoPE metadata (`video_grid_thw`, `second_per_grid_ts`) MUST come from the real HF
processor — never hand-rolled — or temporal position ids break silently.

## 3. Reward scorer (net-new) — Phase 3

`verl/utils/reward_score/echo.py` + dispatch on `data_source="echo"`. See spec §5.2 / §8.
```

- [ ] **Step 7: Run the full `echo_env` suite**

Run: `.venv/bin/pytest echo_env -q`
Expected: ALL pass, 0 collection errors. Also confirm no regression in Phase 1: `.venv/bin/pytest echo_rl -q` → still all pass.

- [ ] **Step 8: Commit**

```bash
git add echo_env/packaging.py echo_env/INTEGRATION.md echo_env/__init__.py echo_env/tests/test_packaging.py
git commit -m "feat(env): to_deepeyes_obs packaging seam + Phase-3 INTEGRATION contract"
```

---

## Self-Review (completed during authoring)

**1. Spec coverage** (spec §4 + Phase-2 scope in §10):
- Initial cheap observation (one thumbnail per view) → Task 10 `reset`. ✓
- `select_view` sparse temporal preview → Task 6. ✓
- `select_frames` temporal high-res → Task 7. ✓
- `zoom` spatiotemporal crop with bbox guardrails → Task 8 (+ Task 3 bbox math). ✓
- Free-form, graceful-failure tools (unknown view / bad indices / bad bbox never crash) → Tasks 6–8, 10. ✓
- Budget guardrails (tool-call cap, frame cap, dedup) → Task 10 `Budget`. ✓
- XML-wrapped-JSON tool-call + `<answer>` format → Task 9 `parse_action`, matches `visual_toolbox_v5`. ✓
- Frame server over `preprocessed_data` reusing `echo_rl` → Tasks 4, 5. ✓
- Unit-tested **offline**, DeepEyes-runtime-free → every task uses fixture PNGs; no `verl` import. ✓
- DeepEyes seam without forking → Task 11 `to_deepeyes_obs` + `INTEGRATION.md`; live adapter/patch deferred to Phase 3 per spec §8. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step has complete code; every failure path has an explicit message. ✓

**3. Type consistency:** `Observation`/`FrameImg` field names (`frames`, `n_frames`, `ok`, `error`, `kind`, `frame_index`, `bbox`) are used identically across Tasks 2, 6–8, 10, 11. `normalize_bbox` signature `(bbox, width, height, min_side, max_aspect)` consistent between Tasks 3 and 8. `EnvConfig` field names consistent across all consumers. Tool handler signatures `(manifest, loader, cfg, ...)` consistent between Tasks 6–8 and the `EchoEnv._dispatch` caller in Task 10. `to_deepeyes_obs(observation, user_prompt)` consistent between Tasks 11 def and `INTEGRATION.md` usage. ✓
