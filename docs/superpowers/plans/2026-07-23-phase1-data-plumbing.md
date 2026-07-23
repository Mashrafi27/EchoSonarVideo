# Echo DeepEyes — Phase 1: Data Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU-free data layer that joins the echo VQA text to the on-disk PNG frames and emits (a) synthetic agentic SFT trajectories and (b) balanced RL prompt records with reward keys — the foundation every later phase consumes.

**Architecture:** A small stdlib-only Python package `echo_rl` that (1) indexes each `study_uuid` to its labeled clip folders and frame files, (2) parses the per-view `thinking` report into an ordered section list, (3) rewrites each QA into a tool-using trajectory (`select_view` turns → answer), (4) extracts reward targets and gold-CSV values, and (5) produces study-disjoint splits and a class-balanced RL sampling pool. Everything is pure text/filesystem logic, unit-testable without a GPU or heavy deps.

**Tech Stack:** Python 3.13 (existing conda base), pytest (installed into a local `.venv`), Python standard library only (`json`, `csv`, `re`, `os`, `hashlib`, `random`, `dataclasses`, `argparse`). **No** pandas / pyarrow / Pillow in this phase.

## Global Constraints

- Python interpreter: `python3` (3.13.5). Tests run via `.venv/bin/pytest`.
- **Dependencies:** standard library + `pytest` only. Do NOT add pandas, pyarrow, numpy, or Pillow in Phase 1. Frame *loading/decoding* is deferred to the Phase-2 environment; Phase 1 only manipulates **paths** and text.
- Package root: `echo_rl/` under `/vast/users/mohammad.yaqub/project/EchoSonarVideo/`.
- Data paths (defaults; overridable via env vars in `echo_rl/config.py`):
  - `PREPROCESSED_DIR = ../preprocessed_data` → `/vast/users/mohammad.yaqub/project/preprocessed_data`
  - `VQA_TRAIN = Archive 2 (1)/train_vqa_with_thinking.jsonl`
  - `VQA_TEST  = Archive 2 (1)/test_vqa.jsonl`
  - `GOLD_DIR  = ../output_with_labels/output`
  - `OUT_DIR   = build/` (created; git-ignored)
- **Verified data invariants (rely on these, assert them in code):**
  - Study folder = `preprocessed_data/<study_uuid>/`; clip subfolder = `di-XXXX-XXXX-XXXX_<ViewName>`; **view label is the substring after the first `_`.**
  - Frames are `0.png … N.png` inside each clip folder (integer stems, not zero-padded).
  - Gold CSV `study_id` **equals** `study_uuid`; columns: `study_id,label,study designation,from_measurement,from_text,text,text_fields`.
  - Train QA record keys: `study_uuid`, `question_type`, `thinking`, `messages` (a JSON **string** of a chat list; last `assistant` turn = target answer).
  - `thinking` view sections start with `#### <n>. **<ViewName>**`.
- **Non-negotiable design rules (from spec §2, §6):**
  - Splits are **study-level** (all QA of a study on one side).
  - The RL pool is **class-balanced / abnormal-enriched** — a prior-parroting ("always No") policy must score badly.
  - SFT trajectories are a **behavioral prior** (format + look-then-reason), NOT ground-truth grounding — keep synthesis simple; do not over-engineer per-view finding attribution.

---

### Task 1: Repo init, package scaffold, config, test harness

**Files:**
- Create: `.gitignore`, `echo_rl/__init__.py`, `echo_rl/config.py`, `echo_rl/data/__init__.py`, `echo_rl/tests/__init__.py`, `echo_rl/tests/test_config.py`

**Interfaces:**
- Produces: `echo_rl.config.Config` dataclass with fields `preprocessed_dir, vqa_train, vqa_test, gold_dir, out_dir` (all `str`), and classmethod `Config.from_env() -> Config` resolving the Global-Constraints defaults, plus params `n_overview_views:int=19`, `n_preview_frames:int=5`, `n_highres_frames:int=8`, `seed:int=0`.

- [ ] **Step 1: Initialize git and venv**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
git init
python3 -m venv .venv
.venv/bin/pip install --quiet pytest
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
build/
__pycache__/
*.pyc
```

- [ ] **Step 3: Write the failing test** — `echo_rl/tests/test_config.py`

```python
import os
from echo_rl.config import Config


def test_from_env_defaults():
    cfg = Config.from_env()
    assert cfg.preprocessed_dir.endswith("preprocessed_data")
    assert cfg.vqa_train.endswith("train_vqa_with_thinking.jsonl")
    assert cfg.n_preview_frames == 5
    assert cfg.n_highres_frames == 8


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("ECHO_PREPROCESSED_DIR", "/tmp/foo")
    cfg = Config.from_env()
    assert cfg.preprocessed_dir == "/tmp/foo"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_rl.config'`.

- [ ] **Step 5: Write minimal implementation** — `echo_rl/__init__.py` (empty), `echo_rl/data/__init__.py` (empty), `echo_rl/tests/__init__.py` (empty), and `echo_rl/config.py`:

```python
import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)                       # .../EchoSonarVideo
_PARENT = os.path.dirname(_PROJ)                     # .../project


@dataclass
class Config:
    preprocessed_dir: str
    vqa_train: str
    vqa_test: str
    gold_dir: str
    out_dir: str
    n_overview_views: int = 19
    n_preview_frames: int = 5
    n_highres_frames: int = 8
    seed: int = 0

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            preprocessed_dir=os.environ.get(
                "ECHO_PREPROCESSED_DIR", os.path.join(_PARENT, "preprocessed_data")),
            vqa_train=os.environ.get(
                "ECHO_VQA_TRAIN", os.path.join(_PROJ, "Archive 2 (1)", "train_vqa_with_thinking.jsonl")),
            vqa_test=os.environ.get(
                "ECHO_VQA_TEST", os.path.join(_PROJ, "Archive 2 (1)", "test_vqa.jsonl")),
            gold_dir=os.environ.get(
                "ECHO_GOLD_DIR", os.path.join(_PARENT, "output_with_labels", "output")),
            out_dir=os.environ.get("ECHO_OUT_DIR", os.path.join(_PROJ, "build")),
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add .gitignore echo_rl docs
git commit -m "feat(data): scaffold echo_rl package + config"
```

---

### Task 2: View-label parsing

**Files:**
- Create: `echo_rl/data/views.py`, `echo_rl/tests/test_views.py`

**Interfaces:**
- Produces:
  - `parse_clip_dirname(name: str) -> tuple[str, str]` → `(di_id, view_name)`, splitting on the **first** `_`. Raises `ValueError` if no `_`.
  - `canonical_view(view_name: str) -> str` → trimmed, single-spaced, original casing preserved (e.g. `"A4C  Zoomed Mitral"` → `"A4C Zoomed Mitral"`).
  - `base_view(view_name: str) -> str` → first whitespace token (e.g. `"PLAX Standard"` → `"PLAX"`, `"PSAXA Zoomed Aorta"` → `"PSAXA"`).

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_views.py`

```python
import pytest
from echo_rl.data.views import parse_clip_dirname, canonical_view, base_view


def test_parse_clip_dirname():
    assert parse_clip_dirname("di-2040-A343-7E7F_PLAX Standard") == ("di-2040-A343-7E7F", "PLAX Standard")
    assert parse_clip_dirname("di-4503-8690-7008_PSAXA Zoomed Aorta")[1] == "PSAXA Zoomed Aorta"


def test_parse_clip_dirname_no_underscore():
    with pytest.raises(ValueError):
        parse_clip_dirname("di-2040-A343-7E7F")


def test_canonical_view():
    assert canonical_view("  A4C   Zoomed Mitral ") == "A4C Zoomed Mitral"


def test_base_view():
    assert base_view("PLAX Standard") == "PLAX"
    assert base_view("A4C") == "A4C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_views.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/views.py`

```python
def parse_clip_dirname(name: str) -> tuple[str, str]:
    if "_" not in name:
        raise ValueError(f"clip dir name has no view label: {name!r}")
    di_id, view = name.split("_", 1)
    return di_id, canonical_view(view)


def canonical_view(view_name: str) -> str:
    return " ".join(view_name.split())


def base_view(view_name: str) -> str:
    return canonical_view(view_name).split(" ")[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_views.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/views.py echo_rl/tests/test_views.py
git commit -m "feat(data): view-label parsing from clip dir names"
```

---

### Task 3: Study index (filesystem → clips + frames)

**Files:**
- Create: `echo_rl/data/studies.py`, `echo_rl/tests/test_studies.py`

**Interfaces:**
- Consumes: `echo_rl.data.views.parse_clip_dirname`.
- Produces:
  - `@dataclass Clip{ di_id:str, view:str, path:str, frame_files:list[str] }` where `frame_files` is the list of PNG basenames **sorted numerically** by integer stem.
  - `Clip.frame_count -> int` (property).
  - `Clip.frame_path(idx:int) -> str` → absolute path to the `idx`-th frame *in numeric order* (raises `IndexError` if out of range).
  - `index_study(study_dir:str) -> list[Clip]` → all clip subdirs (names starting `di-`), each with frames; skips empty clips; raises `FileNotFoundError` if `study_dir` missing.
  - `study_dir(preprocessed_dir:str, study_uuid:str) -> str`.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_studies.py`

```python
import os
import pytest
from echo_rl.data.studies import index_study, study_dir


def _make_study(tmp_path):
    root = tmp_path / "st-TEST-0000-0000"
    clip = root / "di-AAAA-BBBB-CCCC_A4C"
    clip.mkdir(parents=True)
    # deliberately out of lexical order to prove numeric sort
    for i in [0, 1, 2, 10, 11]:
        (clip / f"{i}.png").write_bytes(b"x")
    (root / "di-DDDD-EEEE-FFFF_PLAX Standard").mkdir()  # empty clip -> skipped
    for i in [0, 1]:
        (root / "di-DDDD-EEEE-FFFF_PLAX Standard" / f"{i}.png").write_bytes(b"x")
    return str(root)


def test_index_study(tmp_path):
    root = _make_study(tmp_path)
    clips = index_study(root)
    views = sorted(c.view for c in clips)
    assert views == ["A4C", "PLAX Standard"]
    a4c = next(c for c in clips if c.view == "A4C")
    assert a4c.frame_count == 5
    assert a4c.frame_files == ["0.png", "1.png", "2.png", "10.png", "11.png"]
    assert a4c.frame_path(3).endswith("/10.png")


def test_index_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_study(str(tmp_path / "nope"))


def test_study_dir():
    assert study_dir("/data", "st-1").endswith("/data/st-1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_studies.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/studies.py`

```python
import os
from dataclasses import dataclass
from echo_rl.data.views import parse_clip_dirname


def _num_stem(fname: str) -> int:
    return int(os.path.splitext(fname)[0])


@dataclass
class Clip:
    di_id: str
    view: str
    path: str
    frame_files: list

    @property
    def frame_count(self) -> int:
        return len(self.frame_files)

    def frame_path(self, idx: int) -> str:
        return os.path.join(self.path, self.frame_files[idx])


def study_dir(preprocessed_dir: str, study_uuid: str) -> str:
    return os.path.join(preprocessed_dir, study_uuid)


def index_study(study_dir_path: str) -> list:
    if not os.path.isdir(study_dir_path):
        raise FileNotFoundError(study_dir_path)
    clips = []
    for name in sorted(os.listdir(study_dir_path)):
        if not name.startswith("di-"):
            continue
        cpath = os.path.join(study_dir_path, name)
        if not os.path.isdir(cpath):
            continue
        frames = [f for f in os.listdir(cpath) if f.endswith(".png")]
        if not frames:
            continue
        frames.sort(key=_num_stem)
        di_id, view = parse_clip_dirname(name)
        clips.append(Clip(di_id=di_id, view=view, path=cpath, frame_files=frames))
    return clips
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_studies.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/studies.py echo_rl/tests/test_studies.py
git commit -m "feat(data): study index with numeric frame ordering"
```

---

### Task 4: Frame-selection helpers

**Files:**
- Create: `echo_rl/data/frames.py`, `echo_rl/tests/test_frames.py`

**Interfaces:**
- Produces:
  - `evenly_spaced(n_total:int, k:int) -> list[int]` → `k` distinct indices spanning `[0, n_total-1]` inclusive of both ends when `k>=2`; returns `[0]` for `k==1`; returns `list(range(n_total))` when `k>=n_total`.
  - `midframe(n_total:int) -> int` → `n_total // 2`.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_frames.py`

```python
from echo_rl.data.frames import evenly_spaced, midframe


def test_evenly_spaced_basic():
    assert evenly_spaced(42, 5) == [0, 10, 20, 31, 41]


def test_evenly_spaced_edges():
    assert evenly_spaced(10, 1) == [0]
    assert evenly_spaced(3, 5) == [0, 1, 2]
    r = evenly_spaced(100, 8)
    assert r[0] == 0 and r[-1] == 99 and len(set(r)) == 8


def test_midframe():
    assert midframe(42) == 21
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_frames.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/frames.py`

```python
def evenly_spaced(n_total: int, k: int) -> list:
    if n_total <= 0 or k <= 0:
        return []
    if k >= n_total:
        return list(range(n_total))
    if k == 1:
        return [0]
    step = (n_total - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def midframe(n_total: int) -> int:
    return n_total // 2
```

Note: `evenly_spaced(42,5)` → rounds of `{0,10.25,20.5,30.75,41}` = `[0,10,20,31,41]`. If a rounding collision ever drops below `k`, that is acceptable (dedup by design); tests assert the documented cases.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_frames.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/frames.py echo_rl/tests/test_frames.py
git commit -m "feat(data): frame-selection helpers"
```

---

### Task 5: Thinking-trace parser

**Files:**
- Create: `echo_rl/data/trace.py`, `echo_rl/tests/test_trace.py`

**Interfaces:**
- Produces:
  - `parse_thinking(thinking:str) -> list[tuple[str,str]]` → ordered `[(view_name, body), ...]` from `#### <n>. **<ViewName>**` headers; `body` is the text up to the next header (stripped). Returns `[]` if no headers.
  - `findings_text(body:str) -> str` → the `**Clinical Findings:**` block up to `**Implications` (or whole body if absent), stripped.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_trace.py`

```python
from echo_rl.data.trace import parse_thinking, findings_text

SAMPLE = (
    "### Report\n\n"
    "#### 1. **PLAX Standard View**\n"
    "- **Detected Structures:** LA, LV\n"
    "- **Clinical Findings:**\n  - LV normal.\n"
    "- **Implications:**\n  - none\n\n"
    "#### 2. **A4C View**\n"
    "- **Clinical Findings:**\n  - RV normal.\n"
)


def test_parse_thinking_sections():
    secs = parse_thinking(SAMPLE)
    assert [v for v, _ in secs] == ["PLAX Standard View", "A4C View"]
    assert "LV normal" in secs[0][1]


def test_findings_text():
    secs = parse_thinking(SAMPLE)
    ft = findings_text(secs[0][1])
    assert "LV normal" in ft and "Implications" not in ft


def test_parse_thinking_empty():
    assert parse_thinking("no headers here") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_trace.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/trace.py`

```python
import re

_HEADER = re.compile(r"####\s*\d+\.\s*\*\*(.+?)\*\*")


def parse_thinking(thinking: str) -> list:
    parts = _HEADER.split(thinking)
    # parts = [pre, view1, body1, view2, body2, ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i].strip(), parts[i + 1].strip()))
    return out


def findings_text(body: str) -> str:
    m = re.search(r"Clinical Findings:\*\*(.*?)(?:\*\*Implications|$)", body, re.S)
    return (m.group(1) if m else body).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_trace.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/trace.py echo_rl/tests/test_trace.py
git commit -m "feat(data): thinking-trace per-view parser"
```

---

### Task 6: Answer / reward-target extraction

**Files:**
- Create: `echo_rl/data/answers.py`, `echo_rl/tests/test_answers.py`

**Interfaces:**
- Produces:
  - `last_answer(messages_field) -> str` → accepts a JSON **string** or a list; returns the last `assistant` content, stripped.
  - `parse_yes_no(answer:str) -> str | None` → `"yes"`/`"no"`/`None` from the leading token.
  - `finding_set(list_answer:str) -> set[str]` → lowercased finding names from `- ` bullet lines; `set()` if "no ... abnormalities".
  - `is_abnormal(question_type:str, answer:str) -> bool` → heuristic used for RL balancing: `parse_yes_no=="yes"` for classification; non-empty `finding_set` for `abnormality_list`; presence of any non-normal keyword otherwise.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_answers.py`

```python
import json
from echo_rl.data.answers import last_answer, parse_yes_no, finding_set, is_abnormal


def test_last_answer_from_string():
    msgs = json.dumps([{"role": "user", "content": "q"},
                       {"role": "assistant", "content": " Normal. "}])
    assert last_answer(msgs) == "Normal."


def test_parse_yes_no():
    assert parse_yes_no("No, there is no evidence of RAE.") == "no"
    assert parse_yes_no("Yes, present.") == "yes"
    assert parse_yes_no("Normal dimensions.") is None


def test_finding_set():
    ans = "The following abnormalities are identified:\n- Tricuspid Regurgitation\n- Mitral Valve Prolapse"
    assert finding_set(ans) == {"tricuspid regurgitation", "mitral valve prolapse"}
    assert finding_set("No significant abnormalities are identified.") == set()


def test_is_abnormal():
    assert is_abnormal("abnormality_classification", "Yes, present.") is True
    assert is_abnormal("abnormality_classification", "No, none.") is False
    assert is_abnormal("abnormality_list", "- Tricuspid Regurgitation") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_answers.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/answers.py`

```python
import json
import re

_NORMAL_HINT = re.compile(r"\bno (significant )?abnormalit", re.I)


def last_answer(messages_field) -> str:
    msgs = json.loads(messages_field) if isinstance(messages_field, str) else messages_field
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m["content"].strip()
    return ""


def parse_yes_no(answer: str):
    tok = answer.strip().lower().lstrip(".,! ")[:4]
    if tok.startswith("yes"):
        return "yes"
    if tok.startswith("no"):
        return "no"
    return None


def finding_set(list_answer: str) -> set:
    if _NORMAL_HINT.search(list_answer):
        return set()
    out = set()
    for line in list_answer.splitlines():
        line = line.strip()
        if line.startswith("-"):
            name = line.lstrip("- ").strip().lower()
            if name:
                out.add(name)
    return out


def is_abnormal(question_type: str, answer: str) -> bool:
    if question_type == "abnormality_classification":
        return parse_yes_no(answer) == "yes"
    if question_type == "abnormality_list":
        return len(finding_set(answer)) > 0
    a = answer.lower()
    if _NORMAL_HINT.search(a):
        return False
    return bool(re.search(r"(dilat|reduced|abnormal|severe|moderate|regurgitat|stenos|hypertroph|impaired|akinet|hypokinet)", a))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_answers.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/answers.py echo_rl/tests/test_answers.py
git commit -m "feat(data): answer parsing + abnormality heuristic for balancing"
```

---

### Task 7: Gold-CSV loader

**Files:**
- Create: `echo_rl/data/gold.py`, `echo_rl/tests/test_gold.py`

**Interfaces:**
- Produces:
  - `load_metric(csv_path:str) -> dict[str, dict]` → `{study_id: {"label": <str>, "designation": <str>}}` using columns `study_id`, `label`, `study designation`.
  - `load_all(gold_dir:str) -> dict[str, dict[str, str]]` → `{study_id: {<metric_name>: label, "designation": <str>}}` where `<metric_name>` is the CSV filename stem; `designation` taken from the first file that has one.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_gold.py`

```python
from echo_rl.data.gold import load_metric, load_all


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("study_id,label,study designation,from_measurement,from_text,text,text_fields\n" + rows)
    return str(p)


def test_load_metric(tmp_path):
    p = _write(tmp_path, "ejection_fraction_regression.csv",
               "st-1,51.0,TRAIN,True,True,txt,conclusions\nst-2,60.0,TEST,True,False,,\n")
    d = load_metric(p)
    assert d["st-1"]["label"] == "51.0"
    assert d["st-2"]["designation"] == "TEST"


def test_load_all(tmp_path):
    _write(tmp_path, "ejection_fraction_regression.csv", "st-1,51.0,TRAIN,True,True,t,c\n")
    _write(tmp_path, "heart_failure_classification.csv", "st-1,1,TRAIN,True,True,t,c\n")
    alld = load_all(str(tmp_path))
    assert alld["st-1"]["ejection_fraction_regression"] == "51.0"
    assert alld["st-1"]["heart_failure_classification"] == "1"
    assert alld["st-1"]["designation"] == "TRAIN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_gold.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/gold.py`

```python
import csv
import os


def load_metric(csv_path: str) -> dict:
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("study_id")
            if not sid:
                continue
            out[sid] = {"label": row.get("label", ""),
                        "designation": row.get("study designation", "")}
    return out


def load_all(gold_dir: str) -> dict:
    merged: dict = {}
    for fname in sorted(os.listdir(gold_dir)):
        if not fname.endswith(".csv"):
            continue
        metric = os.path.splitext(fname)[0]
        for sid, rec in load_metric(os.path.join(gold_dir, fname)).items():
            slot = merged.setdefault(sid, {})
            slot[metric] = rec["label"]
            if rec["designation"] and "designation" not in slot:
                slot["designation"] = rec["designation"]
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_gold.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/gold.py echo_rl/tests/test_gold.py
git commit -m "feat(data): gold-CSV loader"
```

---

### Task 8: Study-level split assignment

**Files:**
- Create: `echo_rl/data/split.py`, `echo_rl/tests/test_split.py`

**Interfaces:**
- Consumes: gold `designation` map from Task 7.
- Produces:
  - `assign_split(study_uuid:str, designation:str|None, test_studies:set[str]) -> str` → returns `"test"` if in `test_studies` OR `designation.upper()=="TEST"`; `"val"` if `designation.upper()=="VAL"`; `"train"` if `designation.upper()=="TRAIN"`; otherwise a **deterministic hash bucket** (95% train / 5% val, no test) via `hashlib.md5(study_uuid)`.
  - `load_test_studies(vqa_test_path:str) -> set[str]` → the set of `study_uuid` present in `test_vqa.jsonl`.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_split.py`

```python
import json
from echo_rl.data.split import assign_split, load_test_studies


def test_assign_split_explicit():
    assert assign_split("st-x", "TEST", set()) == "test"
    assert assign_split("st-x", "VAL", set()) == "val"
    assert assign_split("st-x", "TRAIN", set()) == "train"


def test_assign_split_test_set_wins():
    assert assign_split("st-x", "TRAIN", {"st-x"}) == "test"


def test_assign_split_hash_deterministic():
    a = assign_split("st-abc", None, set())
    b = assign_split("st-abc", None, set())
    assert a == b and a in {"train", "val"}


def test_load_test_studies(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"study_uuid": "st-1"}) + "\n" + json.dumps({"study_uuid": "st-1"}) + "\n"
                 + json.dumps({"study_uuid": "st-2"}) + "\n")
    assert load_test_studies(str(p)) == {"st-1", "st-2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_split.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/split.py`

```python
import hashlib
import json


def load_test_studies(vqa_test_path: str) -> set:
    out = set()
    with open(vqa_test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["study_uuid"])
    return out


def assign_split(study_uuid: str, designation, test_studies: set) -> str:
    if study_uuid in test_studies:
        return "test"
    d = (designation or "").upper()
    if d == "TEST":
        return "test"
    if d == "VAL":
        return "val"
    if d == "TRAIN":
        return "train"
    h = int(hashlib.md5(study_uuid.encode()).hexdigest(), 16) % 100
    return "val" if h < 5 else "train"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_split.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/split.py echo_rl/tests/test_split.py
git commit -m "feat(data): study-level split assignment"
```

---

### Task 9: Synthetic agentic trajectory builder (SFT)

**Files:**
- Create: `echo_rl/data/trajectory.py`, `echo_rl/tests/test_trajectory.py`

**Interfaces:**
- Consumes: `Clip` (Task 3), `parse_thinking`/`findings_text` (Task 5), `canonical_view` (Task 2), `evenly_spaced`/`midframe` (Task 4), `Config` params.
- Produces:
  - `overview_turn(clips:list[Clip], cfg) -> dict` → `{"type":"overview", "views":[{"view":..., "frame":<abs path>, "frame_count":int}, ...]}` using `midframe` of each clip, capped at `cfg.n_overview_views`.
  - `build_trajectory(question:str, answer:str, thinking:str, clips:list[Clip], cfg, max_views:int=4) -> dict` → an ordered, framework-agnostic trajectory:
    ```
    {"overview": <overview_turn>,
     "turns": [ {"type":"tool","name":"select_view","args":{"view":V},
                 "obs":{"view":V,"frames":[abs paths...]}},
                {"type":"think","text": <trimmed findings for V>}, ... ],
     "answer": <answer>}
    ```
    Only trace views that **map to an available clip** (by `canonical_view` on the clip's `view`, matching the trace view with a trailing " View" stripped) are emitted, in trace order, capped at `max_views`. Frames per view = `evenly_spaced(frame_count, cfg.n_preview_frames)`. If no trace views map, `turns == []` (answer-only trajectory — still valid for SFT format).

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_trajectory.py`

```python
from echo_rl.config import Config
from echo_rl.data.studies import Clip
from echo_rl.data.trajectory import overview_turn, build_trajectory


def _clips():
    return [
        Clip("di-1", "PLAX Standard", "/d/di-1_PLAX Standard", [f"{i}.png" for i in range(10)]),
        Clip("di-2", "A4C", "/d/di-2_A4C", [f"{i}.png" for i in range(20)]),
    ]


def _thinking():
    return ("#### 1. **PLAX Standard View**\n- **Clinical Findings:**\n  - LV normal.\n"
            "#### 2. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n"
            "#### 3. **Subcostal View**\n- **Clinical Findings:**\n  - IVC normal.\n")


def test_overview_turn():
    cfg = Config.from_env()
    ov = overview_turn(_clips(), cfg)
    assert len(ov["views"]) == 2
    assert ov["views"][0]["frame"].endswith("/5.png")   # midframe of 10
    assert ov["views"][1]["frame_count"] == 20


def test_build_trajectory_maps_available_views():
    cfg = Config.from_env()
    traj = build_trajectory("Describe LV.", "LV normal.", _thinking(), _clips(), cfg)
    tool_views = [t["args"]["view"] for t in traj["turns"] if t["type"] == "tool"]
    assert tool_views == ["PLAX Standard", "A4C"]      # Subcostal has no clip -> skipped
    assert traj["answer"] == "LV normal."
    first_obs = next(t for t in traj["turns"] if t["type"] == "tool")["obs"]
    assert len(first_obs["frames"]) == cfg.n_preview_frames
    assert any(t["type"] == "think" and "LV normal" in t["text"] for t in traj["turns"])


def test_build_trajectory_no_map():
    cfg = Config.from_env()
    traj = build_trajectory("q", "a", "#### 1. **Suprasternal View**\n- **Clinical Findings:**\n  - x\n",
                            _clips(), cfg)
    assert traj["turns"] == [] and traj["answer"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_trajectory.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/trajectory.py`

```python
from echo_rl.data.trace import parse_thinking, findings_text
from echo_rl.data.views import canonical_view
from echo_rl.data.frames import evenly_spaced, midframe


def _trace_view_key(trace_view: str) -> str:
    v = trace_view.strip()
    if v.endswith(" View"):
        v = v[: -len(" View")]
    return canonical_view(v)


def overview_turn(clips: list, cfg) -> dict:
    views = []
    for c in clips[: cfg.n_overview_views]:
        views.append({"view": c.view,
                      "frame": c.frame_path(midframe(c.frame_count)),
                      "frame_count": c.frame_count})
    return {"type": "overview", "views": views}


def build_trajectory(question: str, answer: str, thinking: str, clips: list, cfg,
                     max_views: int = 4) -> dict:
    by_view = {canonical_view(c.view): c for c in clips}
    turns = []
    used = 0
    for trace_view, body in parse_thinking(thinking):
        key = _trace_view_key(trace_view)
        clip = by_view.get(key)
        if clip is None:
            continue
        idxs = evenly_spaced(clip.frame_count, cfg.n_preview_frames)
        turns.append({"type": "tool", "name": "select_view",
                      "args": {"view": clip.view},
                      "obs": {"view": clip.view,
                              "frames": [clip.frame_path(i) for i in idxs]}})
        turns.append({"type": "think", "text": findings_text(body)})
        used += 1
        if used >= max_views:
            break
    return {"overview": overview_turn(clips, cfg), "turns": turns, "answer": answer}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_trajectory.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/trajectory.py echo_rl/tests/test_trajectory.py
git commit -m "feat(data): synthetic agentic trajectory builder"
```

---

### Task 10: Balanced RL pool

**Files:**
- Create: `echo_rl/data/balance.py`, `echo_rl/tests/test_balance.py`

**Interfaces:**
- Consumes: nothing from other tasks (operates on precomputed labels).
- Produces:
  - `class_weights(labels:list[str]) -> dict[str,float]` → inverse-frequency weights normalized so the **mean weight over the label list is 1.0**.
  - `resample_indices(labels:list[str], n:int, seed:int) -> list[int]` → `n` indices sampled **with replacement** using `class_weights`, deterministic given `seed`; yields an approximately class-balanced multiset.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_balance.py`

```python
from echo_rl.data.balance import class_weights, resample_indices


def test_class_weights_inverse_freq():
    w = class_weights(["no"] * 77 + ["yes"] * 23)
    assert w["yes"] > w["no"]


def test_resample_balances():
    labels = ["no"] * 90 + ["yes"] * 10
    idx = resample_indices(labels, 1000, seed=0)
    picked = [labels[i] for i in idx]
    frac_yes = picked.count("yes") / len(picked)
    assert 0.4 < frac_yes < 0.6          # balanced, not the 10% prior


def test_resample_deterministic():
    labels = ["a", "b", "c"] * 10
    assert resample_indices(labels, 50, seed=1) == resample_indices(labels, 50, seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_balance.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/balance.py`

```python
import random
from collections import Counter


def class_weights(labels: list) -> dict:
    counts = Counter(labels)
    n = len(labels)
    k = len(counts)
    # inverse frequency, normalized so mean weight over the list == 1.0
    raw = {c: n / (k * cnt) for c, cnt in counts.items()}
    return raw


def resample_indices(labels: list, n: int, seed: int) -> list:
    w = class_weights(labels)
    weights = [w[lab] for lab in labels]
    rng = random.Random(seed)
    return rng.choices(range(len(labels)), weights=weights, k=n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_balance.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/balance.py echo_rl/tests/test_balance.py
git commit -m "feat(data): class-balanced RL resampling"
```

---

### Task 11: SFT + RL record builders (streaming)

**Files:**
- Create: `echo_rl/data/builders.py`, `echo_rl/tests/test_builders.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `sft_record(rec:dict, clips:list[Clip], cfg) -> dict` → `{"study_uuid","question_type","question","trajectory"}` where `question` is the user turn and `trajectory` is Task 9 output.
  - `rl_record(rec:dict, clips:list[Clip], cfg, gold:dict) -> dict` → `{"study_uuid","question_type","question","answer","reward_key":{...},"overview","designation"}` where `reward_key` holds `{"kind": <"yesno"|"set"|"text">, "target": <answer or set list>, "gold": <matching gold values or {}>, "is_abnormal": bool}`.
  - `iter_jsonl(path:str) -> Iterator[dict]`.

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_builders.py`

```python
import json
from echo_rl.config import Config
from echo_rl.data.studies import Clip
from echo_rl.data.builders import sft_record, rl_record


def _rec(qtype, answer):
    return {"study_uuid": "st-1", "question_type": qtype, "thinking": "#### 1. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n",
            "messages": json.dumps([{"role": "user", "content": "Q?"},
                                    {"role": "assistant", "content": answer}])}


def _clips():
    return [Clip("di-2", "A4C", "/d/di-2_A4C", [f"{i}.png" for i in range(20)])]


def test_sft_record():
    cfg = Config.from_env()
    out = sft_record(_rec("structure_description", "RV normal."), _clips(), cfg)
    assert out["question"] == "Q?"
    assert out["trajectory"]["answer"] == "RV normal."


def test_rl_record_yesno_and_gold():
    cfg = Config.from_env()
    gold = {"st-1": {"ejection_fraction_regression": "51.0", "designation": "TRAIN"}}
    out = rl_record(_rec("abnormality_classification", "No, none."), _clips(), cfg, gold)
    assert out["reward_key"]["kind"] == "yesno"
    assert out["reward_key"]["is_abnormal"] is False
    assert out["reward_key"]["gold"]["ejection_fraction_regression"] == "51.0"
    assert out["designation"] == "TRAIN"
    assert len(out["overview"]["views"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_builders.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/data/builders.py`

```python
import json
from echo_rl.data.answers import last_answer, parse_yes_no, finding_set, is_abnormal
from echo_rl.data.trajectory import build_trajectory, overview_turn


def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _question(rec: dict) -> str:
    msgs = json.loads(rec["messages"]) if isinstance(rec["messages"], str) else rec["messages"]
    for m in msgs:
        if m.get("role") == "user":
            return m["content"].strip()
    return ""


def sft_record(rec: dict, clips: list, cfg) -> dict:
    answer = last_answer(rec["messages"])
    traj = build_trajectory(_question(rec), answer, rec.get("thinking", ""), clips, cfg)
    return {"study_uuid": rec["study_uuid"], "question_type": rec["question_type"],
            "question": _question(rec), "trajectory": traj}


def _reward_key(qtype: str, answer: str, gold_for_study: dict) -> dict:
    if qtype == "abnormality_classification":
        kind, target = "yesno", parse_yes_no(answer)
    elif qtype == "abnormality_list":
        kind, target = "set", sorted(finding_set(answer))
    else:
        kind, target = "text", answer
    gold = {k: v for k, v in (gold_for_study or {}).items() if k != "designation"}
    return {"kind": kind, "target": target, "gold": gold,
            "is_abnormal": is_abnormal(qtype, answer)}


def rl_record(rec: dict, clips: list, cfg, gold: dict) -> dict:
    answer = last_answer(rec["messages"])
    g = gold.get(rec["study_uuid"], {})
    return {"study_uuid": rec["study_uuid"], "question_type": rec["question_type"],
            "question": _question(rec), "answer": answer,
            "reward_key": _reward_key(rec["question_type"], answer, g),
            "overview": overview_turn(clips, cfg),
            "designation": g.get("designation", "")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_builders.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add echo_rl/data/builders.py echo_rl/tests/test_builders.py
git commit -m "feat(data): SFT and RL record builders"
```

---

### Task 12: CLI + end-to-end run on real data

**Files:**
- Create: `echo_rl/cli.py`, `echo_rl/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above; caches per-study `index_study` results to avoid re-listing folders.
- Produces a CLI with subcommands writing to `cfg.out_dir`:
  - `build-sft`  → `build/sft.jsonl` (one line per QA whose study joins).
  - `build-rl`   → `build/rl.jsonl` + `build/rl_pool.json` (balanced index list over abnormal/normal label) + split tag per record.
  - `stats`      → prints join rate, per-`question_type` counts, yes/no balance, and split sizes.
  - Flags: `--limit N` (process first N QA records, for smoke runs), `--split {train,val,test,all}` (default `all`).

- [ ] **Step 1: Write the failing test** — `echo_rl/tests/test_cli.py`

```python
import json, os
from echo_rl.cli import run


def _setup(tmp_path):
    # fake preprocessed_data with one study + one A4C clip
    pp = tmp_path / "pp"; clip = pp / "st-1" / "di-2_A4C"; clip.mkdir(parents=True)
    for i in range(6):
        (clip / f"{i}.png").write_bytes(b"x")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({
        "study_uuid": "st-1", "question_type": "abnormality_classification",
        "thinking": "#### 1. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n",
        "messages": json.dumps([{"role": "user", "content": "Q?"},
                                {"role": "assistant", "content": "No, none."}])}) + "\n")
    test = tmp_path / "test.jsonl"; test.write_text("")
    gold = tmp_path / "gold"; gold.mkdir()
    (gold / "heart_failure_classification.csv").write_text(
        "study_id,label,study designation,from_measurement,from_text,text,text_fields\nst-1,0,TRAIN,T,T,t,c\n")
    out = tmp_path / "out"
    env = {"ECHO_PREPROCESSED_DIR": str(pp), "ECHO_VQA_TRAIN": str(train),
           "ECHO_VQA_TEST": str(test), "ECHO_GOLD_DIR": str(gold), "ECHO_OUT_DIR": str(out)}
    return env, out


def test_build_sft_and_rl(tmp_path, monkeypatch):
    env, out = _setup(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    run(["build-sft"])
    run(["build-rl"])
    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 1 and sft[0]["trajectory"]["turns"][0]["args"]["view"] == "A4C"
    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert rl[0]["reward_key"]["kind"] == "yesno"
    pool = json.load(open(out / "rl_pool.json"))
    assert isinstance(pool["indices"], list) and len(pool["indices"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation** — `echo_rl/cli.py`

```python
import argparse
import json
import os
from collections import Counter
from echo_rl.config import Config
from echo_rl.data.studies import study_dir, index_study
from echo_rl.data.builders import iter_jsonl, sft_record, rl_record
from echo_rl.data.gold import load_all
from echo_rl.data.split import assign_split, load_test_studies
from echo_rl.data.balance import resample_indices


def _clip_cache(cfg):
    cache = {}
    def get(uuid):
        if uuid not in cache:
            try:
                cache[uuid] = index_study(study_dir(cfg.preprocessed_dir, uuid))
            except FileNotFoundError:
                cache[uuid] = None
        return cache[uuid]
    return get


def _out(cfg, name):
    os.makedirs(cfg.out_dir, exist_ok=True)
    return os.path.join(cfg.out_dir, name)


def build_sft(cfg, limit):
    get = _clip_cache(cfg)
    n = 0
    with open(_out(cfg, "sft.jsonl"), "w") as w:
        for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
            if limit and i >= limit:
                break
            clips = get(rec["study_uuid"])
            if not clips:
                continue
            w.write(json.dumps(sft_record(rec, clips, cfg)) + "\n")
            n += 1
    print(f"[build-sft] wrote {n} records")


def build_rl(cfg, limit):
    get = _clip_cache(cfg)
    gold = load_all(cfg.gold_dir) if os.path.isdir(cfg.gold_dir) else {}
    tests = load_test_studies(cfg.vqa_test) if os.path.exists(cfg.vqa_test) else set()
    records, labels = [], []
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        clips = get(rec["study_uuid"])
        if not clips:
            continue
        out = rl_record(rec, clips, cfg, gold)
        out["split"] = assign_split(rec["study_uuid"], gold.get(rec["study_uuid"], {}).get("designation"), tests)
        records.append(out)
        labels.append("abn" if out["reward_key"]["is_abnormal"] else "norm")
    with open(_out(cfg, "rl.jsonl"), "w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]
    train_labels = [labels[i] for i in train_idx]
    pool = resample_indices(train_labels, len(train_labels), cfg.seed) if train_labels else []
    pool = [train_idx[j] for j in pool]
    json.dump({"indices": pool}, open(_out(cfg, "rl_pool.json"), "w"))
    print(f"[build-rl] wrote {len(records)} records; balanced pool {len(pool)}")


def stats(cfg, limit):
    get = _clip_cache(cfg)
    joined = total = 0
    qt = Counter(); yn = Counter()
    from echo_rl.data.answers import last_answer, parse_yes_no
    for i, rec in enumerate(iter_jsonl(cfg.vqa_train)):
        if limit and i >= limit:
            break
        total += 1
        if get(rec["study_uuid"]):
            joined += 1
        qt[rec["question_type"]] += 1
        if rec["question_type"] == "abnormality_classification":
            yn[parse_yes_no(last_answer(rec["messages"]))] += 1
    print(f"[stats] join {joined}/{total}; types {dict(qt)}; yes/no {dict(yn)}")


def run(argv=None):
    p = argparse.ArgumentParser(prog="echo_rl")
    p.add_argument("cmd", choices=["build-sft", "build-rl", "stats"])
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    cfg = Config.from_env()
    {"build-sft": build_sft, "build-rl": build_rl, "stats": stats}[args.cmd](cfg, args.limit)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the FULL suite + a real smoke run**

```bash
.venv/bin/pytest echo_rl -v
.venv/bin/python -m echo_rl.cli stats --limit 2000
.venv/bin/python -m echo_rl.cli build-sft --limit 200
.venv/bin/python -m echo_rl.cli build-rl --limit 2000
head -1 build/sft.jsonl
head -1 build/rl.jsonl
```
Expected: all tests pass; `stats` shows join ≈ 2000/2000 and yes/no ≈ 77/23; `build/sft.jsonl` line has a `trajectory.turns` array with `select_view` tool calls whose `obs.frames` are real `.png` paths; `build/rl.jsonl` line has a populated `reward_key` and non-empty `overview.views`.

- [ ] **Step 6: Commit**

```bash
git add echo_rl/cli.py echo_rl/tests/test_cli.py
git commit -m "feat(data): CLI (build-sft/build-rl/stats) + end-to-end smoke run"
```

---

## Self-Review

**Spec coverage (spec §2, §3, §4, §5.1, §6):**
- Join `study_uuid`→clips/frames → Tasks 3, 4 ✓
- View labels from folder names → Task 2 ✓
- Per-view `thinking` parse → Task 5 ✓
- Synthetic agentic trajectory (trace breakdown, §5.1) → Task 9 ✓
- Reward targets incl. gold-CSV (§5.2 target extraction; reward *scoring* is Phase 4) → Tasks 6, 7, 11 ✓
- Study-level splits (§6) → Task 8 ✓
- Class-balanced / abnormal-enriched RL pool (§6, non-negotiable) → Tasks 6, 10, 12 ✓
- Overview / initial observation spec (§4) → Task 9 (`overview_turn`) ✓
- **Deferred to later phases (correctly out of scope here):** actual PNG loading/decoding & tool execution (Phase 2 env); reward *scoring* functions & LLM-judge (Phase 4); Qwen chat-template serialization of trajectories (Phase 2/3 consumer). Noted so no reviewer expects them here.

**Placeholder scan:** none — every step has runnable code/commands. ✓

**Type consistency:** `Clip` fields (`di_id, view, path, frame_files`) consistent across Tasks 3, 9, 11, 12; `build_trajectory`/`overview_turn` signatures consistent between Task 9 and Task 11; `reward_key` shape consistent between Task 11 def and Task 12 consumer. ✓

**Known simplification (intentional):** `select_frames` / `zoom` are **runtime** tools (Phase 2 env) — Phase 1 only emits `select_view` in SFT trajectories (§5.1 shows `select_view` warm-start; the other tools are meant to emerge in RL). This matches the spec and is not a gap.
