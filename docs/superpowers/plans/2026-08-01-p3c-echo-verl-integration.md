# P3c — Echo↔VeRL Integration (hybrid native) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the echo environment into upstream VeRL v0.7.1 as a native tool + reward + data-gen, using the HYBRID frame path (initial obs = dataset video; tool obs = images). No patch to verl's agent loop.

**Architecture:** A new `echo_verl/` package. A verl-free **pure core** (`session.py`) drives the three echo ops over the existing `echo_env` tools and is fully unit-tested with the PIL fixtures. A thin **`EchoTool(BaseTool)`** adapter wraps the core into verl's native tool contract, returning `ToolResponse(image=[...])`. A **reward** `compute_score` delegates to `echo_rl.reward.score.total_reward`. A **data-gen** builds verl parquet rows. Tool + training **configs** are net-new YAML.

**Tech Stack:** Python 3.13; `echo_verl` may import `echo_env` (PIL) and `echo_rl` (stdlib) but the verl-dependent files import `verl.*` (NOT installable in `.venv`). Tests: `.venv/bin/pytest`.

## Global Constraints

- **Verification tiers (label every deliverable):**
  - 🟢 = real passing unit tests in `.venv` (no `verl` import). Genuinely done.
  - 🟡 = "applies/compiles + contract matches verl-071 + reviewer-confirmed; UNRUN." The GPU run is the true gate. Files importing `verl.*` are 🟡 — verify with `.venv/bin/python -m py_compile <file>` (syntax) + reviewer cross-check against the fetched tree; do NOT attempt to import them in `.venv`.
- **Fetched reference tree (read-only, for authoring/review):** `/tmp/claude-2032/-vast-users-mohammad-yaqub-project-EchoSonarVideo/5cd7720e-76db-4161-8ab2-d9f2a25ddd78/scratchpad/verl-071`.
- **BaseTool contract (verbatim, `verl/tools/base_tool.py`):** `__init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema)` (sets `self.name = tool_schema.function.name`); `async create(self, instance_id=None, **kwargs) -> tuple[str, ToolResponse]` (operand in `kwargs["create_kwargs"]`); `@rollout_trace_op async execute(self, instance_id, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]`; `async calc_reward(...)` (NOT called by ToolAgentLoop); `async release(self, instance_id, **kwargs) -> None`. `ToolResponse(text=None, image=None, video=None)` — image/video MUST be lists.
- **Tool name = `echo`** (matches the P3a SFT serializer's `{"name":"echo",...}`). Hermes tool-call format.
- **Reward interface (verbatim):** `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float | dict`. Wired via verl `custom_reward_function` config (no upstream edit).
- **Reuse, don't reimplement:** `echo_env.{manifest.build_manifest, tools.{select_view,select_frames,zoom}, frames.PILFrameLoader, observation.Observation, config.EnvConfig}`; `echo_rl.reward.score.total_reward`; `echo_rl.data.builders.rl_record`.
- Commit after each task with a `feat(p3c):` (or `chore(p3c):` for config) message.

---

### Task 1: EchoSession pure core (🟢)

**Files:**
- Create: `echo_verl/__init__.py` (empty)
- Create: `echo_verl/session.py`
- Test: `echo_verl/tests/__init__.py` (empty), `echo_verl/tests/test_session.py`

**Interfaces:**
- Consumes: `echo_env.config.EnvConfig`, `echo_env.manifest.build_manifest`, `echo_env.frames.PILFrameLoader`, `echo_env.observation.Observation`, `echo_env.tools.{select_view,select_frames,zoom}`.
- Produces: `class EchoSession` — `__init__(self, cfg, study_uuid, loader=None)` (builds the manifest for the study); `run(self, op: str, params: dict) -> Observation` (dispatches one op; unknown op → `Observation.failure`). This is termination-agnostic (no budget/step loop — ToolAgentLoop owns turns).

- [ ] **Step 1: Write the failing test** — reuse the echo_env PIL fixture. Add to `echo_verl/tests/test_session.py`:

```python
import pytest
from echo_env.config import EnvConfig
from echo_verl.session import EchoSession

# Reuse echo_env's study_fixture (PIL PNG study). Pull it in via a local conftest import path.
pytest_plugins = ["echo_env.tests.conftest"]


def _cfg(preprocessed_dir):
    return EnvConfig(preprocessed_dir=preprocessed_dir, min_crop_side=32,
                     highres_max_side=320, preview_max_side=160)


def test_select_view_returns_frames(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_view", {"view_name": "A4C"})
    assert obs.ok
    assert obs.n_frames > 0


def test_select_frames_op(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_frames", {"view_name": "A4C", "indices": [0, 2]})
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [0, 2]


def test_zoom_op(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("zoom", {"view_name": "A4C", "bbox": [40, 40, 200, 240], "frame_indices": [1]})
    assert obs.ok
    assert obs.frames[0].kind == "crop"


def test_unknown_op_fails_cleanly(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("teleport", {"view_name": "A4C"})
    assert not obs.ok
    assert "unknown op" in obs.error


def test_missing_view_name_does_not_raise(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_view", {})           # no view_name
    assert not obs.ok                         # failure Observation, never an exception
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest echo_verl/tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_verl.session'`

- [ ] **Step 3: Implement**

```python
# echo_verl/session.py
"""Per-trajectory echo session — the verl-free core behind EchoTool.

Mirrors echo_env.EchoEnv._dispatch's op→tool mapping, but standalone and
termination-agnostic: VeRL's ToolAgentLoop owns turns/termination, so this
core just dispatches one op and returns an Observation. Kept import-clean of
verl so it is fully unit-testable offline.
"""
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.observation import Observation
from echo_env.tools import select_view, select_frames, zoom


class EchoSession:
    def __init__(self, cfg, study_uuid, loader=None):
        self.cfg = cfg
        self.loader = loader or PILFrameLoader()
        self.manifest = build_manifest(cfg.preprocessed_dir, study_uuid)

    def run(self, op: str, params: dict) -> Observation:
        params = params or {}
        if op == "select_view":
            return select_view(self.manifest, self.loader, self.cfg, params.get("view_name"))
        if op == "select_frames":
            return select_frames(self.manifest, self.loader, self.cfg,
                                 params.get("view_name"), params.get("indices", []))
        if op == "zoom":
            return zoom(self.manifest, self.loader, self.cfg, params.get("view_name"),
                        params.get("bbox"), params.get("frame_indices", []))
        return Observation.failure(op or "echo", f"unknown op {op!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest echo_verl/tests/test_session.py -v`
Expected: PASS (5 tests). If the `pytest_plugins`/conftest reuse errors, instead copy the fixture pattern from `echo_env/tests/conftest.py` into `echo_verl/tests/conftest.py` (a small PNG study) and drop the `pytest_plugins` line — the behavior asserted is identical.

- [ ] **Step 5: Commit**

```bash
git add echo_verl/__init__.py echo_verl/session.py echo_verl/tests/
git commit -m "feat(p3c): EchoSession pure core (per-trajectory op dispatch over echo_env)"
```

---

### Task 2: EchoTool BaseTool adapter (🟡 UNRUN)

**Files:**
- Create: `echo_verl/echo_tool.py`
- Test: none runnable in `.venv` (imports `verl`). Verification = `py_compile` + reviewer contract-check.

**Interfaces:**
- Consumes: `verl.tools.base_tool.BaseTool`, `verl.tools.schemas.{OpenAIFunctionToolSchema, ToolResponse}`, `echo_verl.session.EchoSession`, `echo_env.config.EnvConfig`.
- Produces: `class EchoTool(BaseTool)` implementing the BaseTool contract; tool `self.name == "echo"`; `execute` returns `ToolResponse(image=[<PIL frames>], text=...)`.

- [ ] **Step 1: Implement** (no failing-test step — verl is not importable here; this is 🟡)

```python
# echo_verl/echo_tool.py
"""Native VeRL v0.7.1 tool adapter for the echo environment (HYBRID frame path).

UNRUN in this repo: imports verl.* which need the torch/vLLM runtime. Verified
here only by py_compile + contract review against the fetched verl-071 tree
(BaseTool/ToolResponse signatures in echo_env/INTEGRATION.md §0.2). GPU run = gate.

Tool observations return as IMAGES (ToolResponse(image=[...])) because v0.7.1's
ToolAgentLoop raises NotImplementedError on tool-returned video
(tool_agent_loop.py:335-340). The initial full-clip video reaches the model via
the dataset `videos` column, not this tool.
"""
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from echo_env.config import EnvConfig
from echo_verl.session import EchoSession

_ERR_REWARD = -0.05   # small per-step shaping penalty on a failed tool call


class EchoTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)   # sets self.name = tool_schema.function.name
        self._instances: dict[str, EchoSession] = {}
        # EnvConfig knobs may be overridden via the tool `config` block; fall back to env.
        self._cfg = EnvConfig.from_env()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        study_uuid = create_kwargs.get("study_uuid")
        self._instances[instance_id] = EchoSession(self._cfg, study_uuid)
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any],
                      **kwargs) -> tuple[ToolResponse, float, dict]:
        session = self._instances.get(instance_id)
        if session is None:
            return ToolResponse(text="echo session not initialized"), _ERR_REWARD, {"success": False}
        op = parameters.get("op")
        obs = session.run(op, parameters)
        if not obs.ok:
            return ToolResponse(text=obs.error), _ERR_REWARD, {"success": False}
        images = [f.image for f in obs.frames]
        return ToolResponse(image=images, text=obs.text), 0.0, {"success": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
```

- [ ] **Step 2: Syntax-check**

Run: `.venv/bin/python -m py_compile echo_verl/echo_tool.py && echo OK`
Expected: `OK` (no verl import happens at compile time).

- [ ] **Step 3: Commit**

```bash
git add echo_verl/echo_tool.py
git commit -m "feat(p3c): EchoTool BaseTool adapter (image ToolResponse, hybrid) [UNRUN]"
```

---

### Task 3: Reward compute_score wrapper (🟢)

**Files:**
- Create: `echo_verl/reward.py`
- Test: `echo_verl/tests/test_reward.py`

**Interfaces:**
- Consumes: `echo_rl.reward.score.total_reward`.
- Produces: `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float`. `ground_truth` IS the `reward_key` dict (the data-gen writes it there). Counts tool calls in `solution_str` for the annealed bonus; reads `tool_bonus_coef` from `extra_info` (default 0.0).

- [ ] **Step 1: Write the failing test**

```python
# echo_verl/tests/test_reward.py
from echo_verl.reward import compute_score, _count_tool_calls


def test_count_tool_calls():
    s = "<tool_call>{}</tool_call> ... <tool_call>{}</tool_call>"
    assert _count_tool_calls(s) == 2
    assert _count_tool_calls("no calls") == 0


def test_compute_score_yesno():
    rk = {"kind": "yesno", "target": "no", "gold": {}}
    sol = "<think>normal</think>\n<answer>No.</answer>"
    r = compute_score("echo", sol, rk)
    assert r == 1.0 * 1.0 + 0.2 * 1.0 + 0.0   # outcome + format, no tool bonus (no tool_call)


def test_compute_score_applies_annealed_bonus():
    rk = {"kind": "yesno", "target": "yes", "gold": {}}
    sol = "<think>t</think>\n<tool_call>{}</tool_call>\n<answer>Yes.</answer>"
    r = compute_score("echo", sol, rk, extra_info={"tool_bonus_coef": 0.1})
    assert abs(r - (1.0 + 0.2 + 0.1)) < 1e-9


def test_compute_score_none_ground_truth_safe():
    # defensive: missing reward_key must not crash; scores format only
    r = compute_score("echo", "<think>t</think>\n<answer>x</answer>", None)
    assert r >= 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest echo_verl/tests/test_reward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_verl.reward'`

- [ ] **Step 3: Implement**

```python
# echo_verl/reward.py
"""Echo reward for VeRL's custom_reward_function hook (no upstream edit).

verl calls compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs).
ground_truth carries the P1 reward_key ({kind,target,gold,is_abnormal}); we delegate
outcome+format+annealed-tool-bonus scoring to echo_rl.reward.score.total_reward.
"""
import re
from echo_rl.reward.score import total_reward

_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _count_tool_calls(solution_str: str) -> int:
    return len(_TOOLCALL_RE.findall(solution_str or ""))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    reward_key = ground_truth if isinstance(ground_truth, dict) else {"kind": "text", "target": "", "gold": {}}
    info = extra_info or {}
    result = total_reward(
        reward_key,
        solution_str or "",
        tool_calls=_count_tool_calls(solution_str),
        tool_bonus_coef=float(info.get("tool_bonus_coef", 0.0)),
    )
    return result["reward"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest echo_verl/tests/test_reward.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add echo_verl/reward.py echo_verl/tests/test_reward.py
git commit -m "feat(p3c): reward compute_score wrapper (custom_reward_function -> total_reward)"
```

---

### Task 4: Parquet data-gen — row-shaping (🟢) + writer (🟡)

**Files:**
- Create: `echo_verl/generate_trainset.py`
- Test: `echo_verl/tests/test_trainset.py`

**Interfaces:**
- Consumes: `echo_rl.data.builders.rl_record` output shape (`{study_uuid, question_type, question, answer, reward_key, overview, designation}`).
- Produces: `build_row(rl_rec: dict, video_spec: dict) -> dict` (🟢, the exact verl parquet-row dict per INTEGRATION.md §0.2); `write_parquet(rows: list, path: str) -> int` (🟡, lazy-imports pyarrow so the module imports without it).

- [ ] **Step 1: Write the failing test**

```python
# echo_verl/tests/test_trainset.py
from echo_verl.generate_trainset import build_row


def _rl_rec():
    return {"study_uuid": "S1", "question_type": "abnormality_classification",
            "question": "Any abnormality?", "answer": "No.",
            "reward_key": {"kind": "yesno", "target": "no", "gold": {}, "is_abnormal": False},
            "overview": {}, "designation": ""}


def test_build_row_shape():
    row = build_row(_rl_rec(), {"video": "file:///data/S1/clip.mp4", "fps": 2})
    assert row["data_source"] == "echo"
    assert row["agent_name"] == "tool_agent"                       # selects ToolAgentLoop
    assert isinstance(row["prompt"], list) and row["prompt"][0]["role"] == "user"
    assert "<video>" in row["prompt"][0]["content"]
    assert row["videos"] == [{"video": "file:///data/S1/clip.mp4", "fps": 2}]
    assert row["reward_model"]["ground_truth"] == _rl_rec()["reward_key"]
    ek = row["extra_info"]
    assert ek["need_tools_kwargs"] is True
    assert ek["tools_kwargs"]["echo"]["create_kwargs"]["study_uuid"] == "S1"


def test_build_row_supplies_clip_twice():
    # clip appears both as the initial dataset video AND the tool operand
    row = build_row(_rl_rec(), {"video": "file:///c.mp4"})
    assert row["videos"][0]["video"] == "file:///c.mp4"
    assert row["extra_info"]["tools_kwargs"]["echo"]["create_kwargs"]["study_uuid"] == "S1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest echo_verl/tests/test_trainset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_verl.generate_trainset'`

- [ ] **Step 3: Implement**

```python
# echo_verl/generate_trainset.py
"""Build VeRL v0.7.1 parquet rows for echo RL (hybrid frame path).

Row contract: INTEGRATION.md §0.2. The clip is supplied twice — as the initial
dataset `videos` (true video the model sees up front) and as the tool operand in
extra_info.tools_kwargs.echo.create_kwargs. Row-shaping is unit-tested; the
parquet write lazy-imports pyarrow (UNRUN here — pyarrow lives in the training env).
"""

_DATA_SOURCE = "echo"


def build_row(rl_rec: dict, video_spec: dict) -> dict:
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [{"role": "user", "content": "<video>\n" + rl_rec["question"]}],
        "videos": [video_spec],
        "images": [],
        "reward_model": {"ground_truth": rl_rec["reward_key"], "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {
            "index": rl_rec["study_uuid"],
            "question_type": rl_rec["question_type"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"echo": {"create_kwargs": {"study_uuid": rl_rec["study_uuid"]}}},
        },
    }


def write_parquet(rows: list, path: str) -> int:
    import pyarrow as pa            # lazy: not installed in the offline .venv
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest echo_verl/tests/test_trainset.py -v`
Expected: PASS (2 tests). (`write_parquet` is not exercised — pyarrow absent — that's the 🟡 boundary.)

- [ ] **Step 5: Commit**

```bash
git add echo_verl/generate_trainset.py echo_verl/tests/test_trainset.py
git commit -m "feat(p3c): verl parquet row-shaping (tested) + lazy pyarrow writer [write UNRUN]"
```

---

### Task 5: Tool-config + training config YAML (🟡 UNRUN)

**Files:**
- Create: `echo_verl/configs/echo_tool_config.yaml`
- Create: `echo_verl/configs/README.md` (how these wire into a verl launch; the GRPO/SFT launch flags)

**Interfaces:** none (config). Verification = YAML parses (`.venv/bin/python -c "import yaml; yaml.safe_load(open(...))"`) + reviewer cross-check of keys against `verl-071` (`tool_registry.py` expected structure; `geo3k_tool_config.yaml` reference).

- [ ] **Step 1: Write the tool-config YAML**

```yaml
# echo_verl/configs/echo_tool_config.yaml
# Wired via rollout.multi_turn.tool_config_path. Structure per verl-071
# verl/tools/utils/tool_registry.py + examples/.../geo3k_tool_config.yaml.
# NOTE: pydantic OpenAIFunctionPropertySchema drops non-scalar keys (extra=ignore),
# so only flat scalar props + an `op` enum are advertised; EchoTool.execute reads
# the full arguments dict (view_name/indices/bbox/frame_indices) at runtime.
tools:
  - class_name: "echo_verl.echo_tool.EchoTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "echo"
        description: >-
          Inspect an echocardiography study. op=select_view shows preview frames of a
          named view; op=select_frames returns specific high-res frames; op=zoom crops a
          bbox region on chosen frames. Provide view_name; for select_frames provide
          indices; for zoom provide bbox and frame_indices.
        parameters:
          type: "object"
          properties:
            op:
              type: "string"
              enum: ["select_view", "select_frames", "zoom"]
              description: "Which echo operation to perform."
            view_name:
              type: "string"
              description: "Target view, e.g. 'A4C', 'PLAX Standard'."
          required: ["op"]
```

- [ ] **Step 2: Verify the YAML parses**

Run: `.venv/bin/python -c "import yaml; d=yaml.safe_load(open('echo_verl/configs/echo_tool_config.yaml')); print(d['tools'][0]['class_name'], d['tools'][0]['tool_schema']['function']['name'])"`
Expected: `echo_verl.echo_tool.EchoTool echo`
(If `yaml` is missing in `.venv`, install with `.venv/bin/pip install pyyaml` — it is a plain dev dep — or skip and have the reviewer parse-check.)

- [ ] **Step 3: Write the launch README** — document the concrete wiring the GPU run needs:
  - `rollout.multi_turn.enable=true`, `rollout.multi_turn.format=hermes`, `rollout.multi_turn.tool_config_path=echo_verl/configs/echo_tool_config.yaml`, `rollout.multi_turn.max_assistant_turns=<N>`.
  - `custom_reward_function.path=echo_verl/reward.py`, `custom_reward_function.name=compute_score`.
  - Dataset parquet from `echo_verl.generate_trainset` (rows carry `agent_name="tool_agent"`).
  - `transformers>=4.57`, base model `Qwen3-VL-8B-Instruct`, `data.video_key=videos`.
  - Mark clearly: **UNRUN — these are the flags a GPU operator sets; validate end-to-end before RL.**

- [ ] **Step 4: Commit**

```bash
git add echo_verl/configs/
git commit -m "chore(p3c): echo tool-config YAML + verl launch wiring README [UNRUN]"
```

---

## Self-Review

**Spec coverage** (INTEGRATION.md §0.2 + design §1 P3c):
- Composite `EchoTool` BaseTool, op-dispatch, image ToolResponse → Tasks 1–2. ✓
- Reward registration via custom_reward_function → Task 3. ✓
- Parquet data-gen (row contract, clip-supplied-twice) → Task 4. ✓
- Tool-config YAML + launch wiring → Task 5. ✓
- No agent-loop patch (hybrid) → by construction (images, not video). ✓

**Placeholder scan:** none — 🟢 tasks carry full code+tests; 🟡 tasks carry full code + explicit py_compile/parse gates.

**Type consistency:** `EchoSession.run(op, params)` used by `EchoTool.execute`; `compute_score` signature matches verl's `default_compute_score`; `build_row` consumes `rl_record`'s exact keys; tool name `"echo"` consistent across the YAML `function.name`, `tools_kwargs` key, and the P3a serializer.

**Verification tiers labeled:** T1/T3/T4-rowshaping 🟢 (real tests); T2/T4-writer/T5 🟡 (py_compile/parse + contract review, UNRUN). Honest — no fabricated "green" on verl-coupled code.
