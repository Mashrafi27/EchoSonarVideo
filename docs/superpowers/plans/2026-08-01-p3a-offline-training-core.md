# P3a — Offline Echo-Training Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two stdlib, fully-offline-testable pieces that sit on top of Phase-1's structured records — an SFT trajectory **text serializer** and a reward **pure-scorer** — so Phase-3 training has a verified data/reward core before any GPU-coupled work.

**Architecture:** Phase-1 (`echo_rl/data/`) already emits structured `sft_record` (a trajectory dict) and `rl_record` (a `reward_key` dict). P3a adds two new sub-packages: `echo_rl/sft/` serializes a trajectory dict into a Qwen3-VL multi-turn **messages list** using the composite-`EchoTool` tool-call convention; `echo_rl/reward/` scores a model completion against a `reward_key` (yes/no exact-match, set-F1, clinical-entity-F1, gold-value co-signal, format reward, annealed tool bonus). Both are pure Python, no torch/PIL/verl.

**Tech Stack:** Python 3.13 stdlib only (`json`, `re`). Tests via `.venv/bin/pytest`.

## Global Constraints

- **`echo_rl` stays stdlib-only** — no PIL, torch, pyarrow, or verl imports. (Parquet data-gen is P3c.)
- **Verification tier 🟢** — every P3a deliverable ships with real passing unit tests. This is *genuinely done*, same bar as Phases 1–2.
- **Tool-call convention = composite `EchoTool` op-dispatch** (INTEGRATION.md §0.1): a tool call is JSON `{"name": "echo", "arguments": {"op": <select_view|select_frames|zoom>, ...}}`. Per-op argument keys match `echo_env._dispatch`: `view_name` (all ops), `indices` (select_frames), `bbox`+`frame_indices` (zoom).
- **Reuse Phase-1 helpers, do not reimplement:** `echo_rl.data.answers.{parse_yes_no, finding_set, is_abnormal}`, `echo_rl.data.trace.findings_text`.
- **Reward range** — every scorer returns a float in `[0.0, 1.0]`; `total_reward` returns a dict.
- **Serializer output is a canonical messages list**; the exact chat-template/role wrapping into verl's SFT trainer is a P3c 🟡 concern — do not couple to it here.
- Commit after each task with a `feat(p3a):` message.

---

### Task 1: SFT trajectory text serializer

**Files:**
- Create: `echo_rl/sft/__init__.py` (empty)
- Create: `echo_rl/sft/serialize.py`
- Test: `echo_rl/tests/test_sft_serialize.py`

**Interfaces:**
- Consumes: the trajectory dict produced by `echo_rl.data.trajectory.build_trajectory` — shape:
  `{"overview": {"type":"overview","views":[{"view":str,"frame":str,"frame_count":int}]}, "turns": [{"type":"tool","name":"select_view","args":{"view":str},"obs":{"view":str,"frames":[str]}}, {"type":"think","text":str}, ...], "answer": str}`.
- Produces: `serialize_sft(traj: dict, question: str, *, opening_think: str = "To answer this I should examine the relevant views.") -> list[dict]` — a messages list. Each message is `{"role": str, "content": <str | list[dict]>}`. Roles in order: `user`, then repeating (`assistant`, `tool`), ending in a final `assistant`. Assistant content is a string with `<think>...</think>` and either a `<tool_call>\n{json}\n</tool_call>` or a final `<answer>...</answer>`. Tool content is `[{"type":"video","frames":[str,...],"view":str}]`. User content is `[{"type":"text","text":str}, {"type":"video","frames":[str],"view":str}, ...]` (question + one thumbnail per overview view).

- [ ] **Step 1: Write the failing test**

```python
# echo_rl/tests/test_sft_serialize.py
import json
from echo_rl.sft.serialize import serialize_sft


def _traj():
    return {
        "overview": {"type": "overview", "views": [
            {"view": "A4C", "frame": "a4c/5.png", "frame_count": 10},
            {"view": "PLAX", "frame": "plax/4.png", "frame_count": 8},
        ]},
        "turns": [
            {"type": "tool", "name": "select_view", "args": {"view": "A4C"},
             "obs": {"view": "A4C", "frames": ["a4c/0.png", "a4c/5.png"]}},
            {"type": "think", "text": "A4C shows normal RV, mild TR."},
            {"type": "tool", "name": "select_view", "args": {"view": "PLAX"},
             "obs": {"view": "PLAX", "frames": ["plax/0.png"]}},
            {"type": "think", "text": "PLAX shows normal LV dimensions."},
        ],
        "answer": "No significant abnormality.",
    }


def test_role_sequence():
    msgs = serialize_sft(_traj(), "Is there any abnormality?")
    roles = [m["role"] for m in msgs]
    # user, (assistant, tool) x2, final assistant
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


def test_user_has_question_and_overview_thumbnails():
    msgs = serialize_sft(_traj(), "Is there any abnormality?")
    user = msgs[0]["content"]
    assert user[0] == {"type": "text", "text": "Is there any abnormality?"}
    vids = [c for c in user if c["type"] == "video"]
    assert [v["view"] for v in vids] == ["A4C", "PLAX"]
    assert vids[0]["frames"] == ["a4c/5.png"]  # overview = single thumbnail (the view's frame)


def test_first_assistant_has_opening_think_and_toolcall():
    msgs = serialize_sft(_traj(), "Q")
    a0 = msgs[1]["content"]
    assert "<think>To answer this I should examine the relevant views.</think>" in a0
    assert "<tool_call>" in a0 and "</tool_call>" in a0
    payload = json.loads(a0.split("<tool_call>")[1].split("</tool_call>")[0].strip())
    assert payload == {"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}


def test_tool_obs_carries_frames():
    msgs = serialize_sft(_traj(), "Q")
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["content"] == [{"type": "video", "frames": ["a4c/0.png", "a4c/5.png"], "view": "A4C"}]


def test_findings_become_next_assistant_think():
    msgs = serialize_sft(_traj(), "Q")
    # think about A4C (findings of turn 0) appears in the SECOND assistant turn
    assert "<think>A4C shows normal RV, mild TR.</think>" in msgs[3]["content"]


def test_final_assistant_has_answer_no_toolcall():
    msgs = serialize_sft(_traj(), "Q")
    last = msgs[-1]["content"]
    assert "<answer>No significant abnormality.</answer>" in last
    assert "<tool_call>" not in last
    # last findings (PLAX) precede the answer
    assert "<think>PLAX shows normal LV dimensions.</think>" in last


def test_no_mappable_turns_still_emits_user_and_answer():
    traj = {"overview": {"type": "overview", "views": []}, "turns": [], "answer": "Yes."}
    msgs = serialize_sft(traj, "Q")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "<answer>Yes.</answer>" in msgs[1]["content"]
    assert "<think>To answer this I should examine the relevant views.</think>" in msgs[1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_sft_serialize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_rl.sft'`

- [ ] **Step 3: Write minimal implementation**

```python
# echo_rl/sft/serialize.py
"""Serialize a Phase-1 trajectory dict into a Qwen3-VL multi-turn messages list.

Tool-call convention = composite EchoTool op-dispatch (INTEGRATION.md §0.1):
    <tool_call>
    {"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}
    </tool_call>
The exact chat-template/role wrapping into verl's SFT trainer is a P3c concern;
this produces a canonical messages list only.
"""
import json

_OPENING = "To answer this I should examine the relevant views."


def _tool_call_json(name: str, args: dict) -> str:
    # Phase-1 trajectory uses args={"view": ...}; map to the op-dispatch schema.
    op_args = {"op": name, "view_name": args["view"]}
    payload = {"name": "echo", "arguments": op_args}
    return "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"


def _assistant(think: str, tail: str) -> dict:
    return {"role": "assistant", "content": f"<think>{think}</think>\n{tail}"}


def _iter_tool_think(turns: list):
    """Yield (tool_turn, findings_text) pairs from the flat [tool, think, ...] list."""
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.get("type") != "tool":
            i += 1
            continue
        think = ""
        if i + 1 < len(turns) and turns[i + 1].get("type") == "think":
            think = turns[i + 1]["text"]
        yield t, think
        i += 2


def serialize_sft(traj: dict, question: str, *, opening_think: str = _OPENING) -> list[dict]:
    user_content = [{"type": "text", "text": question}]
    for v in traj["overview"]["views"]:
        user_content.append({"type": "video", "frames": [v["frame"]], "view": v["view"]})
    messages = [{"role": "user", "content": user_content}]

    pending_think = opening_think
    for tool, findings in _iter_tool_think(traj["turns"]):
        messages.append(_assistant(pending_think, _tool_call_json(tool["name"], tool["args"])))
        obs = tool["obs"]
        messages.append({"role": "tool",
                         "content": [{"type": "video", "frames": obs["frames"], "view": obs["view"]}]})
        pending_think = findings

    messages.append(_assistant(pending_think, f"<answer>{traj['answer']}</answer>"))
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_sft_serialize.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add echo_rl/sft/__init__.py echo_rl/sft/serialize.py echo_rl/tests/test_sft_serialize.py
git commit -m "feat(p3a): SFT trajectory text serializer (dict -> Qwen3-VL messages)"
```

---

### Task 2: Reward outcome scorers

**Files:**
- Create: `echo_rl/reward/__init__.py` (empty)
- Create: `echo_rl/reward/score.py`
- Test: `echo_rl/tests/test_reward_score.py`

**Interfaces:**
- Consumes: `echo_rl.data.answers.{parse_yes_no, finding_set}`; the `reward_key` dict from
  `echo_rl.data.builders._reward_key` — shape `{"kind": "yesno"|"set"|"text", "target": <str|list[str]>, "gold": {metric: label_str}, "is_abnormal": bool}`.
- Produces:
  - `f1(pred: set, gold: set) -> float`
  - `score_yesno(pred_answer: str, target: str) -> float`
  - `score_set(pred_answer: str, target: list) -> float`
  - `extract_entities(text: str) -> set` (normalized clinical finding tokens)
  - `score_entity_f1(pred_answer: str, ref_answer: str) -> float`
  - `score_gold_value(pred_answer: str, gold: dict) -> float | None`
  - `class JudgeClient` (protocol) with `score(question, pred, ref) -> float | None`; `class NullJudge` returning `None`
  - `score_outcome(reward_key: dict, pred_answer: str, *, question: str = "", judge: JudgeClient = NullJudge()) -> float`

- [ ] **Step 1: Write the failing test**

```python
# echo_rl/tests/test_reward_score.py
from echo_rl.reward.score import (
    f1, score_yesno, score_set, extract_entities, score_entity_f1,
    score_gold_value, score_outcome, NullJudge,
)


def test_f1_edges():
    assert f1(set(), set()) == 1.0            # both empty = perfect
    assert f1({"a"}, set()) == 0.0
    assert f1(set(), {"a"}) == 0.0
    assert f1({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(f1({"a", "b"}, {"a"}) - 2/3) < 1e-9   # P=1/2, R=1 -> F1=2/3


def test_score_yesno():
    assert score_yesno("Yes, there is dilation.", "yes") == 1.0
    assert score_yesno("No.", "no") == 1.0
    assert score_yesno("Yes", "no") == 0.0
    assert score_yesno("Maybe", "yes") == 0.0   # parse_yes_no -> None


def test_score_set():
    ans = "- mitral regurgitation\n- lv dilation"
    assert score_set(ans, ["mitral regurgitation", "lv dilation"]) == 1.0
    assert score_set("No significant abnormalities.", []) == 1.0
    assert 0.0 < score_set(ans, ["mitral regurgitation"]) < 1.0


def test_extract_entities_and_entity_f1():
    pred = "The LV is dilated with reduced systolic function."
    ref = "Dilated LV, reduced function."
    ents = extract_entities(pred)
    assert "dilat" in ents and "reduced" in ents
    assert score_entity_f1(pred, ref) > 0.0
    assert score_entity_f1("Totally normal study.", "Severe stenosis and dilation.") == 0.0


def test_score_gold_value():
    assert score_gold_value("EF is severely reduced.", {"ef": "severely reduced"}) == 1.0
    assert score_gold_value("Normal.", {"ef": "severely reduced"}) == 0.0
    assert score_gold_value("anything", {}) is None       # nothing to score


def test_score_outcome_dispatch():
    assert score_outcome({"kind": "yesno", "target": "yes", "gold": {}}, "Yes.") == 1.0
    assert score_outcome({"kind": "set", "target": ["lv dilation"], "gold": {}},
                         "- lv dilation") == 1.0
    # text: no gold, NullJudge -> falls back to entity-F1
    r = score_outcome({"kind": "text", "target": "Dilated LV.", "gold": {}},
                      "The LV is dilated.", judge=NullJudge())
    assert r > 0.0


def test_score_outcome_text_prefers_gold_when_present():
    rk = {"kind": "text", "target": "long free text", "gold": {"ef": "reduced"}}
    assert score_outcome(rk, "EF reduced.") == 1.0
    assert score_outcome(rk, "EF normal.") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_reward_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echo_rl.reward'`

- [ ] **Step 3: Write minimal implementation**

```python
# echo_rl/reward/score.py
"""Pure, model-free reward scoring over Phase-1 reward_keys.

Outcome scorers return a float in [0, 1]. The LLM-judge is injected behind the
JudgeClient interface; NullJudge (offline) returns None so score_text falls back
to the clinical-entity-F1 co-signal. A real vLLM judge client is P3e.
"""
import re
from echo_rl.data.answers import parse_yes_no, finding_set

# Clinical-finding vocabulary for free-text entity extraction (mirrors
# echo_rl.data.answers.is_abnormal's abnormal-keyword set).
_ENTITY_RE = re.compile(
    r"(dilat|reduced|abnormal|severe|moderate|mild|regurgitat|stenos|"
    r"hypertroph|impaired|akinet|hypokinet|effusion|thromb|normal)", re.I)


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_yesno(pred_answer: str, target: str) -> float:
    return 1.0 if parse_yes_no(pred_answer) == target else 0.0


def score_set(pred_answer: str, target: list) -> float:
    return f1(finding_set(pred_answer), set(target or []))


def extract_entities(text: str) -> set:
    return {m.group(1).lower() for m in _ENTITY_RE.finditer(text or "")}


def score_entity_f1(pred_answer: str, ref_answer: str) -> float:
    return f1(extract_entities(pred_answer), extract_entities(ref_answer))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def score_gold_value(pred_answer: str, gold: dict) -> float | None:
    labels = [v for v in (gold or {}).values() if v]
    if not labels:
        return None
    p = _norm(pred_answer)
    hits = sum(1 for lab in labels if _norm(lab) in p)
    return hits / len(labels)


class JudgeClient:
    def score(self, question: str, pred: str, ref: str):
        raise NotImplementedError


class NullJudge(JudgeClient):
    def score(self, question: str, pred: str, ref: str):
        return None


def score_outcome(reward_key: dict, pred_answer: str, *, question: str = "",
                  judge: JudgeClient = NullJudge()) -> float:
    kind = reward_key.get("kind")
    if kind == "yesno":
        return score_yesno(pred_answer, reward_key.get("target"))
    if kind == "set":
        return score_set(pred_answer, reward_key.get("target"))
    # text: prefer structured gold; else judge (blended with entity-F1); else entity-F1.
    gv = score_gold_value(pred_answer, reward_key.get("gold"))
    if gv is not None:
        return gv
    ref = reward_key.get("target", "")
    jv = judge.score(question, pred_answer, ref)
    ef = score_entity_f1(pred_answer, ref)
    if jv is not None:
        return 0.5 * jv + 0.5 * ef
    return ef
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_reward_score.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add echo_rl/reward/__init__.py echo_rl/reward/score.py echo_rl/tests/test_reward_score.py
git commit -m "feat(p3a): reward outcome scorers (yes/no, set-F1, entity-F1, gold-value)"
```

---

### Task 3: Format reward + total-reward combiner

**Files:**
- Modify: `echo_rl/reward/score.py` (append format + combiner functions)
- Test: `echo_rl/tests/test_reward_total.py`

**Interfaces:**
- Consumes: `score_outcome` and the `reward_key` from Task 2.
- Produces:
  - `extract_answer(completion: str) -> str | None` — text of the last `<answer>...</answer>`, else None
  - `score_format(completion: str) -> float` — fraction in [0,1] of format criteria met: (a) a `<think>...</think>` block present, (b) at least one syntactically valid `<tool_call>` JSON *or* a well-formed `<answer>`
  - `total_reward(reward_key: dict, completion: str, *, tool_calls: int = 0, tool_bonus_coef: float = 0.0, outcome_weight: float = 1.0, format_weight: float = 0.2, question: str = "", judge: JudgeClient = NullJudge()) -> dict` — returns `{"reward": float, "outcome": float, "format": float, "tool_bonus": float}`. `outcome` is `score_outcome` on the extracted answer (0.0 if no answer). `tool_bonus = tool_bonus_coef` if `tool_calls >= 1` else `0.0` (annealed by decaying `tool_bonus_coef` over training — the caller passes the current coef).

- [ ] **Step 1: Write the failing test**

```python
# echo_rl/tests/test_reward_total.py
from echo_rl.reward.score import extract_answer, score_format, total_reward, NullJudge

_TOOLCALL = '<tool_call>\n{"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}\n</tool_call>'


def test_extract_answer_last_wins():
    assert extract_answer("<answer>first</answer> ... <answer>second</answer>") == "second"
    assert extract_answer("no answer here") is None


def test_score_format():
    good = f"<think>reasoning</think>\n{_TOOLCALL}"
    assert score_format(good) == 1.0
    answer_only = "<think>t</think>\n<answer>No.</answer>"
    assert score_format(answer_only) == 1.0
    no_think = _TOOLCALL
    assert score_format(no_think) == 0.5           # tool-call ok, think missing
    malformed = "<think>t</think>\n<tool_call>not json</tool_call>"
    assert score_format(malformed) == 0.5          # think ok, tool-call invalid, no answer
    assert score_format("plain text") == 0.0


def test_total_reward_combines():
    rk = {"kind": "yesno", "target": "no", "gold": {}}
    completion = "<think>looks normal</think>\n<answer>No.</answer>"
    r = total_reward(rk, completion, tool_calls=2, tool_bonus_coef=0.1)
    assert r["outcome"] == 1.0
    assert r["format"] == 1.0
    assert r["tool_bonus"] == 0.1
    assert abs(r["reward"] - (1.0 * 1.0 + 0.2 * 1.0 + 0.1)) < 1e-9


def test_total_reward_no_answer_scores_zero_outcome():
    rk = {"kind": "yesno", "target": "no", "gold": {}}
    r = total_reward(rk, "<think>hmm</think> no answer tag", tool_calls=0, tool_bonus_coef=0.1)
    assert r["outcome"] == 0.0
    assert r["tool_bonus"] == 0.0                   # no tool call


def test_annealed_bonus_is_caller_controlled():
    rk = {"kind": "yesno", "target": "yes", "gold": {}}
    c = "<think>t</think>\n<answer>Yes.</answer>"
    early = total_reward(rk, c, tool_calls=1, tool_bonus_coef=0.2)
    late = total_reward(rk, c, tool_calls=1, tool_bonus_coef=0.0)
    assert early["tool_bonus"] == 0.2 and late["tool_bonus"] == 0.0
    assert early["reward"] > late["reward"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest echo_rl/tests/test_reward_total.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_answer'`

- [ ] **Step 3: Write minimal implementation** (append to `echo_rl/reward/score.py`)

```python
import json as _json

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def extract_answer(completion: str):
    matches = _ANSWER_RE.findall(completion or "")
    return matches[-1].strip() if matches else None


def _has_valid_tool_call(completion: str) -> bool:
    for body in _TOOLCALL_RE.findall(completion or ""):
        try:
            payload = _json.loads(body.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
            return True
    return False


def score_format(completion: str) -> float:
    criteria = 0
    if _THINK_RE.search(completion or ""):
        criteria += 1
    if _has_valid_tool_call(completion) or extract_answer(completion) is not None:
        criteria += 1
    return criteria / 2.0


def total_reward(reward_key: dict, completion: str, *, tool_calls: int = 0,
                 tool_bonus_coef: float = 0.0, outcome_weight: float = 1.0,
                 format_weight: float = 0.2, question: str = "",
                 judge: JudgeClient = NullJudge()) -> dict:
    answer = extract_answer(completion)
    outcome = score_outcome(reward_key, answer, question=question, judge=judge) if answer else 0.0
    fmt = score_format(completion)
    tool_bonus = tool_bonus_coef if tool_calls >= 1 else 0.0
    reward = outcome_weight * outcome + format_weight * fmt + tool_bonus
    return {"reward": reward, "outcome": outcome, "format": fmt, "tool_bonus": tool_bonus}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest echo_rl/tests/test_reward_total.py echo_rl/tests/test_reward_score.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add echo_rl/reward/score.py echo_rl/tests/test_reward_total.py
git commit -m "feat(p3a): format reward + annealed total-reward combiner"
```

---

## Self-Review

**Spec coverage** (against `2026-08-01-phase3-training-integration-design.md` §4, reduced P3a):
- SFT trajectory text serializer → Task 1. ✓
- Reward pure-scoring (yes/no, set-F1, entity-F1, gold-value, format) → Tasks 2–3. ✓
- Annealed tool bonus (spec §5.2) → Task 3 `total_reward`. ✓
- LLM-judge behind an interface, stubbed offline (spec §4) → Task 2 `JudgeClient`/`NullJudge`. ✓
- Parquet data-gen → **intentionally deferred to P3c** (pyarrow dep + verl-schema coupling; see spec §1 P3c row). Not a gap.

**Placeholder scan:** none — all steps carry real code/tests.

**Type consistency:** `reward_key` shape (`kind`/`target`/`gold`/`is_abnormal`) matches `echo_rl.data.builders._reward_key`. `serialize_sft` consumes exactly `build_trajectory`'s output shape. `score_outcome`/`total_reward`/`JudgeClient` signatures are consistent across Tasks 2–3.

**Verification tier:** all 🟢 — real pytest, no runtime coupling. Genuinely done on completion.
