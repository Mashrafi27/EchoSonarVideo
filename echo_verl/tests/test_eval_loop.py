"""Agentic eval loop tests, driven by a fake client and a fake session.

No model and no GPU: what is under test is the loop's control flow and its
honesty about failure, which is exactly what a live run makes hard to inspect.
"""
import pytest

from echo_verl.eval.agentic_loop import extract_answer, parse_tool_calls, run_episode


class FakeImage:
    def convert(self, mode):
        return self
    def save(self, buf, format):
        buf.write(b"png")


class FakeFrame:
    image = FakeImage()


class FakeObs:
    def __init__(self, ok=True, frames=(), text="obs", error=None):
        self.ok, self.frames, self.text, self.error = ok, list(frames), text, error


class FakeSession:
    def __init__(self, *responses, raises=False):
        self.responses, self.calls, self.raises = list(responses), [], raises
    def run(self, op, params):
        self.calls.append((op, params))
        if self.raises:
            raise FileNotFoundError("clip missing")
        return self.responses.pop(0) if self.responses else FakeObs()


class FakeClient:
    """Returns scripted assistant turns in order."""
    def __init__(self, *turns):
        self.turns, self.seen = list(turns), []
        outer = self
        class Completions:
            def create(self, model, messages, temperature, max_tokens):
                outer.seen.append(messages)
                text = outer.turns.pop(0) if outer.turns else "<answer>fallback</answer>"
                class M: content = text
                class C: message = M()
                class R: choices = [C()]
                return R()
        class Chat:
            completions = Completions()
        self.chat = Chat()


TOOL = ('<think>look</think><tool_call>{"name":"echo","arguments":'
        '{"op":"select_view","view_name":"A4C"}}</tool_call>')


def test_answer_on_first_turn_makes_no_tool_calls():
    s = FakeSession()
    ep = run_episode(FakeClient("<answer>Normal.</answer>"), "m", s, "q?", [FakeImage()])
    assert ep["answer"] == "Normal." and ep["finish_reason"] == "answered"
    assert ep["tool_calls"] == [] and s.calls == []


def test_tool_call_then_answer_is_recorded_in_order():
    s = FakeSession(FakeObs(frames=[FakeFrame()]))
    ep = run_episode(FakeClient(TOOL, "<answer>Mild MR.</answer>"), "m", s, "q?", [FakeImage()])
    assert ep["answer"] == "Mild MR." and ep["turns"] == 2
    assert [c["op"] for c in ep["tool_calls"]] == ["select_view"]
    assert ep["tool_calls"][0]["ok"] is True


def test_failed_tool_is_recorded_not_raised():
    s = FakeSession(FakeObs(ok=False, error="unknown view"))
    ep = run_episode(FakeClient(TOOL, "<answer>x</answer>"), "m", s, "q?", [FakeImage()])
    assert ep["tool_calls"][0]["ok"] is False
    assert ep["tool_calls"][0]["error"] == "unknown view"


def test_io_exception_becomes_a_recorded_failure():
    """A missing clip is data about the run, not a crash that loses the episode."""
    s = FakeSession(raises=True)
    ep = run_episode(FakeClient(TOOL, "<answer>x</answer>"), "m", s, "q?", [FakeImage()])
    assert ep["tool_calls"][0]["ok"] is False
    assert "FileNotFoundError" in ep["tool_calls"][0]["error"]
    assert ep["answer"] == "x"          # the episode still completes


def test_malformed_tool_call_is_counted_not_crashed():
    ep = run_episode(FakeClient("<tool_call>{not json</tool_call>"), "m",
                     FakeSession(), "q?", [FakeImage()])
    assert ep["malformed_tool_calls"] == 1
    assert ep["finish_reason"] == "no_answer_no_tool"
    assert ep["answer"] is None


def test_neither_answer_nor_tool_ends_the_episode():
    ep = run_episode(FakeClient("I think it looks fine."), "m",
                     FakeSession(), "q?", [FakeImage()])
    assert ep["finish_reason"] == "no_answer_no_tool" and ep["answer"] is None


def test_max_turns_is_enforced():
    """A model that only ever calls tools must terminate, and say why."""
    s = FakeSession(*[FakeObs(frames=[FakeFrame()]) for _ in range(10)])
    ep = run_episode(FakeClient(*[TOOL] * 10), "m", s, "q?", [FakeImage()], max_turns=3)
    assert ep["turns"] == 3 and ep["finish_reason"] == "max_turns"
    assert ep["answer"] is None


def test_image_budget_caps_observations():
    many = FakeObs(frames=[FakeFrame() for _ in range(50)])
    s = FakeSession(many)
    ep = run_episode(FakeClient(TOOL, "<answer>x</answer>"), "m", s, "q?",
                     [FakeImage()], max_images=5)
    assert ep["images_used"] <= 5


def test_overview_images_count_toward_the_budget():
    ep = run_episode(FakeClient("<answer>x</answer>"), "m", FakeSession(), "q?",
                     [FakeImage() for _ in range(19)])
    assert ep["images_used"] == 19


# ---- parsing ----

def test_parse_multiple_tool_calls_in_one_turn():
    text = TOOL + TOOL
    calls, malformed = parse_tool_calls(text)
    assert len(calls) == 2 and malformed == 0


def test_parse_rejects_non_dict_arguments():
    calls, malformed = parse_tool_calls('<tool_call>{"arguments": "A4C"}</tool_call>')
    assert calls == [] and malformed == 1


def test_extract_answer_takes_the_last_one():
    assert extract_answer("<answer>a</answer> then <answer>b</answer>") == "b"


def test_extract_answer_missing_is_none():
    assert extract_answer("no tags here") is None
