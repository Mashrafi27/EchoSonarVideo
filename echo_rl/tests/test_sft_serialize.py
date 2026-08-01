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
