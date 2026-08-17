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


def test_user_is_one_image_per_view_then_question():
    # The view menu is one image per view -- NOT a video. Qwen3-VL's video processor
    # would resample a 19-frame "clip" down to 4, hiding most of the menu.
    msgs = serialize_sft(_traj(), "Is there any abnormality?")
    user = msgs[0]["content"]
    imgs = [c for c in user if c["type"] == "image"]
    assert len(imgs) == 2
    assert [i["frames"][0] for i in imgs] == ["a4c/5.png", "plax/4.png"]
    assert [i["views"][0] for i in imgs] == ["A4C", "PLAX"]
    assert user[-1] == {"type": "text", "text": "Is there any abnormality?"}


def test_no_video_content_anywhere():
    msgs = serialize_sft(_traj(), "Q")
    types = {c["type"] for m in msgs if not isinstance(m["content"], str) for c in m["content"]}
    assert "video" not in types


def test_first_assistant_has_opening_think_and_toolcall():
    msgs = serialize_sft(_traj(), "Q")
    a0 = msgs[1]["content"]
    assert "<think>To answer this I should examine the relevant views.</think>" in a0
    assert "<tool_call>" in a0 and "</tool_call>" in a0
    payload = json.loads(a0.split("<tool_call>")[1].split("</tool_call>")[0].strip())
    assert payload == {"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}


def test_tool_obs_carries_frames_as_images():
    # HYBRID: tool observations are IMAGES at rollout time, so SFT tags them "image".
    msgs = serialize_sft(_traj(), "Q")
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["content"] == [{"type": "image", "frames": ["a4c/0.png", "a4c/5.png"], "view": "A4C"}]


def test_no_tool_obs_is_tagged_video():
    msgs = serialize_sft(_traj(), "Q")
    tool_content = [c for m in msgs if m["role"] == "tool" for c in m["content"]]
    assert tool_content and all(c["type"] == "image" for c in tool_content)


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
