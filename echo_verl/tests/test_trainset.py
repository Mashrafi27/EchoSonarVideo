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
