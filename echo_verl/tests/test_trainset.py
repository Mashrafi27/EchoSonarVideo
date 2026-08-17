# echo_verl/tests/test_trainset.py
import json

from echo_verl.generate_trainset import build_row, overview_image_specs
from echo_verl.reward import compute_score


def _rl_rec():
    return {"study_uuid": "S1", "question_type": "abnormality_classification",
            "question": "Any abnormality?", "answer": "No.",
            "reward_key": {"kind": "yesno", "target": "no", "gold": {}, "is_abnormal": False},
            "overview": {"views": [{"view": "A4C", "frame": "a4c/5.png"},
                                   {"view": "PLAX", "frame": "plax/4.png"}]},
            "designation": ""}


def test_build_row_shape():
    row = build_row(_rl_rec(), overview_image_specs(_rl_rec()))
    assert row["data_source"] == "echo"
    assert row["agent_name"] == "tool_agent"                       # selects ToolAgentLoop
    assert isinstance(row["prompt"], list) and row["prompt"][0]["role"] == "user"
    # view menu = one <image> per view, no video anywhere
    assert row["prompt"][0]["content"].count("<image>") == 2
    assert "<video>" not in row["prompt"][0]["content"]
    assert row["videos"] == []
    assert row["images"] == [{"image": "a4c/5.png"}, {"image": "plax/4.png"}]
    assert json.loads(row["reward_model"]["ground_truth"]) == _rl_rec()["reward_key"]
    ek = row["extra_info"]
    assert ek["need_tools_kwargs"] is True
    assert ek["tools_kwargs"]["echo"]["create_kwargs"]["study_uuid"] == "S1"


def test_placeholder_count_matches_images_column():
    # verl asserts image placeholders == len(images); a mismatch crashes the rollout.
    row = build_row(_rl_rec(), overview_image_specs(_rl_rec()))
    assert row["prompt"][0]["content"].count("<image>") == len(row["images"])


def test_study_uuid_reaches_the_tool():
    row = build_row(_rl_rec(), overview_image_specs(_rl_rec()))
    assert row["extra_info"]["tools_kwargs"]["echo"]["create_kwargs"]["study_uuid"] == "S1"


def _rec(qtype, target):
    return {"study_uuid": "S1", "question_type": qtype, "question": "Q?", "answer": "A",
            "reward_key": {"kind": "yesno" if qtype == "abnormality_classification" else "set",
                           "target": target, "gold": {}, "is_abnormal": False},
            "overview": {"views": []}, "designation": ""}


def test_ground_truth_uniform_string_and_roundtrips():
    yesno = build_row(_rec("abnormality_classification", "no"), [])
    lst = build_row(_rec("abnormality_list", ["mild mr"]), [])
    # uniform column type -> pyarrow can unify the schema across mixed question types
    assert isinstance(yesno["reward_model"]["ground_truth"], str)
    assert isinstance(lst["reward_model"]["ground_truth"], str)
    # round-trips back to the original reward_key and scores without crashing
    assert json.loads(yesno["reward_model"]["ground_truth"])["target"] == "no"
    assert json.loads(lst["reward_model"]["ground_truth"])["target"] == ["mild mr"]
    r = compute_score("echo", "<think>t</think>\n<answer>No.</answer>",
                      yesno["reward_model"]["ground_truth"])
    assert r >= 0.0
