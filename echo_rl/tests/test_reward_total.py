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


def test_extract_answer_empty_is_none():
    from echo_rl.reward.score import extract_answer, score_format
    assert extract_answer("<answer>   </answer>") is None
    assert extract_answer("<answer></answer>") is None
    # empty answer no longer earns the format answer-criterion
    assert score_format("<answer></answer>") == 0.0  # no think, no valid tool_call, no real answer


def test_serialized_toolcall_scores_as_valid_format():
    from echo_rl.sft.serialize import serialize_sft
    from echo_rl.reward.score import score_format
    traj = {
        "overview": {"type": "overview", "views": [{"view": "A4C", "frame": "a/5.png", "frame_count": 10}]},
        "turns": [
            {"type": "tool", "name": "select_view", "args": {"view": "A4C"},
             "obs": {"view": "A4C", "frames": ["a/0.png"]}},
            {"type": "think", "text": "A4C normal."},
        ],
        "answer": "No.",
    }
    msgs = serialize_sft(traj, "Q")
    first_assistant = msgs[1]["content"]   # "<think>...</think>\n<tool_call>...</tool_call>"
    assert score_format(first_assistant) == 1.0   # think present + valid tool_call
