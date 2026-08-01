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
