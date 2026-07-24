from echo_env.parse import parse_action


def test_parse_answer():
    p = parse_action("reasoning... <answer>EF is 55%</answer>")
    assert p.answer == "EF is 55%"
    assert p.calls == []


def test_parse_single_tool_call():
    s = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    p = parse_action(s)
    assert p.answer is None
    assert len(p.calls) == 1
    assert p.calls[0]["name"] == "select_view"
    assert p.calls[0]["arguments"]["view_name"] == "A4C"


def test_parse_multiple_tool_calls():
    s = ('<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
         '<tool_call>{"name": "select_view", "arguments": {"view_name": "PLAX"}}</tool_call>')
    p = parse_action(s)
    assert len(p.calls) == 2


def test_parse_malformed_json_recorded_not_raised():
    s = '<tool_call>{"name": "zoom", "arguments": {oops}}</tool_call>'
    p = parse_action(s)
    assert p.calls == []
    assert len(p.errors) == 1


def test_missing_arguments_defaults_empty():
    s = '<tool_call>{"name": "select_view"}</tool_call>'
    p = parse_action(s)
    assert p.calls[0]["arguments"] == {}


def test_last_answer_wins():
    p = parse_action("<answer>first</answer> more <answer>second</answer>")
    assert p.answer == "second"
