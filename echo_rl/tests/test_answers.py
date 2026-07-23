import json
from echo_rl.data.answers import last_answer, parse_yes_no, finding_set, is_abnormal


def test_last_answer_from_string():
    msgs = json.dumps([{"role": "user", "content": "q"},
                       {"role": "assistant", "content": " Normal. "}])
    assert last_answer(msgs) == "Normal."


def test_parse_yes_no():
    assert parse_yes_no("No, there is no evidence of RAE.") == "no"
    assert parse_yes_no("Yes, present.") == "yes"
    assert parse_yes_no("Normal dimensions.") is None


def test_finding_set():
    ans = "The following abnormalities are identified:\n- Tricuspid Regurgitation\n- Mitral Valve Prolapse"
    assert finding_set(ans) == {"tricuspid regurgitation", "mitral valve prolapse"}
    assert finding_set("No significant abnormalities are identified.") == set()


def test_is_abnormal():
    assert is_abnormal("abnormality_classification", "Yes, present.") is True
    assert is_abnormal("abnormality_classification", "No, none.") is False
    assert is_abnormal("abnormality_list", "- Tricuspid Regurgitation") is True
