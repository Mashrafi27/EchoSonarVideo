import json
from echo_rl.config import Config
from echo_rl.data.studies import Clip
from echo_rl.data.builders import sft_record, rl_record


def _rec(qtype, answer):
    return {"study_uuid": "st-1", "question_type": qtype, "thinking": "#### 1. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n",
            "messages": json.dumps([{"role": "user", "content": "Q?"},
                                    {"role": "assistant", "content": answer}])}


def _clips():
    return [Clip("di-2", "A4C", "/d/di-2_A4C", [f"{i}.png" for i in range(20)])]


def test_sft_record():
    cfg = Config.from_env()
    out = sft_record(_rec("structure_description", "RV normal."), _clips(), cfg)
    assert out["question"] == "Q?"
    assert out["trajectory"]["answer"] == "RV normal."


def test_rl_record_yesno_and_gold():
    cfg = Config.from_env()
    gold = {"st-1": {"ejection_fraction_regression": "51.0", "designation": "TRAIN"}}
    out = rl_record(_rec("abnormality_classification", "No, none."), _clips(), cfg, gold)
    assert out["reward_key"]["kind"] == "yesno"
    assert out["reward_key"]["is_abnormal"] is False
    assert out["reward_key"]["gold"]["ejection_fraction_regression"] == "51.0"
    assert out["designation"] == "TRAIN"
    assert len(out["overview"]["views"]) == 1
