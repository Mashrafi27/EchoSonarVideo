import json
from echo_rl.data.answers import last_answer, parse_yes_no, finding_set, is_abnormal
from echo_rl.data.trajectory import build_trajectory, overview_turn


def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _question(rec: dict) -> str:
    msgs = json.loads(rec["messages"]) if isinstance(rec["messages"], str) else rec["messages"]
    for m in msgs:
        if m.get("role") == "user":
            return m["content"].strip()
    return ""


def sft_record(rec: dict, clips: list, cfg) -> dict:
    answer = last_answer(rec["messages"])
    traj = build_trajectory(_question(rec), answer, rec.get("thinking", ""), clips, cfg)
    return {"study_uuid": rec["study_uuid"], "question_type": rec["question_type"],
            "question": _question(rec), "trajectory": traj}


def _reward_key(qtype: str, answer: str, gold_for_study: dict) -> dict:
    if qtype == "abnormality_classification":
        kind, target = "yesno", parse_yes_no(answer)
    elif qtype == "abnormality_list":
        kind, target = "set", sorted(finding_set(answer))
    else:
        kind, target = "text", answer
    gold = {k: v for k, v in (gold_for_study or {}).items() if k != "designation"}
    return {"kind": kind, "target": target, "gold": gold,
            "is_abnormal": is_abnormal(qtype, answer)}


def rl_record(rec: dict, clips: list, cfg, gold: dict) -> dict:
    answer = last_answer(rec["messages"])
    g = gold.get(rec["study_uuid"], {})
    return {"study_uuid": rec["study_uuid"], "question_type": rec["question_type"],
            "question": _question(rec), "answer": answer,
            "reward_key": _reward_key(rec["question_type"], answer, g),
            "overview": overview_turn(clips, cfg),
            "designation": g.get("designation", "")}
