import csv
import os


def load_metric(csv_path: str) -> dict:
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("study_id")
            if not sid:
                continue
            out[sid] = {"label": row.get("label", ""),
                        "designation": row.get("study designation", "")}
    return out


def load_all(gold_dir: str) -> dict:
    merged: dict = {}
    for fname in sorted(os.listdir(gold_dir)):
        if not fname.endswith(".csv"):
            continue
        metric = os.path.splitext(fname)[0]
        for sid, rec in load_metric(os.path.join(gold_dir, fname)).items():
            slot = merged.setdefault(sid, {})
            slot[metric] = rec["label"]
            if rec["designation"] and "designation" not in slot:
                slot["designation"] = rec["designation"]
    return merged
