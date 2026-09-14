import csv
import hashlib
import json


def load_test_studies(vqa_test_path: str) -> set:
    out = set()
    with open(vqa_test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["study_uuid"])
    return out


def load_canonical_split(path: str) -> dict:
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                u = row.get("study_uuid")
                if u:
                    out[u] = (row.get("split") or "").upper()
    except FileNotFoundError:
        return {}
    return out


def assign_split(study_uuid: str, canonical: dict) -> str:
    s = canonical.get(study_uuid)
    if s in ("TRAIN", "VAL", "TEST"):
        return s.lower()
    h = int(hashlib.md5(study_uuid.encode()).hexdigest(), 16) % 100
    return "val" if h < 5 else "train"
