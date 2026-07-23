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


def assign_split(study_uuid: str, designation, test_studies: set) -> str:
    if study_uuid in test_studies:
        return "test"
    d = (designation or "").upper()
    if d == "TEST":
        return "test"
    if d == "VAL":
        return "val"
    if d == "TRAIN":
        return "train"
    h = int(hashlib.md5(study_uuid.encode()).hexdigest(), 16) % 100
    return "val" if h < 5 else "train"
