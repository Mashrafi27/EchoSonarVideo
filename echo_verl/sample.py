"""Reproducible, stratified sampling of the SFT corpus.

Every training run must be reproducible from a record of what it sampled, because
we will run many experiments and need to compare them. So sampling is a named,
seeded operation that writes a MANIFEST next to the data: the parameters, the input
file's hash, the counts, and the id of every record selected.

Two problems this replaces:

  * `--limit N` took the first N records off the input stream. It is not random, it
    is not recorded, and it silently trained one full run on 2.3% of the corpus.
  * A uniform per-study cap would gut the report-style question types. Each study
    has ~11 structure_description and ~11 abnormality_classification records but
    EXACTLY ONE each of abnormality_list, conclusion and full_report. Sampling k of
    26 keeps only k/26 of the rare types, which are the ones the NLG metrics score.
    So caps are per question TYPE, and a type with no cap is kept whole.

Determinism does not depend on input order: records are keyed by a stable content
hash and each (study, type) group is shuffled with its own derived seed. The same
seed and caps give the same ids on any machine, whatever order the file is in.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path


def record_id(rec: dict) -> str:
    """Stable id for an SFT record. The corpus has no id field of its own.

    Keyed on content rather than position so that rebuilding sft.jsonl in a
    different order does not renumber everything and invalidate old manifests.
    """
    key = "|".join((rec.get("study_uuid", ""), rec.get("question_type", ""),
                    rec.get("question", "")))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def parse_caps(specs) -> dict:
    """['structure_description=3', 'conclusion=1'] -> {'structure_description': 3, ...}"""
    caps = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"bad --per-type {spec!r}, expected TYPE=N")
        name, _, count = spec.partition("=")
        caps[name.strip()] = int(count)
    return caps


def select(records, *, caps, default_cap=None, seed=0):
    """-> list of selected records, in a deterministic order.

    caps: {question_type: max per study}. default_cap applies to types not named;
    None there means keep every record of that type.
    """
    groups = defaultdict(list)
    for rec in records:
        groups[(rec.get("study_uuid"), rec.get("question_type"))].append(rec)

    chosen = []
    for (study, qtype), group in groups.items():
        cap = caps.get(qtype, default_cap)
        group = sorted(group, key=record_id)
        if cap is not None and len(group) > cap:
            # A seed derived from the group, so adding or removing an unrelated
            # study never reshuffles this one.
            rng = random.Random(f"{seed}:{study}:{qtype}")
            group = sorted(rng.sample(group, cap), key=record_id)
        chosen.extend(group)
    chosen.sort(key=record_id)
    return chosen


def build_manifest(*, selected, source, caps, default_cap, seed, out_path,
                   ids_path, n_source_records):
    counts = defaultdict(int)
    studies = set()
    for rec in selected:
        counts[rec.get("question_type")] += 1
        studies.add(rec.get("study_uuid"))
    ids = [record_id(r) for r in selected]
    return {
        "git_commit": _git_commit(),
        "source": {"path": str(source), "sha256": file_sha256(source),
                   "n_records": n_source_records},
        "params": {"seed": seed, "per_type_caps": dict(sorted(caps.items())),
                   "default_cap": default_cap},
        "output": {"parquet": str(out_path), "ids_file": str(ids_path)},
        "selected": {
            "n_records": len(selected),
            "n_studies": len(studies),
            "by_question_type": dict(sorted(counts.items())),
            # Lets a later run prove it selected the same set without diffing
            # a 36k-line file.
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        },
    }


def write_ids(ids_path, selected) -> None:
    Path(ids_path).write_text("\n".join(record_id(r) for r in selected) + "\n")
