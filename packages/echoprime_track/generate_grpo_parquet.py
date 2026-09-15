"""GRPO parquet builder for the frozen-EchoPrime + Qwen3-8B-text track.

Prompts now match Darya's real SFT observation format (report_generation/sft_thinking/
dataset.py::__getitem__, reused verbatim): per view, "{view_name}:\n" + up to 393 CLIP_TOKEN
placeholders + (if detections exist) "Structure features:\n" + one DETR_TOKEN per detected
class. Training/eval data is restricted to verifiable question types only (spec: docs/
superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md section 4) -- structure_description/
conclusion/full_report fall back to unverifiable entity_F1 for reward and are excluded here,
at generation time, not filtered later.

Reads Darya's h5 caches (report_generation/data/clip_tokens_{train,test}.h5,
{train,test}_detections*.h5 -- note the real on-disk eval file is test_detections_merged.h5,
not test_detections.h5) directly, lazily, per dicom_uuid, via darya_cache.py -- no intermediate
per-study cache (see that module's docstring / this plan's Step-0 design correction).

build/rl.jsonl's records don't carry a `dicoms_by_view` field directly -- they carry
`overview.views[]` ({"view", "frame", "frame_count"}), one entry per raw acquisition (a study
can have 2-3 duplicate clips of the same view). `build_dicoms_by_view` groups these by
view_name and picks one dicom_uuid per view, preferring whichever candidate is already present
in Darya's clip h5 (a strict improvement over Darya's own `_select_dicoms` in
report_generation/sft_thinking/dataset.py, which picks `random.choice(group)` with no coverage
awareness -- preferring a covered candidate only ever increases effective coverage, never
deviates from what her SFT checkpoint already tolerates: a missing dicom just means that view's
clip-token block is silently empty, spec section 1).
"""
import json
import os

from echoprime_track.darya_cache import detr_class_ids_present, load_clip_tokens
from echoprime_track.dataset import SYSTEM_PROMPT
from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN

_DATA_SOURCE = "echoprime_grpo"
_AGENT_NAME = "echoprime_tool_agent"  # Task 8 registers the new multi-turn loop under this name

VERIFIABLE_QUESTION_TYPES = {"abnormality_classification", "abnormality_list"}


def _dicom_uuid_from_frame_path(frame_path: str) -> str:
    """Extract dicom_uuid from a frame path's parent directory name, e.g.
    ".../di-2BFB-05C5-90B8_A4C/14.png" -> "di-2BFB-05C5-90B8". Same "di-<uuid>_<View>"
    convention as packages/data_core/data/views.py::parse_clip_dirname -- duplicated here as a
    one-liner rather than cross-imported, same call scripts/check_darya_cache_coverage.py
    already made."""
    clip_dirname = os.path.basename(os.path.dirname(frame_path))
    return clip_dirname.split("_", 1)[0]


def build_dicoms_by_view(views: list, clip_h5) -> dict:
    """Group `overview.views[]` by view_name, picking one dicom_uuid per view -- a study can
    have multiple raw acquisitions of the same view (e.g. 2-3 duplicate A4C clips). Prefers
    whichever candidate is already present in Darya's clip h5 (checked via membership, not by
    loading tokens); falls back to the first candidate when none of a view's candidates are
    covered -- `_view_block` below already handles a picked-but-uncovered dicom correctly
    (n_clip = 0). Preserves each view's first-seen order in `views` (Python dict insertion
    order) -- this order is what fixes the CLIP_TOKEN/DETR_TOKEN placement order in the built
    prompt text, so it must be identical to whatever order the tensors are loaded in later
    (rl_dataset.py / Task 8's agent loop) -- see write_parquet's docstring for how that
    ordering is preserved through the parquet round-trip."""
    groups: dict = {}
    for v in views:
        dicom_uuid = _dicom_uuid_from_frame_path(v["frame"])
        groups.setdefault(v["view"], []).append(dicom_uuid)

    picked = {}
    for view_name, candidates in groups.items():
        covered = [c for c in candidates if c in clip_h5]
        picked[view_name] = covered[0] if covered else candidates[0]
    return picked


def _view_block(view_name: str, dicom_uuid: str, clip_h5, detr_h5) -> str:
    parts = [f"{view_name}:\n"]
    tokens = load_clip_tokens(clip_h5, dicom_uuid)
    n_clip = tokens.shape[0] if tokens is not None else 0
    parts.append(CLIP_TOKEN * n_clip)
    class_ids = detr_class_ids_present(detr_h5, dicom_uuid)
    if class_ids:
        parts.append("\nStructure features:\n")
        parts.append(DETR_TOKEN * len(class_ids))
    return "".join(parts)


def build_row(rl_rec: dict, clip_h5, detr_h5) -> dict | None:
    """Returns None for a row whose question_type isn't verifiable (spec section 4) --
    caller drops those rather than writing them to the parquet at all."""
    if rl_rec["question_type"] not in VERIFIABLE_QUESTION_TYPES:
        return None

    dicoms_by_view = rl_rec["dicoms_by_view"]  # {view_name: dicom_uuid}, from build/rl.jsonl
    blocks = [_view_block(view, dicom_uuid, clip_h5, detr_h5)
              for view, dicom_uuid in dicoms_by_view.items()]
    user_turn = "\n".join(blocks) + f"\n{rl_rec['question']}"

    return {
        "data_source": _DATA_SOURCE,
        "agent_name": _AGENT_NAME,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_turn}],
        "reward_model": {"ground_truth": json.dumps(rl_rec["reward_key"]), "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {"study_uuid": rl_rec["study_uuid"],
                        "question_type": rl_rec["question_type"],
                        "dicom_uuids_by_view": dicoms_by_view,
                        "need_tools_kwargs": False},
    }


def write_parquet(rows: list, path: str) -> int:
    """`extra_info.dicom_uuids_by_view` is JSON-encoded before writing to parquet -- confirmed
    this session: `pa.Table.from_pylist` infers a STRUCT type for a nested dict field by
    unioning ALL keys seen across every row (alphabetically), nulling missing ones per row --
    it does NOT preserve each row's own insertion order. Since different studies have different
    view sets, leaving this a raw dict would desync the per-view h5-tensor-read order
    (rl_dataset.py / Task 8's agent loop, which iterate the read-back dict) from the
    CLIP_TOKEN/DETR_TOKEN placement order baked into the prompt TEXT at generation time above
    -- a silent training-data-corruption bug, not just a style issue. json.dumps (matching
    reward_model.ground_truth's existing convention in this same function) sidesteps both
    problems: order survives exactly, and the column stays a single string instead of
    exploding into one struct field per distinct view name across the whole dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    encoded_rows = []
    for row in rows:
        row = dict(row)
        extra_info = dict(row["extra_info"])
        extra_info["dicom_uuids_by_view"] = json.dumps(extra_info["dicom_uuids_by_view"])
        row["extra_info"] = extra_info
        encoded_rows.append(row)

    table = pa.Table.from_pylist(encoded_rows)
    pq.write_table(table, path)
    return len(rows)


def main(argv=None) -> int:
    import argparse
    import random

    import h5py

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-jsonl", default="build/rl.jsonl")
    ap.add_argument("--study-list", default=None,
                     help="only include rows for these study_uuids (one per line). "
                          "build/rl.jsonl and build/eval.jsonl are already the clean "
                          "train/eval partition (zero study overlap, per CLAUDE.md's "
                          "data-pipeline section), so this is an optional extra filter, "
                          "not required for a normal train/val split.")
    ap.add_argument("--out", default="build/echoprime_grpo_train.parquet")
    ap.add_argument("--clip-h5", required=True,
                     help="Darya's clip_tokens_{train,test}.h5 -- MUST match --rl-jsonl's "
                          "split (clip_tokens_train.h5 for build/rl.jsonl, clip_tokens_test.h5 "
                          "for build/eval.jsonl)")
    ap.add_argument("--detr-h5", required=True,
                     help="Darya's {train,test}_detections*.h5, matching --clip-h5's split -- "
                          "note the real on-disk eval file is test_detections_merged.h5, not "
                          "test_detections.h5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                     help="shuffle rows before applying --limit, so a capped sample is a real "
                          "random sample instead of just file order (see CLAUDE.md's "
                          "data-pipeline section)")
    ap.add_argument("--one-per-study", action="store_true",
                     help="exactly one randomly-chosen verifiable QA pair per study, covering "
                          "every study in --rl-jsonl (after --study-list filtering and "
                          "verifiable-question-type filtering). Requires --shuffle-seed.")
    args = ap.parse_args(argv)

    if args.one_per_study and args.shuffle_seed is None:
        ap.error("--one-per-study requires --shuffle-seed")

    allowed = None
    if args.study_list:
        with open(args.study_list) as fh:
            allowed = {line.strip() for line in fh if line.strip()}

    with open(args.rl_jsonl) as fh:
        all_recs = [json.loads(line) for line in fh if line.strip()]

    skipped_split = 0
    skipped_qtype = 0
    candidates = []
    for rec in all_recs:
        if allowed is not None and rec["study_uuid"] not in allowed:
            skipped_split += 1
            continue
        if rec["question_type"] not in VERIFIABLE_QUESTION_TYPES:
            skipped_qtype += 1
            continue
        candidates.append(rec)

    if args.one_per_study:
        rng = random.Random(args.shuffle_seed)
        by_study: dict = {}
        for rec in candidates:
            by_study.setdefault(rec["study_uuid"], []).append(rec)
        candidates = [rng.choice(qs) for qs in by_study.values()]
        rng.shuffle(candidates)
    elif args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(candidates)

    if args.limit is not None:
        candidates = candidates[:args.limit]

    rows = []
    with h5py.File(args.clip_h5, "r") as clip_h5, h5py.File(args.detr_h5, "r") as detr_h5:
        for rec in candidates:
            rec = dict(rec)
            rec["dicoms_by_view"] = build_dicoms_by_view(rec["overview"]["views"], clip_h5)
            row = build_row(rec, clip_h5, detr_h5)
            assert row is not None  # already filtered by question_type above
            rows.append(row)

    write_parquet(rows, args.out)
    n_studies = len({r["extra_info"]["study_uuid"] for r in rows})
    print(f"[echoprime_track] wrote {len(rows)} rows ({n_studies} studies) -> {args.out} "
          f"(skipped {skipped_split} not-in-split, {skipped_qtype} not-verifiable-question-type "
          f"out of {len(all_recs)} total records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
