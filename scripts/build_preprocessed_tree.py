"""Assemble the `<preprocessed_dir>/<study_uuid>/di-<dicom>_<View>/<n>.png` tree
that data_core / tool_env expect, from:

  1. the iCardio manifest parquet  (study_uuid -> [(dicom_uuid, view)], + split)
     default: /home/mashrafimonon/iCardio/output_with_labels/output/
              manifest_clinical_findings_with_eval_labels.parquet
  2. the per-dicom PNG frame dirs scattered across a handful of on-disk roots.

We only need the studies referenced by the two VQA files. For each such study we
look up its clips + view labels in the manifest, locate each clip's frame dir on
disk, and drop a symlink

    <out>/<study_uuid>/di-<dicom>_<View>/  ->  <found frame dir>

so nothing is copied. A JSON coverage report is written next to --out.

Run with the echojepav2 python (it has pyarrow):
  /home/mashrafimonon/miniconda3/envs/echojepav2/bin/python \
      scripts/build_preprocessed_tree.py --out /hdd2/<you>/echo_preprocessed
"""
import argparse
import collections
import json
import os
import sys
import time

DEFAULT_MANIFEST = ("/home/mashrafimonon/iCardio/output_with_labels/output/"
                    "manifest_clinical_findings_with_eval_labels.parquet")
DEFAULT_VQA_TRAIN = "/home/mashrafimonon/EchoSonarVideo/data/raw_vqa/train_vqa_with_thinking.jsonl"
DEFAULT_VQA_TEST = "/home/mashrafimonon/EchoSonarVideo/data/raw_vqa/test_vqa.jsonl"

# Roots that hold `di-XXXX-XXXX-XXXX/` (or `st-*/di-*_view/`) PNG frame dirs.
# Order = priority; first hit wins.
DEFAULT_ROOTS = [
    "/hdd2/ahmedaly/report_processed",          # already st-/di-_view, most specific
    "/data/ahmedaly/icardio",
    "/hdd2/ahmedaly/downloaded",
    "/hdd1/ahmedaly/downloaded",
    "/hdd2/ahmedaly/downloaded_not_standard",
    "/hdd2/ahmedaly/hdd2/ahmedaly/downloaded_not_standard_images",
    "/hdd2/ahmedaly/missing_segmentation",
    "/hdd2/ahmedaly/missing_eval_downloaded",
    "/hdd2/ahmedaly/missing_intersection_downloaded",
]


def load_vqa_studies(path):
    out = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(json.loads(line)["study_uuid"])
    return out


def load_manifest(path, want_studies):
    import pyarrow.parquet as pq
    cols = ["dicom_uuid", "study_uuid", "view", "split", "n_frames"]
    tbl = pq.read_table(path, columns=cols)
    by_study = collections.defaultdict(list)
    split_of = {}
    for r in tbl.to_pylist():
        s = r["study_uuid"]
        if s not in want_studies:
            continue
        by_study[s].append((r["dicom_uuid"], r["view"], r["n_frames"]))
        split_of.setdefault(s, r["split"])
    return by_study, split_of


def _has_png(d):
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith(".png"):
                    return True
    except OSError:
        pass
    return False


def build_disk_index(roots, cache_path):
    if cache_path and os.path.exists(cache_path):
        print(f"[index] loading cached {cache_path}")
        with open(cache_path) as fh:
            return json.load(fh)
    idx = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f"[index] skip (missing): {root}")
            continue
        t = time.time()
        n = 0
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if name.startswith("di-"):
                idx.setdefault(name.split("_", 1)[0], p)
                n += 1
            elif name.startswith("st-") and os.path.isdir(p):
                for c in os.listdir(p):
                    if c.startswith("di-"):
                        idx.setdefault(c.split("_", 1)[0], os.path.join(p, c))
                        n += 1
        print(f"[index] {root}: {n} di- entries ({time.time() - t:.0f}s)")
    print(f"[index] total distinct dicoms on disk: {len(idx)}")
    if cache_path:
        with open(cache_path, "w") as fh:
            json.dump(idx, fh)
        print(f"[index] cached -> {cache_path}")
    return idx


def sanitize_view(view):
    # data_core.data.views.canonical_view: collapse whitespace, keep casing.
    # Also strip characters that break a path component: "/" (nested dir),
    # plus "*"/":" which VAST and some tools reject.
    v = str(view or "")
    for bad in ("/", "\\", "*", ":"):
        v = v.replace(bad, " ")
    v = " ".join(v.split())
    return v or "Unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output preprocessed_dir to create")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--vqa-train", default=DEFAULT_VQA_TRAIN)
    ap.add_argument("--vqa-test", default=DEFAULT_VQA_TEST)
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--index-cache", default=None,
                    help="json path to cache the on-disk dicom index")
    ap.add_argument("--copy", action="store_true",
                    help="copy frames instead of symlinking (slow, large)")
    ap.add_argument("--verify-png", action="store_true",
                    help="only link a clip dir if it actually contains .png files")
    ap.add_argument("--dry-run", action="store_true",
                    help="report coverage, create nothing")
    args = ap.parse_args(argv)

    tr = load_vqa_studies(args.vqa_train)
    te = load_vqa_studies(args.vqa_test)
    want = tr | te
    print(f"[vqa] train {len(tr)} studies, test {len(te)} studies, union {len(want)}")

    by_study, split_of = load_manifest(args.manifest, want)
    print(f"[manifest] covers {len(by_study)}/{len(want)} studies "
          f"(train {len(set(by_study) & tr)}/{len(tr)}, "
          f"test {len(set(by_study) & te)}/{len(te)})")

    disk = build_disk_index(args.roots, args.index_cache)

    report = {
        "out": os.path.abspath(args.out),
        "studies_wanted": len(want),
        "studies_in_manifest": len(by_study),
        "per_study": {},
        "summary": {},
    }
    made_links = 0
    full = part = none = 0
    miss_dicoms = collections.Counter()

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    import shutil
    for study, clips in by_study.items():
        sdir = os.path.join(args.out, study)
        got, missing = [], []
        for dicom, view, _nf in clips:
            src = disk.get(dicom)
            if src and (not args.verify_png or _has_png(src)):
                got.append((dicom, view, src))
            else:
                missing.append(dicom)
                miss_dicoms[dicom] += 1
        if got and not missing:
            full += 1
        elif got:
            part += 1
        else:
            none += 1
        report["per_study"][study] = {
            "split_manifest": split_of.get(study),
            "split_vqa": "test" if study in te else "train",
            "clips_total": len(clips),
            "clips_found": len(got),
            "clips_missing": missing,
        }
        if args.dry_run or not got:
            continue
        os.makedirs(sdir, exist_ok=True)
        for dicom, view, src in got:
            # data_core.data.studies.index_study keeps dirs that start with "di-";
            # parse_clip_dirname splits on the FIRST "_" -> (di_id, view).
            dst = os.path.join(sdir, f"{dicom}_{sanitize_view(view)}")
            if os.path.lexists(dst):
                continue
            try:
                if args.copy:
                    shutil.copytree(src, dst)
                else:
                    os.symlink(src, dst)
                made_links += 1
            except OSError as e:
                print(f"[warn] link failed {study}/{dicom} ({view!r}): {e}")

    report["summary"] = {
        "studies_full": full,
        "studies_partial": part,
        "studies_none": none,
        "full_train": sum(1 for s in by_study
                          if s in tr and not report["per_study"][s]["clips_missing"]),
        "full_test": sum(1 for s in by_study
                         if s in te and not report["per_study"][s]["clips_missing"]),
        "links_created": made_links,
        "distinct_missing_dicoms": len(miss_dicoms),
    }
    rep_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            os.path.basename(args.out.rstrip("/")) + "_coverage.json")
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"[report] {rep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
