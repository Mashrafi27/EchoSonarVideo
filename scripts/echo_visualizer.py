"""Seeded random clip browser over `preprocessed_by_alikhan_for_echojepa_grouped_*`.

Picks one clip (optionally filtered by dicom mode and/or view), renders a frame
filmstrip plus a metadata + report sidebar into one PNG, and prints the same
info to stdout. Read-only -- never modifies the source shards.

Why seeded, not fully random: same --seed always resolves to the same clip, so
"look at this one again" is reproducible without saving state anywhere.

Usage:
    ./.venv-train/bin/python scripts/echo_visualizer.py --seed 42
    ./.venv-train/bin/python scripts/echo_visualizer.py --seed 42 --mode color --view A4C
    ./.venv-train/bin/python scripts/echo_visualizer.py --seed 7 --mode standard --view PLAX \\
        --out .tmp_work/echo_visualizer/plax_7.png

--mode: standard | color | images (maps to the three top-level shard dirs).
--view: case-insensitive substring match against the sample's "view" metadata
        field (e.g. "A4C", "PLAX", "PSAX Mitral"); omit to accept any view.

How selection works (kept fast on a 4000+-shard dataset): the seed drives a
Python Random instance that picks a shard, whose members get scanned (tar
header list, no frame data touched) and whose every sample's metadata.json is
read to check the view filter; the same RNG picks among the matches. If a
shard has no match, the RNG draws ANOTHER random shard (not shard+1) -- this
"grouped" reorg clusters shards by something correlated with view (confirmed:
26 consecutive standard_videos shards with zero A4C samples), so trying
neighbours wastes tries on data as skewed as whatever produced the miss.
Up to --max-shard-tries draws. Only matching shards' JSON gets read --
frames.npz for the losing candidates is never opened.
"""
import argparse
import io
import json
import os
import random
import tarfile
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = "/vast/users/mohammad.yaqub/project/preprocessed_by_alikhan_for_echojepa_grouped_20260513_fullrerun"
MODE_DIRS = {"standard": "standard_videos", "color": "color_doppler_videos", "images": "images"}

REPORT_FIELDS = [
    "left_ventricle", "right_ventricle", "left_atrium", "right_atrium",
    "aortic_valve", "mitral_valve", "tricuspid_valve", "pulmonic_valve",
    "pericardium", "aortic_root", "aortic_arch", "pulmonary_artery",
    "conclusions",
]
META_FIELDS = [
    "dicom_uuid", "study_uuid", "view", "dicom_type", "manufacturer",
    "original_fps", "target_fps", "n_original_frames", "n_output_frames",
    "original_spacing_mm", "target_spacing_mm", "target_size",
    "study_designation", "age_at_visit", "ejection_fraction",
]

SIDEBAR_W = 560
FONT_SIZE = 15
LINE_H = 19
MARGIN = 16
GRID_COLS = 4
GRID_ROWS = 4
CELL = 168


def _font(size=FONT_SIZE, bold=False):
    # DejaVu ships with matplotlib/Pillow's own font dir on most envs; fall back
    # to Pillow's built-in bitmap font if not found rather than erroring.
    names = (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick_shard_list(mode: str) -> list:
    d = os.path.join(DATA_ROOT, MODE_DIRS[mode])
    shards = sorted(f for f in os.listdir(d) if f.startswith("shard-") and f.endswith(".tar"))
    if not shards:
        raise SystemExit(f"no shards found under {d}")
    return [os.path.join(d, f) for f in shards]


def find_candidates(shard_path: str, view_filter: str):
    """Return [(dicom_uuid, metadata_dict), ...] in this shard matching view_filter
    (case-insensitive substring, or all samples if view_filter is None)."""
    out = []
    with tarfile.open(shard_path, "r") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".metadata.json"):
                continue
            dicom_uuid = member.name[: -len(".metadata.json")]
            meta = json.loads(tf.extractfile(member).read())
            view = str(meta.get("view", ""))
            if view_filter is None or view_filter.lower() in view.lower():
                out.append((dicom_uuid, meta))
    return out


def select_clip(mode: str, seed: int, view_filter: str, max_tries: int):
    shards = pick_shard_list(mode)
    rng = random.Random(seed)
    tried_idxs = []
    for _ in range(max_tries):
        idx = rng.randrange(len(shards))
        tried_idxs.append(idx)
        candidates = find_candidates(shards[idx], view_filter)
        if candidates:
            dicom_uuid, meta = candidates[rng.randrange(len(candidates))]
            return shards[idx], dicom_uuid, meta
    raise SystemExit(
        f"no clip matching view={view_filter!r} found across {max_tries} random shards "
        f"{tried_idxs} (mode={mode}). Try a broader --view, a different --seed, or "
        f"raise --max-shard-tries.")


def load_frames(shard_path: str, dicom_uuid: str) -> np.ndarray:
    with tarfile.open(shard_path, "r") as tf:
        member = tf.getmember(f"{dicom_uuid}.frames.npz")
        buf = io.BytesIO(tf.extractfile(member).read())
    return np.load(buf)["frames"]


def build_filmstrip(frames: np.ndarray) -> Image.Image:
    n = frames.shape[0]
    n_cells = GRID_COLS * GRID_ROWS
    if n <= n_cells:
        idxs = list(range(n)) + [n - 1] * (n_cells - n)  # pad by repeating last frame
    else:
        idxs = np.linspace(0, n - 1, n_cells).round().astype(int).tolist()

    grid = Image.new("RGB", (GRID_COLS * CELL, GRID_ROWS * CELL), (10, 10, 10))
    draw = ImageDraw.Draw(grid)
    font_small = _font(12)
    for cell, fi in enumerate(idxs):
        r, c = divmod(cell, GRID_COLS)
        tile = Image.fromarray(frames[fi]).resize((CELL, CELL))
        grid.paste(tile, (c * CELL, r * CELL))
        draw.text((c * CELL + 4, r * CELL + 2), f"f{fi}", font=font_small, fill=(0, 255, 120))
    return grid


def wrap_kv(draw, x, y, label, value, width_px, font_label, font_value, max_w_chars=58):
    draw.text((x, y), f"{label}:", font=font_label, fill=(120, 190, 255))
    y += LINE_H
    text = "-" if value in (None, "") else str(value)
    for line in textwrap.wrap(text, max_w_chars) or ["-"]:
        draw.text((x + 10, y), line, font=font_value, fill=(230, 230, 230))
        y += LINE_H
    return y + 4


def build_sidebar(meta: dict, min_height: int) -> Image.Image:
    # Allocate generously tall up front -- PIL silently clips draws past an image's
    # actual bounds (no error), so cropping UP after the fact can't recover text that
    # was never rendered. Crop DOWN to the real content height once we know it.
    sb = Image.new("RGB", (SIDEBAR_W, max(min_height, 4000)), (18, 20, 24))
    draw = ImageDraw.Draw(sb)
    font_h = _font(18, bold=True)
    font_l = _font(FONT_SIZE, bold=True)
    font_v = _font(FONT_SIZE)

    x, y = MARGIN, MARGIN
    draw.text((x, y), "METADATA", font=font_h, fill=(255, 210, 120))
    y += LINE_H + 6
    for f in META_FIELDS:
        y = wrap_kv(draw, x, y, f, meta.get(f), SIDEBAR_W, font_l, font_v)

    y += 10
    draw.line((x, y, SIDEBAR_W - MARGIN, y), fill=(70, 70, 70))
    y += 14
    draw.text((x, y), "REPORT", font=font_h, fill=(255, 210, 120))
    y += LINE_H + 6
    for f in REPORT_FIELDS:
        val = meta.get(f)
        if not val:
            continue
        y = wrap_kv(draw, x, y, f, val, SIDEBAR_W, font_l, font_v)

    return sb.crop((0, 0, SIDEBAR_W, max(y + MARGIN, min_height)))


def render(shard_path, dicom_uuid, meta, frames) -> Image.Image:
    filmstrip = build_filmstrip(frames)
    sidebar = build_sidebar(meta, filmstrip.height)
    canvas = Image.new("RGB", (filmstrip.width + sidebar.width, max(filmstrip.height, sidebar.height)),
                        (10, 10, 10))
    canvas.paste(filmstrip, (0, 0))
    canvas.paste(sidebar, (filmstrip.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, filmstrip.height - 18), os.path.basename(shard_path), font=_font(11),
               fill=(140, 140, 140))
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=list(MODE_DIRS), default="standard")
    ap.add_argument("--view", default=None, help="case-insensitive substring filter, e.g. 'A4C'")
    ap.add_argument("--max-shard-tries", type=int, default=25)
    ap.add_argument("--out", default=None, help="default: .tmp_work/echo_visualizer/<mode>_seed<seed>.png")
    args = ap.parse_args(argv)

    shard_path, dicom_uuid, meta = select_clip(args.mode, args.seed, args.view, args.max_shard_tries)
    frames = load_frames(shard_path, dicom_uuid)

    print(f"[echo_visualizer] shard={os.path.basename(shard_path)} dicom={dicom_uuid} "
          f"view={meta.get('view')!r} type={meta.get('dicom_type')!r} frames={frames.shape}")
    print(json.dumps({k: meta.get(k) for k in META_FIELDS}, indent=2))
    report = {k: meta.get(k) for k in REPORT_FIELDS if meta.get(k)}
    if report:
        print(json.dumps(report, indent=2))

    out_path = args.out or f".tmp_work/echo_visualizer/{args.mode}_seed{args.seed}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    render(shard_path, dicom_uuid, meta, frames).save(out_path)
    print(f"[echo_visualizer] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
