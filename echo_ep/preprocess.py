"""Reproduce EchoPrime's own preprocessing exactly, sourcing frames from OUR preprocessed
PNG-per-frame tree (`ECHO_PREPROCESSED_DIR/<study_uuid>/di-*_<View>/<n>.png`) instead of
their `process_mp4s`/`process_dicoms` (which decode a video file). Frame-index-based, not
fps-based -- their code takes raw frame indices [start:start+32:stride] from whatever frame
sequence it's handed, so sourcing from our already-extracted, sequentially-numbered PNGs is
equivalent, not an approximation, as long as playback order is preserved (it is: filenames
are `0.png, 1.png, ...` in native frame order -- echo_rl/data/studies.py:index_study).

Ported verbatim from /home/mashrafimonon/Mashrafi/EchoPrime (echo_prime/model.py's
process_mp4s + utils/utils.py's crop_and_scale) -- see echo_ep/__init__.py for why this is
ported rather than imported from that external, unversioned path. Constants (frame count,
stride, resolution, per-channel mean/std) are EchoPrime's own trained-on values; changing
them would feed the frozen encoder out-of-distribution input.
"""
import glob
import os

import cv2
import numpy as np
import torch

FRAMES_TO_TAKE = 32
FRAME_STRIDE = 2
VIDEO_SIZE = 224
# On the 0-255 pixel scale (NOT divided by 255 first) -- matches EchoPrime's own
# process_mp4s/process_dicoms exactly (echo_prime/model.py:129-130,185-186).
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)


def crop_and_scale(img: np.ndarray, res=(VIDEO_SIZE, VIDEO_SIZE),
                    interpolation=cv2.INTER_CUBIC, zoom: float = 0.1) -> np.ndarray:
    """Center-crop to the target aspect ratio, then a fixed 10% zoom-in crop, then resize.

    Verbatim port of EchoPrime's utils.utils.crop_and_scale -- handles our preprocessed PNGs'
    variable native resolutions/aspect ratios (confirmed non-square: 636x422, 880x650, etc.)
    the same way EchoPrime's own training data was cropped.
    """
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding] if padding > 0 else img
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding] if padding > 0 else img
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]
    return cv2.resize(img, res, interpolation=interpolation)


def load_clip_frames(clip_dir: str, max_frames: int = FRAMES_TO_TAKE) -> list:
    """Native-order PNG frames for one view/clip directory, as HxWx3 uint8 RGB arrays.

    Only reads the first `max_frames` files -- clip_to_tensor's [0:32:2] slice never looks
    past index 32 (start=0, hardcoded upstream), so clips with 50-90 native frames (real
    range seen in this data) were wastefully reading 2-3x more PNGs than ever get used.
    That was the actual bottleneck in the first full-scale timing pass (~75-150s/study).
    """
    paths = glob.glob(os.path.join(clip_dir, "*.png"))
    # Filenames are bare frame indices ("0.png", "1.png", ...) -- sort numerically, not
    # lexically (lexical would put "10.png" before "2.png").
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    paths = paths[:max_frames]
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return frames


def clip_to_tensor(clip_dir: str) -> torch.Tensor:
    """One view/clip directory -> a (3, 16, 224, 224) float tensor, EchoPrime-normalized.

    Mirrors echo_prime/model.py's process_mp4s body exactly: crop+scale every native frame,
    normalize, zero-pad to FRAMES_TO_TAKE if short, then take [0:32:2] -> 16 frames.
    """
    frames = load_clip_frames(clip_dir)
    if not frames:
        raise ValueError(f"no frames found in {clip_dir}")
    x = np.zeros((len(frames), VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        x[i] = crop_and_scale(f)
    x = torch.as_tensor(x, dtype=torch.float).permute(3, 0, 1, 2)  # (3, T, 224, 224)
    x.sub_(MEAN).div_(STD)
    if x.shape[1] < FRAMES_TO_TAKE:
        padding = torch.zeros((3, FRAMES_TO_TAKE - x.shape[1], VIDEO_SIZE, VIDEO_SIZE),
                               dtype=torch.float)
        x = torch.cat((x, padding), dim=1)
    start = 0  # EchoPrime's own code hardcodes start=0 -- no random window, see model.py:146,202
    return x[:, start:start + FRAMES_TO_TAKE:FRAME_STRIDE, :, :]  # (3, 16, 224, 224)


def study_to_tensor(study_dir: str) -> tuple:
    """A study's preprocessed-tree directory -> (stack_of_videos, view_names).

    stack_of_videos: (N, 3, 16, 224, 224) float tensor, N = number of view/clip dirs.
    view_names: list[str], same order as the first dim of stack_of_videos, parsed from each
    `di-<dicom_uuid>_<View>` directory name (echo_rl/data/studies.py's own convention).
    """
    clip_dirs = sorted(
        d for d in glob.glob(os.path.join(study_dir, "di-*")) if os.path.isdir(d))
    tensors, view_names = [], []
    for clip_dir in clip_dirs:
        base = os.path.basename(clip_dir)
        view = base.split("_", 1)[1] if "_" in base else base
        try:
            tensors.append(clip_to_tensor(clip_dir))
            view_names.append(view)
        except Exception as e:  # a handful of clips are corrupt/empty -- skip, don't fail the study
            print(f"[echo_ep] skipping {clip_dir}: {type(e).__name__}: {e}", flush=True)
    if not tensors:
        raise ValueError(f"no usable clips in {study_dir}")
    return torch.stack(tensors), view_names
