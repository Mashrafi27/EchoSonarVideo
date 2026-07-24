from echo_env.observation import Observation, FrameImg
from echo_env.bbox import normalize_bbox
from echo_rl.data.frames import evenly_spaced, midframe


def _unknown(tool, view_name, manifest):
    avail = ", ".join(manifest.view_names())
    return Observation.failure(tool, f"unknown view '{view_name}'; available: {avail}")


def select_view(manifest, loader, cfg, view_name) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("select_view", view_name, manifest)
    n = entry.frame_count
    idxs = evenly_spaced(n, cfg.n_preview_frames)
    frames = []
    for i in idxs:
        img = loader.downscale(loader.load(entry.clip.frame_path(i)), cfg.preview_max_side)
        frames.append(FrameImg(image=img, view_name=entry.view_name, frame_index=i, kind="preview"))
    text = f"{entry.view_name}: preview frames {idxs} of {n}"
    return Observation(tool="select_view", frames=frames, text=text)


def _clean_indices(indices, n):
    out = []
    for v in indices or []:
        if isinstance(v, bool):        # bool is a subclass of int; reject
            continue
        if isinstance(v, int) and 0 <= v < n:
            out.append(v)
    return sorted(set(out))


def select_frames(manifest, loader, cfg, view_name, indices) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("select_frames", view_name, manifest)
    n = entry.frame_count
    valid = _clean_indices(indices, n)[: cfg.n_highres_frames]
    if not valid:
        return Observation.failure(
            "select_frames", f"no valid frame indices for {entry.view_name} (0..{n-1})")
    frames = []
    for i in valid:
        img = loader.downscale(loader.load(entry.clip.frame_path(i)), cfg.highres_max_side)
        frames.append(FrameImg(image=img, view_name=entry.view_name, frame_index=i, kind="highres"))
    text = f"{entry.view_name}: frames {valid} of {n}"
    return Observation(tool="select_frames", frames=frames, text=text)


def zoom(manifest, loader, cfg, view_name, bbox, frame_indices) -> Observation:
    entry = manifest.resolve(view_name)
    if entry is None:
        return _unknown("zoom", view_name, manifest)
    n = entry.frame_count
    valid = _clean_indices(frame_indices, n)[: cfg.n_highres_frames]
    if not valid:
        valid = [midframe(n)]
    frames = []
    for i in valid:
        img = loader.load(entry.clip.frame_path(i))
        w, h = loader.size(img)
        nb = normalize_bbox(bbox, w, h, cfg.min_crop_side, cfg.max_aspect)
        if nb is None:
            return Observation.failure("zoom", f"invalid bbox {bbox}")
        # normalize_bbox guarantees nb is >= min_crop_side on both sides (or None),
        # so the crop is always at or above the patch floor -- no post-fix needed.
        crop = loader.crop(img, nb)
        frames.append(FrameImg(image=crop, view_name=entry.view_name,
                               frame_index=i, kind="crop", bbox=nb))
    text = f"{entry.view_name}: zoom {frames[0].bbox} on frames {valid} of {n}"
    return Observation(tool="zoom", frames=frames, text=text)
