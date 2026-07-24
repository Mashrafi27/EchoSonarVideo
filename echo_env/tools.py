from echo_env.observation import Observation, FrameImg
from echo_rl.data.frames import evenly_spaced


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
