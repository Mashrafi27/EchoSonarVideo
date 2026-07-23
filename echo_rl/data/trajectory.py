from echo_rl.data.trace import parse_thinking, findings_text
from echo_rl.data.views import canonical_view
from echo_rl.data.frames import evenly_spaced, midframe


def _trace_view_key(trace_view: str) -> str:
    v = trace_view.strip()
    if v.endswith(" View"):
        v = v[: -len(" View")]
    return canonical_view(v)


def overview_turn(clips: list, cfg) -> dict:
    views = []
    for c in clips[: cfg.n_overview_views]:
        views.append({"view": c.view,
                      "frame": c.frame_path(midframe(c.frame_count)),
                      "frame_count": c.frame_count})
    return {"type": "overview", "views": views}


def build_trajectory(question: str, answer: str, thinking: str, clips: list, cfg,
                     max_views: int = 4) -> dict:
    by_view = {canonical_view(c.view): c for c in clips}
    turns = []
    used = 0
    for trace_view, body in parse_thinking(thinking):
        key = _trace_view_key(trace_view)
        clip = by_view.get(key)
        if clip is None:
            continue
        idxs = evenly_spaced(clip.frame_count, cfg.n_preview_frames)
        turns.append({"type": "tool", "name": "select_view",
                      "args": {"view": clip.view},
                      "obs": {"view": clip.view,
                              "frames": [clip.frame_path(i) for i in idxs]}})
        turns.append({"type": "think", "text": findings_text(body)})
        used += 1
        if used >= max_views:
            break
    return {"overview": overview_turn(clips, cfg), "turns": turns, "answer": answer}
