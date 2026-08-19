"""Per-trajectory echo session — the verl-free core behind EchoTool.

Mirrors echo_env.EchoEnv._dispatch's op→tool mapping, but standalone and
termination-agnostic: VeRL's ToolAgentLoop owns turns/termination, so this
core just dispatches one op and returns an Observation. Kept import-clean of
verl so it is fully unit-testable offline.
"""
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.observation import Observation
from echo_env.tools import select_view, select_frames, zoom


class EchoSession:
    def __init__(self, cfg, study_uuid, loader=None):
        self.cfg = cfg
        self.loader = loader or PILFrameLoader()
        self.manifest = build_manifest(cfg.preprocessed_dir, study_uuid)

    @staticmethod
    def _frame_indices(params: dict) -> list:
        """Read a frame-index list under either spelling.

        The tool schema names the same concept `indices` on select_frames but
        `frame_indices` on zoom, and models conflate the two -- observed in eval
        job 144199, where a select_frames call carrying `frame_indices` silently
        degraded to an empty list and failed with "no valid frame indices". That
        is our naming inconsistency, not a model error, and under GRPO it would
        spend a tool call and lose reward for it. Accepting both spellings is
        strictly more permissive: any call that worked before still works.
        """
        for key in ("indices", "frame_indices"):
            value = params.get(key)
            if value:
                return value
        return []

    def run(self, op: str, params: dict) -> Observation:
        params = params or {}
        if op == "select_view":
            return select_view(self.manifest, self.loader, self.cfg, params.get("view_name"))
        if op == "select_frames":
            return select_frames(self.manifest, self.loader, self.cfg,
                                 params.get("view_name"), self._frame_indices(params))
        if op == "zoom":
            # bbox may be absent/None; echo_env.normalize_bbox guards non-len-4/None -> failure Observation.
            return zoom(self.manifest, self.loader, self.cfg, params.get("view_name"),
                        params.get("bbox"), self._frame_indices(params))
        return Observation.failure(op or "echo", f"unknown op {op!r}")
