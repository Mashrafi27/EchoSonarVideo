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

    def run(self, op: str, params: dict) -> Observation:
        params = params or {}
        if op == "select_view":
            return select_view(self.manifest, self.loader, self.cfg, params.get("view_name"))
        if op == "select_frames":
            return select_frames(self.manifest, self.loader, self.cfg,
                                 params.get("view_name"), params.get("indices", []))
        if op == "zoom":
            return zoom(self.manifest, self.loader, self.cfg, params.get("view_name"),
                        params.get("bbox"), params.get("frame_indices", []))
        return Observation.failure(op or "echo", f"unknown op {op!r}")
