from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.observation import Observation, FrameImg
from echo_env.budget import Budget
from echo_env.parse import parse_action
from echo_env.tools import select_view, select_frames, zoom
from echo_rl.data.frames import midframe


class EchoEnv:
    def __init__(self, cfg: EnvConfig, loader=None):
        self.cfg = cfg
        self.loader = loader or PILFrameLoader()
        self.manifest = None
        self.budget = None

    def reset(self, study_uuid: str) -> Observation:
        self.manifest = build_manifest(self.cfg.preprocessed_dir, study_uuid)
        self.budget = Budget(self.cfg)
        frames = []
        lines = []
        for entry in self.manifest.overview(self.cfg.n_overview_views):
            n = entry.frame_count
            i = midframe(n)
            img = self.loader.downscale(
                self.loader.load(entry.clip.frame_path(i)), self.cfg.preview_max_side)
            frames.append(FrameImg(image=img, view_name=entry.view_name,
                                   frame_index=i, kind="thumbnail"))
            lines.append(f"{entry.view_name}: {n} frames")
        text = "Available views:\n" + "\n".join(lines)
        return Observation(tool="reset", frames=frames, text=text)

    def _dispatch(self, name, args):
        if name == "select_view":
            return select_view(self.manifest, self.loader, self.cfg, args.get("view_name"))
        if name == "select_frames":
            return select_frames(self.manifest, self.loader, self.cfg,
                                 args.get("view_name"), args.get("indices", []))
        if name == "zoom":
            return zoom(self.manifest, self.loader, self.cfg, args.get("view_name"),
                        args.get("bbox"), args.get("frame_indices", []))
        return Observation.failure(name or "unknown", f"unknown tool '{name}'")

    def step(self, action_string: str):
        if self.manifest is None or self.budget is None:
            raise RuntimeError("EchoEnv.reset(study_uuid) must be called before step()")
        parsed = parse_action(action_string)
        if parsed.answer is not None:
            return "", 0.0, True, {"answer": parsed.answer}
        if not parsed.calls:
            info = {"tool_calls": self.budget.tool_calls,
                    "total_frames": self.budget.total_frames, "errors": parsed.errors}
            return Observation.failure("step", "no tool_call or answer found"), 0.0, False, info

        merged_frames = []
        texts = []
        errors = list(parsed.errors)
        for call in parsed.calls[: self.cfg.max_calls_per_turn]:
            if not self.budget.can_call():
                errors.append("tool budget exhausted; provide <answer>")
                break
            try:
                obs = self._dispatch(call["name"], call["arguments"])
            except Exception as e:
                # step() must never raise on model-produced input: any exception
                # escaping a tool for a malformed argument becomes a failure Observation.
                obs = Observation.failure(call["name"] or "step", f"tool error: {e}")
            if not obs.ok:
                errors.append(obs.error)
                # a failed call still counts as an attempt
                self.budget.register(call["name"], call["arguments"], obs)
                continue
            keep = obs.frames[: self.budget.frames_left()]
            obs.frames = keep
            # Budget.signature/seen/seen_before are recorded here (via register) but
            # not enforced -- dedup is not wired into step() in Phase 2. Whether to
            # penalize/skip repeated calls is a Phase-3 reward-shaping decision.
            self.budget.register(call["name"], call["arguments"], obs)
            if keep:
                merged_frames.extend(keep)
                texts.append(obs.text)

        info = {"tool_calls": self.budget.tool_calls,
                "total_frames": self.budget.total_frames, "errors": errors}
        if not merged_frames:
            msg = "; ".join(errors) or "no frames returned"
            return Observation.failure("step", msg), 0.0, False, info
        return (Observation(tool="step", frames=merged_frames, text="\n".join(texts)),
                0.0, False, info)
