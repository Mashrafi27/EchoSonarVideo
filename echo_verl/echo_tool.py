"""Native VeRL v0.7.1 tool adapter for the echo environment (HYBRID frame+video path).

RUN, on the CUDA box: confirmed clean through 160+ real GRPO training steps as of
2026-09-10 (docs/OPEN_ISSUES.md), video-on by default (ECHO_SELECT_VIEW_VIDEO). The
NCCL hang that blocked this for two sessions was fsdp_size (ZeRO-3 param sharding
re-gathering a data-dependent number of times per rank), not this file -- see the
gate comment on execute()'s select_view branch and external/verl-video-nccl-fix.patch.

select_view returns the chosen view's clip as VIDEO -- motion matters for cardiac
findings (wall motion, regurgitation, EF) and a still contact sheet can't carry it.
select_frames/zoom stay IMAGES: those ops are about a specific instant or region,
not motion, and verl's chat-template only allows one media placeholder per
ToolResponse anyway (tool_agent_loop.py:307-314, same constraint _contact_sheet
below works around for images).

verl v0.7.1 upstream ships `raise NotImplementedError` on ANY tool-returned video
(tool_agent_loop.py, `_handle_processing_tools_state`). PATCHED there (not
forked -- a real, first deviation from "pinned verl, not forked", tracked in
docs/OPEN_ISSUES.md) to thread video through the same way images already were,
confirmed empirically 2026-09-09 that Qwen3-VL's own processor does NOT silently
drop frames from a real per-view clip (that old failure was from feeding a
19-DIFFERENT-VIEWS menu through the video path, not one view's own frames).
"""
import os
import time
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.vision_utils import process_video

from echo_env.config import EnvConfig
from echo_verl.session import EchoSession

_ERR_REWARD = -0.05   # small per-step shaping penalty on a failed tool call
_SHEET_COLS = 4       # frames per row in the contact sheet
_SHEET_MAX_W = 768    # cap composed width -- vision tokens ~= (W/14)*(H/14),
                      # and every tool turn's sheet is re-encoded into the context


def _contact_sheet(frames: list) -> Image.Image:
    """Compose >=1 PIL frames into ONE image.

    verl v0.7.1's ToolAgentLoop (tool_agent_loop.py:308) inserts exactly ONE
    `<image>` placeholder per ToolResponse but forwards every element of
    `ToolResponse.image` as multimodal data -- a list of N images then fails
    vLLM's prompt-replacement with "mm_items['image'][k]". So the echo tool hands
    back a single grid image instead of a list of frames.
    """
    frames = [f for f in frames if f is not None]
    if len(frames) == 1:
        return frames[0]
    cols = min(_SHEET_COLS, len(frames))
    rows = (len(frames) + cols - 1) // cols
    cw = max(f.width for f in frames)
    ch = max(f.height for f in frames)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet.paste(f, (c * cw, r * ch))
    if sheet.width > _SHEET_MAX_W:
        s = _SHEET_MAX_W / sheet.width
        sheet = sheet.resize((_SHEET_MAX_W, max(1, round(sheet.height * s))))
    return sheet


class EchoTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)   # sets self.name = tool_schema.function.name
        self._instances: dict[str, EchoSession] = {}
        # EnvConfig from environment; per-tool config-block overrides are a future enhancement.
        self._cfg = EnvConfig.from_env()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        study_uuid = create_kwargs.get("study_uuid")
        self._instances[instance_id] = EchoSession(self._cfg, study_uuid)
        print(f"[echo_tool pid={os.getpid()}] episode start study={study_uuid} "
              f"in_flight={len(self._instances)}", flush=True)
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any],
                      **kwargs) -> tuple[ToolResponse, float, dict]:
        session = self._instances.get(instance_id)
        if session is None:
            return ToolResponse(text="echo session not initialized"), _ERR_REWARD, {"success": False}
        op = parameters.get("op")
        t0 = time.monotonic()
        # echo_env tools turn logical failures into Observation.failure, but I/O-level
        # errors (e.g. FileNotFoundError from loader.load) raise. Catch here so ALL
        # failure classes get this tool's _ERR_REWARD shaping rather than the agent
        # loop's generic reward=0.0 fallback (tool_agent_loop.py:436).
        try:
            obs = session.run(op, parameters)
        except Exception as e:
            self._log_call(op, t0, ok=False, note=f"raised: {e}")
            return ToolResponse(text=f"echo tool error: {e}"), _ERR_REWARD, {"success": False}
        if not obs.ok:
            self._log_call(op, t0, ok=False, note=obs.error)
            return ToolResponse(text=obs.error), _ERR_REWARD, {"success": False}
        frames = [f.image for f in obs.frames]
        if not frames:
            self._log_call(op, t0, ok=True, note="no frames")
            return ToolResponse(text=obs.text or "no frames"), 0.0, {"success": True}

        # ECHO_SELECT_VIEW_VIDEO gate: default ON as of 2026-09-10. Root cause of
        # the earlier NCCL ALLGATHER hang (ranks landing on different collective
        # counts, gap of 7 enqueued ops) found: FSDP's ZeRO-3 sharding re-gathers
        # each wrapped submodule's params on every forward call, and Qwen3-VL's
        # forward calls its vision/video encoder a DATA-DEPENDENT number of times
        # (whether a micro-batch's samples carry video content varies per rank).
        # Fixed by actor_rollout_ref.actor.fsdp_config.fsdp_size=1 (DDP-style: full
        # params replicated per GPU, no per-module shard all-gather to desync) --
        # see external/verl-video-nccl-fix.patch for the accompanying verl patch
        # (one upstream reshard() call asserts a sharded strategy; skipped under
        # fsdp_size=1). Confirmed clean for 160+ steps of real training
        # (docs/OPEN_ISSUES.md). Env var kept as an escape hatch for debugging.
        if op == "select_view" and os.environ.get("ECHO_SELECT_VIEW_VIDEO", "1") == "1":
            try:
                video_tensor, video_meta, n_frames = self._view_as_video(session, obs.frames[0].view_name)
            except Exception as e:
                # Fall back to the old still contact-sheet rather than fail the
                # whole tool call over a video-build problem -- see _view_as_video.
                self._log_call(op, t0, ok=True, note=f"video build failed ({e}), fell back to stills")
                return ToolResponse(image=[_contact_sheet(frames)], text=obs.text or ""), 0.0, {"success": True}
            text = f"{obs.text or ''}\n[{n_frames} frames of the clip, in order]".strip()
            self._log_call(op, t0, ok=True, note=f"video, {n_frames} frames")
            return ToolResponse(video=[(video_tensor, video_meta)], text=text), 0.0, {"success": True}

        n = len(frames)
        text = obs.text or ""
        if n > 1:
            idx = ", ".join(str(f.frame_index) for f in obs.frames)
            text = (f"{text}\n[{n} frames shown as a {min(_SHEET_COLS, n)}-wide grid, "
                    f"row-major; frame indices: {idx}]").strip()
        self._log_call(op, t0, ok=True, note=f"{n} frames")
        return ToolResponse(image=[_contact_sheet(frames)], text=text), 0.0, {"success": True}

    def _view_as_video(self, session: EchoSession, view_name: str):
        """Build a (tensor, metadata) tuple -- verl's own video shape -- from a
        CONTINUOUS window of n_video_frames at the start of the resolved view's
        clip. Continuous, not evenly-spaced: frames were preprocessed at a fixed
        capture rate (video_fps), so a continuous window is a real, fixed-duration
        slice of the clip, not an arbitrary subsample -- see EnvConfig's comment.
        Passes `sample_fps` explicitly so verl's process_video reports the TRUE
        rate in the returned metadata, not its own hardcoded 2.0 default; Qwen3-VL
        uses that value to build its interleaved per-frame timestamp text, so a
        wrong fps here means the model is told the wrong elapsed time between
        frames even though the frame content itself is correct.

        view_name is the ALREADY-RESOLVED name (obs.frames[0].view_name), not the
        model's raw query string.
        """
        entry = session.manifest.resolve(view_name)
        if entry is None:
            raise ValueError(f"view {view_name!r} vanished between select_view and video build")
        n_total = entry.frame_count
        k = min(session.cfg.n_video_frames, n_total)
        idxs = list(range(k))  # continuous window from the start, not evenly_spaced

        # Resize+center-crop every frame to an EXACT fixed square before building
        # the tensor. Bounding just the longest side (like the still-image path
        # does) still lets different aspect ratios produce different final H/W --
        # confirmed 2026-09-09: A4C (280x392) vs PLAX Standard (336x336) after a
        # longest-side-only downscale, still not uniform. Every view/study must
        # produce an IDENTICAL tensor shape: different ranks otherwise process
        # genuinely different-shaped batches, and FSDP's collectives (every rank
        # must call them in the same order/count) desync and hang -- also
        # confirmed 2026-09-09, NCCL collective timeout, ranks off by exactly one
        # enqueued op, on the first real multi-GPU run after this tool started
        # returning video.
        side = session.cfg.highres_max_side
        frames = []
        for i in idxs:
            img = session.loader.load(entry.clip.frame_path(i))
            w, h = session.loader.size(img)
            scale = side / min(w, h)
            new_w, new_h = max(side, round(w * scale)), max(side, round(h * scale))
            img = img.resize((new_w, new_h))
            left, top = (new_w - side) // 2, (new_h - side) // 2
            frames.append(img.crop((left, top, left + side, top + side)))
        video_dict = {
            "video": frames,  # list[PIL.Image] -- fetch_image inside process_video accepts these directly
            "sample_fps": session.cfg.video_fps,
        }
        tensor, meta = process_video(video_dict, return_video_metadata=True)
        return tensor, meta, len(idxs)

    def _log_call(self, op: str, t0: float, ok: bool, note: str) -> None:
        # Live per-tool-call timing -- lets us see where an episode's wall time
        # actually goes (frame I/O vs LLM generation) without waiting on a whole
        # rollout batch to finish. Cheap: one print per tool call.
        elapsed = time.monotonic() - t0
        print(f"[echo_tool pid={os.getpid()}] op={op} ok={ok} elapsed={elapsed:.2f}s {note}",
              flush=True)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
