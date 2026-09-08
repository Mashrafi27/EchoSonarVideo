"""Native VeRL v0.7.1 tool adapter for the echo environment (HYBRID frame path).

UNRUN in this repo: imports verl.* which need the torch/vLLM runtime. Verified
here only by py_compile + contract review against the fetched verl-071 tree
(BaseTool/ToolResponse signatures in echo_env/INTEGRATION.md §0.2). GPU run = gate.

Tool observations return as IMAGES (ToolResponse(image=[...])) because v0.7.1's
ToolAgentLoop raises NotImplementedError on tool-returned video
(tool_agent_loop.py:335-340). The initial full-clip video reaches the model via
the dataset `videos` column, not this tool.
"""
import os
import time
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

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
        n = len(frames)
        text = obs.text or ""
        if n > 1:
            idx = ", ".join(str(f.frame_index) for f in obs.frames)
            text = (f"{text}\n[{n} frames shown as a {min(_SHEET_COLS, n)}-wide grid, "
                    f"row-major; frame indices: {idx}]").strip()
        self._log_call(op, t0, ok=True, note=f"{n} frames")
        return ToolResponse(image=[_contact_sheet(frames)], text=text), 0.0, {"success": True}

    def _log_call(self, op: str, t0: float, ok: bool, note: str) -> None:
        # Live per-tool-call timing -- lets us see where an episode's wall time
        # actually goes (frame I/O vs LLM generation) without waiting on a whole
        # rollout batch to finish. Cheap: one print per tool call.
        elapsed = time.monotonic() - t0
        print(f"[echo_tool pid={os.getpid()}] op={op} ok={ok} elapsed={elapsed:.2f}s {note}",
              flush=True)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
