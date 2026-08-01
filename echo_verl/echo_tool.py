"""Native VeRL v0.7.1 tool adapter for the echo environment (HYBRID frame path).

UNRUN in this repo: imports verl.* which need the torch/vLLM runtime. Verified
here only by py_compile + contract review against the fetched verl-071 tree
(BaseTool/ToolResponse signatures in echo_env/INTEGRATION.md §0.2). GPU run = gate.

Tool observations return as IMAGES (ToolResponse(image=[...])) because v0.7.1's
ToolAgentLoop raises NotImplementedError on tool-returned video
(tool_agent_loop.py:335-340). The initial full-clip video reaches the model via
the dataset `videos` column, not this tool.
"""
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from echo_env.config import EnvConfig
from echo_verl.session import EchoSession

_ERR_REWARD = -0.05   # small per-step shaping penalty on a failed tool call


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
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any],
                      **kwargs) -> tuple[ToolResponse, float, dict]:
        session = self._instances.get(instance_id)
        if session is None:
            return ToolResponse(text="echo session not initialized"), _ERR_REWARD, {"success": False}
        op = parameters.get("op")
        # echo_env tools turn logical failures into Observation.failure, but I/O-level
        # errors (e.g. FileNotFoundError from loader.load) raise. Catch here so ALL
        # failure classes get this tool's _ERR_REWARD shaping rather than the agent
        # loop's generic reward=0.0 fallback (tool_agent_loop.py:436).
        try:
            obs = session.run(op, parameters)
        except Exception as e:
            return ToolResponse(text=f"echo tool error: {e}"), _ERR_REWARD, {"success": False}
        if not obs.ok:
            return ToolResponse(text=obs.error), _ERR_REWARD, {"success": False}
        images = [f.image for f in obs.frames]
        return ToolResponse(image=images, text=obs.text), 0.0, {"success": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
