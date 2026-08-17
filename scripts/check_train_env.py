#!/usr/bin/env python3
"""P3b smoke check: does the GPU/training environment match what echo_verl was coded against?

Run this FIRST in the training environment, before any SFT/GRPO launch:

    python scripts/check_train_env.py

Every 🟡 (authored-but-UNRUN) assumption in echo_verl is asserted here, so a
mismatch shows up as one red line instead of a crashed rollout an hour into a
job. Exit code 0 = all checks pass; 1 = at least one FAIL.

STATUS: authored offline, UNRUN against a real verl install.
"""
from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL_CONFIG = REPO / "echo_verl" / "configs" / "echo_tool_config.yaml"

# Run as `python scripts/check_train_env.py`, sys.path[0] is scripts/ -- so the repo
# is not importable unless we add it. The tool-registry check dynamically imports
# echo_verl.echo_tool by FQDN, so this must happen BEFORE any check runs.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_results: list[tuple[str, bool, str]] = []


def check(name):
    """Decorator: run the function, record PASS/FAIL, never abort the run."""
    def deco(fn):
        try:
            detail = fn() or ""
            _results.append((name, True, str(detail)))
        except Exception as e:
            _results.append((name, False, f"{type(e).__name__}: {e}"))
            if _VERBOSE:
                traceback.print_exc()
        return fn
    return deco


def _run_all() -> int:
    @check("transformers >= 4.57 (Qwen3-VL)")
    def _():
        import transformers
        from packaging.version import Version
        v = transformers.__version__
        assert Version(v) >= Version("4.57.0"), f"found {v}, need >=4.57.0"
        return v

    @check("verl importable + version")
    def _():
        import verl
        return getattr(verl, "__version__", "unknown")

    @check("BaseTool signatures match echo_verl/echo_tool.py")
    def _():
        from verl.tools.base_tool import BaseTool
        init = list(inspect.signature(BaseTool.__init__).parameters)
        assert init[:3] == ["self", "config", "tool_schema"], init
        for meth in ("create", "execute", "release"):
            assert hasattr(BaseTool, meth), f"BaseTool.{meth} missing"
        ex = list(inspect.signature(BaseTool.execute).parameters)
        assert ex[:3] == ["self", "instance_id", "parameters"], ex
        cr = list(inspect.signature(BaseTool.create).parameters)
        assert cr[:2] == ["self", "instance_id"], cr
        return "create/execute/release OK"

    @check("ToolResponse(image=[...]) constructs; non-list rejected")
    def _():
        from verl.tools.schemas import ToolResponse
        r = ToolResponse(image=["x"], text="t")
        assert r.image == ["x"] and not r.is_empty()
        try:
            ToolResponse(image="not-a-list")
        except Exception:
            return "list contract enforced"
        raise AssertionError("non-list image was accepted -- contract changed")

    @check("echo tool schema round-trips (op enum survives)")
    def _():
        import yaml
        from verl.tools.schemas import OpenAIFunctionToolSchema
        cfg = yaml.safe_load(TOOL_CONFIG.read_text())
        parsed = OpenAIFunctionToolSchema.model_validate(cfg["tools"][0]["tool_schema"])
        assert parsed.function.name == "echo", parsed.function.name
        op = parsed.function.parameters.properties["op"]
        assert op.enum == ["select_view", "select_frames", "zoom"], op.enum
        return f"props={list(parsed.function.parameters.properties)}"

    @check("tool registry instantiates EchoTool from the YAML")
    def _():
        from verl.tools.utils.tool_registry import initialize_tools_from_config
        tools = initialize_tools_from_config(str(TOOL_CONFIG))
        names = [t.name for t in tools]
        assert "echo" in names, names
        return f"tools={names}"

    @check("'tool_agent' agent loop is registered")
    def _():
        import verl.experimental.agent_loop.tool_agent_loop  # noqa: F401  (registers)
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        assert "tool_agent" in _agent_loop_registry, list(_agent_loop_registry)
        return "tool_agent present"

    @check("ToolAgentLoop still refuses tool-returned VIDEO (hybrid assumption)")
    def _():
        import verl.experimental.agent_loop.tool_agent_loop as m
        src = Path(inspect.getfile(m)).read_text()
        if "Multimedia type 'video' is not currently supported" in src:
            return "still image-only -> hybrid frame path REQUIRED (as designed)"
        return "WARNING: video branch changed -- revisit hybrid decision in INTEGRATION.md §0.2"

    @check("Qwen3-VL processor + rope index available")
    def _():
        from transformers import AutoProcessor  # noqa: F401
        from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401
        from verl.models.transformers.qwen3_vl import get_rope_index  # noqa: F401
        return "qwen3_vl OK"

    @check("pyarrow available for parquet data-gen")
    def _():
        import pyarrow
        return pyarrow.__version__

    @check("echo_verl imports (session/reward/generate_trainset/echo_tool)")
    def _():
        import echo_verl.echo_tool  # noqa: F401
        import echo_verl.generate_trainset  # noqa: F401
        import echo_verl.reward  # noqa: F401
        import echo_verl.session  # noqa: F401
        return "all four import"

    @check("custom_reward_function entrypoint signature")
    def _():
        from echo_verl.reward import compute_score
        params = list(inspect.signature(compute_score).parameters)
        assert params[:3] == ["data_source", "solution_str", "ground_truth"], params
        return ", ".join(params)

    width = max(len(n) for n, _, _ in _results)
    failed = 0
    for name, ok, detail in _results:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{tag}] {name.ljust(width)}  {detail}")
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true", help="print tracebacks for failures")
    args = ap.parse_args()
    _VERBOSE = args.verbose
    raise SystemExit(_run_all())
else:
    _VERBOSE = False
