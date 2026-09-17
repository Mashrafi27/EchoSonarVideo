#!/usr/bin/env python3
"""P3b smoke check: does the GPU/training environment match what verl_bridge was coded against?

Run this FIRST in the training environment, before any SFT/GRPO launch:

    python scripts/check_train_env.py

Every 🟡 (authored-but-UNRUN) assumption in verl_bridge is asserted here, so a
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
TOOL_CONFIG = REPO / "packages" / "verl_bridge" / "configs" / "echo_tool_config.yaml"

# Run as `python scripts/check_train_env.py`, sys.path[0] is scripts/ -- so the repo
# is not importable unless we add it. The tool-registry check dynamically imports
# verl_bridge.echo_tool by FQDN, so this must happen BEFORE any check runs.
if str(REPO / "packages") not in sys.path:
    sys.path.insert(0, str(REPO / "packages"))

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

    @check("BaseTool signatures match packages/verl_bridge/echo_tool.py")
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

    @check("ToolAgentLoop carries the EchoSonarVideo video patch")
    def _():
        # As of 2026-09-10 select_view returns real video by default (echo_tool.py),
        # which needs this local patch to external/verl (see
        # external/verl-video-nccl-fix.patch, applied via `git apply` after a fresh
        # `git submodule update --init` -- it does not survive that on its own).
        # Stock verl still ships `raise NotImplementedError` here; if this check
        # fails, either the patch was lost or verl was re-synced to a fresh
        # checkout, and select_view's video path will silently misbehave.
        import verl.experimental.agent_loop.tool_agent_loop as m
        src = Path(inspect.getfile(m)).read_text()
        assert "Multimedia type 'video' is not currently supported" not in src, (
            "stock verl (no video patch) -- apply external/verl-video-nccl-fix.patch")
        assert "new_videos_this_turn" in src, "patch marker missing"
        return "video patch present"

    @check("verl's rollout registry carries the EchoSonarVideo hf-rollout patch")
    def _():
        # The EchoPrime+Qwen3-8B-text GRPO track (packages/verl_bridge/configs/echoprime_grpo.yaml) needs
        # rollout.name=hf to resolve -- stock verl's _ROLLOUT_REGISTRY never mapped it to a
        # class at all (see external/verl-hf-rollout-registry.patch, applied via `git apply`
        # after a fresh `git submodule update --init` -- it does not survive that on its own).
        # If this check fails, either the patch was lost or verl was re-synced to a fresh
        # checkout, and echoprime_grpo.yaml's rollout will fail its registry assert at startup.
        import verl.workers.rollout.base as m
        src = Path(inspect.getfile(m)).read_text()
        assert '("hf", "async")' in src, (
            "stock verl (no hf-rollout patch) -- apply external/verl-hf-rollout-registry.patch")
        assert "echoprime_track.hf_rollout_video.HFRolloutVideo" in src, "patch marker missing"

        # Same patch file also adds the one line fsdp_workers.py needs to actually bind the
        # live FSDP-wrapped module into the rollout instance after construction -- confirmed
        # necessary this session: HFRollout.__init__'s own signature doesn't match how
        # _build_rollout constructs it, and nothing else in that function ever sets this.
        import verl.workers.fsdp_workers as fw
        fw_src = Path(inspect.getfile(fw)).read_text()
        assert "self.rollout.module = self.actor_module_fsdp" in fw_src, (
            "stock verl (no hf-rollout patch) -- apply external/verl-hf-rollout-registry.patch")
        return "hf-rollout registry patch present"

    @check("verl carries the EchoSonarVideo vLLM-serving patches")
    def _():
        # The real vLLM track (packages/verl_bridge/configs/echoprime_grpo.yaml, rollout.name=vllm) needs
        # two more things stock verl doesn't do on its own: (1) our custom vLLM model class
        # actually registered before the engine builds, (2) the training-side multi_modal_inputs
        # not silently dropped just because our text-only model has no real HF processor. Same
        # patch file (external/verl-hf-rollout-registry.patch) as the checks above.
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vas
        vas_src = Path(inspect.getfile(vas)).read_text()
        assert "register_echoprime_vllm_model" in vas_src, (
            "stock verl (no vLLM registration call) -- "
            "apply external/verl-hf-rollout-registry.patch")

        import verl.experimental.agent_loop.agent_loop as al
        al_src = Path(inspect.getfile(al)).read_text()
        assert "clip_embeddings" in al_src and "detr_embeddings" in al_src, (
            "stock verl (_compute_multi_modal_inputs drops non-processor multimodal data) -- "
            "apply external/verl-hf-rollout-registry.patch")
        return "vLLM-serving patches present"

    @check("verl's generate() passes through clip/detr multi-modal data (Task 8)")
    def _():
        # EchoPrimeToolAgentLoop (packages/echoprime_track/echoprime_tool_agent_loop.py) needs to
        # send clip/detr embeddings into generate() every turn -- stock AsyncLLMServerManager.
        # generate / vLLMHttpServer.generate only accept image_data/video_data, which hardcode
        # the "image"/"video" multi_modal_data keys one layer down. Our custom "clip"/"detr"
        # modalities (vllm_model.py) are silently dropped or mis-tagged through those params
        # (see vllm_model.py's own docstring for the mis-tagging failure mode). See
        # external/verl-mm-generate-passthrough.patch, applied via `git apply` after a fresh
        # `git submodule update --init` -- it does not survive that on its own.
        import verl.experimental.agent_loop.agent_loop as al
        al_src = Path(inspect.getfile(al)).read_text()
        assert "clip_data" in al_src and "detr_data" in al_src, (
            "stock verl (AsyncLLMServerManager.generate has no clip_data/detr_data) -- "
            "apply external/verl-mm-generate-passthrough.patch")

        import verl.workers.rollout.vllm_rollout.vllm_async_server as vas
        vas_src = Path(inspect.getfile(vas)).read_text()
        assert 'multi_modal_data["clip"]' in vas_src and 'multi_modal_data["detr"]' in vas_src, (
            "stock verl (vLLMHttpServer.generate doesn't route clip_data/detr_data into "
            "multi_modal_data) -- apply external/verl-mm-generate-passthrough.patch")
        return "clip/detr generate() passthrough present"

    @check("Qwen3-VL processor + rope index available")
    def _():
        from transformers import AutoProcessor  # noqa: F401
        from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401
        from verl.models.transformers.qwen3_vl import get_rope_index  # noqa: F401
        return "qwen3_vl OK"

    @check("vLLM can actually RUN on this GPU (not just import)")
    def _():
        """The check that was missing on 2026-08-17 and cost a day.

        The old gate only asserted that vllm imported, which a CUDA-built wheel
        does perfectly well on an AMD box -- right up until EngineCore starts and
        dies on libcudart.so.12. Importability is not runnability. This asserts
        the compiled extension matches the platform, which is the actual thing
        that was wrong. It stops short of loading a model (minutes); the extension
        check is what distinguishes a CUDA wheel from a ROCm one.
        """
        import torch
        from vllm.platforms import current_platform
        plat = type(current_platform).__name__
        if plat == "UnspecifiedPlatform":
            raise AssertionError(
                "vLLM resolved UnspecifiedPlatform -- on ROCm this usually means "
                "amdsmi is missing or ABI-mismatched. Install from a WRITABLE copy "
                "of /opt/rocm/share/amd_smi, not from PyPI.")
        is_rocm = torch.version.hip is not None
        if is_rocm:
            try:
                import vllm._rocm_C  # noqa: F401
            except Exception as e:
                raise AssertionError(
                    f"torch is a ROCm build but vllm._rocm_C is unavailable ({e}). "
                    "The installed vLLM is almost certainly a CUDA wheel and CANNOT "
                    "serve on this GPU -- no agentic eval, no GRPO rollouts. "
                    "Use a ROCm vLLM (container or source build).") from None
        else:
            import vllm._C  # noqa: F401
        return f"{plat}, compiled extension present"

    @check("pyarrow available for parquet data-gen")
    def _():
        import pyarrow
        return pyarrow.__version__

    @check("h5py available for clip/detr cache reads (EchoPrimeToolAgentLoop)")
    def _():
        import h5py
        return h5py.__version__

    @check("verl_bridge imports (session/reward/generate_trainset/echo_tool)")
    def _():
        import verl_bridge.echo_tool  # noqa: F401
        import verl_bridge.generate_trainset  # noqa: F401
        import verl_bridge.reward  # noqa: F401
        import verl_bridge.session  # noqa: F401
        return "all four import"

    @check("custom_reward_function entrypoint signature")
    def _():
        from verl_bridge.reward import compute_score
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
