# Phase-3 Integration Contract — wiring `echo_env` into VeRL

`echo_env` is runtime-free. Nothing below runs offline — it needs the torch/vLLM runtime.

> **⚠️ SUPERSEDED APPROACH (2026-08-01, gate G1/G2 research).** Sections 1–2 below were
> written for the DeepEyes custom `ToolBase`/`parallel_env` agent layer on VeRL `0.2.0.dev`.
> That approach is **obsolete** — see **§0.1**. Upstream VeRL **v0.7.1** ships a NATIVE
> agent-loop + tool framework, so echo integrates as a net-new `BaseTool` with **no in-tree
> patch to the agent layer** and DeepEyes' custom `parallel_env` is not ported. Read §0.1
> as the governing contract; §1–§2 are kept only as a record of the deprecated path and the
> still-valid Qwen3-VL video-token facts.

## 0.1. NATIVE VeRL v0.7.1 integration (GOVERNING — 2026-08-01)

**Gate G1 = NATIVE, G2 = v0.7.1** (research, fetched trees at
`<scratchpad>/verl-060` and `/verl-071`). Upstream VeRL already provides the multi-turn,
multimodal, tool-env agent loop DeepEyes hand-rolled. Echo is a **net-new `BaseTool`**, not
a port.

- **Tool seam** — subclass `verl/tools/base_tool.py::BaseTool`: async `create(instance_id,
  **kwargs)` (per-trajectory init = old `reset`), `execute(instance_id, parameters) ->
  (ToolResponse, reward, metrics)` (= old `execute`), `calc_reward`, `release`. Byte-identical
  across v0.6.0/v0.7.1.
- **Multimodal obs return** — `verl/tools/schemas.py::ToolResponse(text=, image=[], video=[])`.
  Echo returns `ToolResponse(video=[...])`; the loop injects it. **Echo is the FIRST upstream
  tool to return `video`** (every shipped tool, incl. zoom, returns `image`) — schema-wired but
  unexercised → budget end-to-end debug time.
- **Agent loop** — `verl/experimental/agent_loop/tool_agent_loop.py::ToolAgentLoop`
  (`@register("tool_agent")`) drives generation, extracts tool calls, calls `tool.execute`, and
  merges `ToolResponse.image/.video` into `multi_modal_data` (v0.7.1 lines ~296-311). Runs on the
  **vLLM async server** (`vllm_async_server.py`) — no SGLang switch.
- **Registration = config, not import-metaclass** — tool: `multi_turn.tool_config_path` YAML lists
  each tool by `class_name` FQDN + `OpenAIFunctionToolSchema` (`verl/tools/utils/tool_registry.py`);
  agent: dataset row sets `agent_name="tool_agent"` in `non_tensor_batch`; per-row `extra_info.tools_kwargs`
  (read at `rl_dataset.py:361`) passes into `create`/`execute`. This is the native equivalent of
  DeepEyes' `env_name`→`ToolBase.create`.
- **Echo design decisions forced by the native seam:**
  1. **Single composite `EchoTool`** (not 3 separate tools) dispatching `parameters.op` ∈
     {select_view, select_frames, zoom}, keeping one per-trajectory state store keyed by
     `instance_id` — because `BaseTool` instances don't share state and echo's 3 tools mutate one
     shared (video + current view + frames) state. (`AgentData.extra_fields` is the alt shared-store.)
  2. **Termination:** native `execute` returns no `done` — episodes end via `max_assistant_turns`
     or the model emitting the final answer. Echo must NOT rely on env-signaled termination.
  3. Model the tools on `verl/tools/image_zoom_in_tool.py::ImageZoomInTool` (DeepEyes' zoom,
     upstreamed — a working `BaseTool` returning a cropped-image `ToolResponse` with Ray rate-limit).
- **Substrate (RESOLVED — user 2026-08-01): pivot to upstream verl v0.7.1.** The runtime base is
  **upstream `volcengine/verl`@v0.7.1 + net-new echo tool files** (BaseTool + tool-config YAML +
  configs), NOT DeepEyes' 0.2.0.dev + patches. DeepEyes is demoted to a **reference** for
  prompt/reward/data-gen patterns only. `transformers>=4.57`. Qwen3-VL model support is native in
  v0.7.1 (`qwen3_vl.py`), so the old §2 `parallel_env`/`rl_dataset` patches are unnecessary —
  v0.7.1 branches Qwen3-VL natively. This supersedes the earlier "DeepEyes = pinned submodule +
  patch set" decision now that the agent layer echo needed is upstream.

## 0. Base model = Qwen3-VL-8B-Instruct (and the #1 Phase-3 risk)

Base model is **Qwen3-VL-8B-Instruct** (user decision 2026-07-24, superseding the
Qwen2.5-VL-7B DeepEyes default). Two consequences for Phase-3 wiring:

- **⚠️ #1 blocker — the vendored VeRL lacks Qwen3-VL support.** VeRL in
  `external/DeepEyes` is `0.2.0.dev` with `transformers==4.51.3` and ZERO `qwen3`
  references; the mRoPE/processor/vision-token logic lives in ~7 Qwen2.5-VL-specific
  files. Enabling Qwen3-VL = a transformers bump (>=4.57) + a VeRL model-support
  port, NOT a config flag.
  - **RESOLVED (research 2026-08-01): (A) VeRL version-bump, project-owned rebase.**
    Overlay upstream VeRL **v0.6.0** (Qwen3-VL landed in `volcengine/verl` PR #3681,
    commit `42c55ac`, Oct 6 2025; v0.6.0 = earliest tag carrying it, Oct 15 2025) and
    re-apply DeepEyes' patch set on top. Pin **transformers>=4.57.0** (Qwen3-VL only
    exists from 4.57; v0.6.0 leaves transformers unpinned). Hand-backport rejected: it
    reimplements PR #3681 on a dead base + forces a tree-wide transformers jump inside
    0.2.0.dev. **Consider v0.7.x before pinning** (more qwen3vl fixes; v0.6.0 is earliest,
    not necessarily most stable).
  - **Execution caveat — the project owns the rebase, NOT a pin bump.** DeepEyes upstream
    never rebased: its latest `main` *is* our pin `11d20c6` (Nov 2025) and still vendors
    VeRL 0.2.0.dev with zero qwen3, despite post-dating PR #3681. So EchoSonarVideo overlays
    v0.6.0's `verl` tree + re-applies DeepEyes' ~7-file patch set + the added `workers/agent`
    layer itself. Keeps the pinned-submodule + versioned-patch model (not a fork) but the
    patch set is now LARGE.
  - **Rebase conflict map (from research):** model mRoPE seam **CLEAN** (`qwen2_vl.py`
    survives in v0.6.0 with the same `get_rope_index(image_grid_thw, video_grid_thw,
    second_per_grid_ts)` signature DeepEyes patches); `monkey_patch.py` + flash-attn forward
    **MODERATE** (renamed `_custom_flash_attention_forward`, removed `ulysses_flash_attn_forward`);
    the agentic layer `workers/agent/{parallel_env,tool_envs,envs}` + its reach into
    `rl_dataset/dp_actor/dp_critic/vllm_rollout_spmd` **HEAVY** — DeepEyes' own addition,
    re-based onto v0.6.0's newer async-rollout framework = where conflicts concentrate.
- **Geometry (already applied to `echo_env`).** Qwen3-VL merged visual patch =
  patch 16 x spatial_merge 2 = **32px** (vs Qwen2.5-VL's 28px). So `min_crop_side=32`,
  `highres_max_side=320`, `preview_max_side=160`, native target 320x320. Phase-1 PNGs
  stay 336x336; Qwen3-VL's processor smart-resizes to the nearest 32-multiple at load.

## 1. `ToolBase` adapter (net-new, registered by import)

`external/DeepEyes/verl/workers/agent/envs/echo/echo_env.py`:

    from verl.workers.agent.tool_envs import ToolBase
    from echo_env import EchoEnv, EnvConfig, to_deepeyes_obs

    class EchoToolEnv(ToolBase):
        name = "echo"
        user_prompt = "..."   # echo turn prompt (Phase-3 prompt design)

        def __init__(self, _n, _d, _p, **kw):
            super().__init__(name=self.name)
            self.env = EchoEnv(EnvConfig.from_env())

        def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kw):
            # study_uuid arrives via the dataset row (extra_info / a dedicated column);
            # Phase-3 data-gen must put it where reset() can read it.
            self.env.reset(study_uuid)

        def execute(self, action_string, **kw):
            obs, reward, done, info = self.env.step(action_string)
            if done or not hasattr(obs, "frames"):
                return "", reward, done, info
            return to_deepeyes_obs(obs, self.user_prompt), reward, done, info

Registration = one import line in the launch/entry module (preferred) so upstream
`verl/workers/agent/__init__.py` stays untouched; patch it only if import ordering forces it.
Dataset rows set `env_name="echo"` so `ToolBase.create("echo")` finds this class.

## 2. Video patch (the one irreducible in-tree edit — `[patch]` in spec §8)

`to_deepeyes_obs` currently emits **`<image>`** blocks (multi-image), which already
works through DeepEyes unchanged. TRUE video (cine motion + temporal mRoPE) is a
separate, better-but-harder path requiring:

- `verl/workers/agent/parallel_env.py`: add a `<video>` →
  `<|vision_start|><|video_pad|><|vision_end|>` branch in `_preprocess_multi_modal_inputs`
  (call `processor(videos=...)`) and extend the obs merge-back to append `mm_data['video']`
  + `video_grid_thw`/`second_per_grid_ts`.
- `verl/utils/dataset/rl_dataset.py`: also populate `origin_multi_modal_data["video"]`.

**Decision (2026-08-01): TRUE VIDEO** (user), superseding the deferred image-vs-video
choice. Phase 2 ships the image-packaging seam; Phase 3 adds the true-video path. The two
edits live as a versioned patch under `external/patches/echo-video-*.patch`, applied to the
(rebased-to-v0.6.0) submodule at setup — per the "submodule + patch set" decision. The
video-token names and mRoPE metadata MUST come from the real **Qwen3-VL** HF processor,
never hand-rolled — getting this wrong breaks temporal position ids silently.

**Qwen3-VL video surface (research-pinned 2026-08-01), the exact seam this patch handles:**
- Pad/vision tokens are **UNCHANGED** from Qwen2.5-VL: `<|video_pad|>`, `<|vision_start|>`,
  `<|vision_end|>`; grids `video_grid_thw` (T,H,W) persist. (transformers `processing_qwen3_vl.py`.)
- **`second_per_grid_ts` is GONE** — it's a Qwen2.5-VL-ism, absent from Qwen3-VL's processor
  output AND from v0.6.0's `qwen3_vl.get_rope_index`. DeepEyes currently pops/passes it and
  calls `qwen2_vl.get_rope_index`. **For Qwen3-VL the video patch must branch on
  `Qwen3VLImageProcessor` and route through `qwen3_vl.get_rope_index` (which takes NO
  `second_per_grid_ts`).** This is the money finding — the exact plumbing change.
- **Timestamp-aware / interleaved-MRoPE:** Qwen3-VL emits interleaved timestamp text per
  segment (e.g. `"<1.5 seconds><|vision_start|><|video_pad|>...<|vision_end|>"`) driven by a
  **`video_metadata`** field (fps + frame indices). Qwen2.5-VL had no such text. The env's
  video-dict must supply `video_metadata` so the processor builds timestamps correctly.
- `rl_dataset.py` currently branches on `Qwen2VLImageProcessor` and imports `get_rope_index`
  from `qwen2_vl`; `parallel_env.py` does the same + pops `second_per_grid_ts`. Both must be
  extended to also handle **`Qwen3VLImageProcessor` → `qwen3_vl.get_rope_index`**.

## 3. Reward scorer (net-new) — Phase 3

`verl/utils/reward_score/echo.py` + dispatch on `data_source="echo"`. See spec §5.2 / §8.
