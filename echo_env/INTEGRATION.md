# Phase-3 Integration Contract — wiring `echo_env` into DeepEyes

`echo_env` is DeepEyes-runtime-free. Phase 3 wires it in via **net-new files** +
a **two-file video patch** to the pinned submodule `external/DeepEyes`
(commit 11d20c6). Nothing below runs offline — it needs the torch/vLLM runtime.

## 0. Base model = Qwen3-VL-8B-Instruct (and the #1 Phase-3 risk)

Base model is **Qwen3-VL-8B-Instruct** (user decision 2026-07-24, superseding the
Qwen2.5-VL-7B DeepEyes default). Two consequences for Phase-3 wiring:

- **⚠️ #1 blocker — the vendored VeRL lacks Qwen3-VL support.** VeRL in
  `external/DeepEyes` is `0.2.0.dev` with `transformers==4.51.3` and ZERO `qwen3`
  references; the mRoPE/processor/vision-token logic lives in ~7 Qwen2.5-VL-specific
  files. Enabling Qwen3-VL = a transformers bump (>=~4.57) + a VeRL model-support
  port/backport, NOT a config flag. **Resolve this BEFORE the video patch in section 2:**
  is it a VeRL version-bump (does a newer VeRL rev carrying `qwen3_vl` still accept the
  DeepEyes patches?) or a hand backport?
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

**Decision (2026-07-24):** Phase 2 ships the image-packaging seam; Phase 3 chooses
image-multi-frame vs. true-video per empirical token/quality tradeoff. If video is chosen,
the two edits live as a versioned patch under `external/patches/echo-video-*.patch`,
applied to the pinned submodule at setup (per the "submodule + patch set" decision).
mRoPE metadata (`video_grid_thw`, `second_per_grid_ts`) and the video-token names
(`<|video_pad|>` etc.) MUST come from the real **Qwen3-VL** HF processor — never
hand-rolled — since the Qwen2.5-VL illustration above may differ under Qwen3-VL's
interleaved-MRoPE / timestamp-aware video, and getting this wrong breaks temporal
position ids silently.

## 3. Reward scorer (net-new) — Phase 3

`verl/utils/reward_score/echo.py` + dispatch on `data_source="echo"`. See spec §5.2 / §8.
