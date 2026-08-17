# Training environment (P3b)

**STATUS: UNRUN.** Nothing in this document has been installed or executed. It is
derived entirely from the pinned submodule `external/verl` @ **v0.7.1**
(`bec9ef74768dd201881cd4e54cd0385e87caae27`) — its `setup.py`,
`requirements.txt`, and `docker/Dockerfile.stable.vllm`. The verification gate is
`scripts/check_train_env.py` running green on a real GPU node.

Scope of P3b: pin the runtime, record the dependency set, and provide a single
smoke command that tells you whether every 🟡 assumption in `echo_verl` holds.
The original P3b scope (VeRL Qwen3-VL *enablement* — tree overlay, re-applied
mRoPE patches, `Qwen3VLImageProcessor` branching) **no longer exists**: v0.7.1
ships Qwen3-VL natively (`verl/models/transformers/qwen3_vl.py::get_rope_index`,
monkey-patch entries for `qwen3_vl`/`qwen3_vl_moe`), and the native
`BaseTool`/`ToolAgentLoop` framework removed the agent-layer port. See
`echo_env/INTEGRATION.md` §0.1–§0.2 for the governing contract.

## 1. What is pinned

| Thing | Pin | Where it comes from |
|---|---|---|
| verl | `external/verl` submodule @ tag `v0.7.1` (`bec9ef7`) | this repo |
| DeepEyes | `external/DeepEyes` @ `11d20c6` | **reference only** — demoted at the native pivot, not deleted |
| base model | `Qwen3-VL-8B-Instruct` (32px merged patch, 320 native) | project decision |
| transformers | `>=4.57.0` | project override; verl leaves it unpinned |
| python | `>=3.10` (stable image uses 3.12) | verl `pyproject.toml` / Dockerfile |

The submodule is the *reference and reproducibility* pin. verl itself is
installed into the training env (editable from that checkout, or from the
prebuilt image) — the repo does not vendor a modified verl. **There are no
patches to verl.** `echo_verl` is net-new code that plugs into upstream seams.

## 2. Recommended install: verl's stable image

`docker/Dockerfile.stable.vllm` at this tag builds on
`nvidia/cuda:12.9.1-devel-ubuntu22.04` with python 3.12, `torch==2.10.0`
(cu129), `vllm==0.17.0`, `flash_attn==2.8.3`, apex, TransformerEngine.
Published as `verlai/verl:vllm017.latest`.

```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size=10g \
  --cap-add=SYS_ADMIN -v .:/workspace/echo --name echo verlai/verl:vllm017.latest sleep infinity
docker start echo && docker exec -it echo bash

# inside: install the PINNED verl without touching the image's torch/vllm
pip3 install --no-deps -e /workspace/echo/external/verl
pip3 install -U "transformers>=4.57.0"        # Qwen3-VL
pip3 install -r /workspace/echo/requirements-train.txt   # see the caveat in §3
python /workspace/echo/scripts/check_train_env.py
```

On a bare cluster env (no docker), install in this order: torch from the CUDA
wheel index → vLLM → `flash-attn --no-build-isolation` → `pip install -e
external/verl` → `requirements-train.txt`.

## 3. Known dependency conflicts (resolve empirically on first install)

1. **vLLM ceiling is stale.** `external/verl/setup.py` declares
   `vllm>=0.8.5,<=0.12.0` in the `[vllm]` extra, but the same tag's stable
   Dockerfile installs `vllm==0.17.0` and then installs verl with `--no-deps`,
   bypassing the ceiling. `requirements-train.txt` follows the Dockerfile. If you
   install verl *with* extras you will hit the conflict — use `--no-deps`.
2. **`numpy<2.0.0`** (verl) vs. whatever torch/vLLM pull in — verl's constraint
   wins; a newer numpy in the image is the thing to check first if imports fail.
3. **transformers is unpinned upstream**, so `>=4.57` is a clean addition rather
   than an override — but it is newer than what verl's CI tests, so treat
   Qwen3-VL processor behaviour as verified only once the smoke check passes.
4. `tensordict>=0.8.0,<=0.10.0,!=0.9.0` is asserted at `import verl` time
   (`verl/__init__.py`) — a mismatch fails loudly and early.

## 4. Smoke check

```bash
python scripts/check_train_env.py     # -v for tracebacks
```

Asserts, in order: transformers ≥4.57; `verl` imports; `BaseTool.__init__/create/
execute/release` signatures match what `echo_verl/echo_tool.py` was coded
against; `ToolResponse(image=[...])` constructs and rejects non-lists;
`echo_tool_config.yaml` round-trips through `OpenAIFunctionToolSchema` with the
`op` enum intact; `initialize_tools_from_config` actually instantiates `EchoTool`
under the name `echo`; `tool_agent` is in `_agent_loop_registry`; Qwen3-VL
processor + `verl.models.transformers.qwen3_vl.get_rope_index` import; pyarrow
present; all four `echo_verl` modules import; `compute_score` has verl's
`custom_reward_function` signature.

It also re-checks that `ToolAgentLoop` **still refuses tool-returned video**
(`tool_agent_loop.py`: `"Multimedia type 'video' is not currently supported"`).
That refusal is the reason for the hybrid frame path — initial observation is
true video via the dataset `videos` column, tool observations come back as
images. If a future verl version lifts it, the check prints a WARNING and
`INTEGRATION.md` §0.2 should be revisited.

Two of these assertions already run offline against the pinned submodule, with
no GPU, in `echo_verl/tests/test_tool_config.py` (schema round-trip + the
`ToolResponse` list contract) — that test loads verl's self-contained
`verl/tools/schemas.py` by path, so the `op` enum surviving pydantic's
`extra="ignore"` is **verified**, not assumed.

## 5. What is still GPU-gated after this

- `echo_verl/echo_tool.py` — imports verl; only py_compile + contract review so far.
- `echo_verl/generate_trainset.py::write_parquet` — needs pyarrow.
- Everything in P3d (SFT) and P3e (GRPO + eval).
