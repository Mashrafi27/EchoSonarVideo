#!/usr/bin/env bash
#SBATCH --job-name=vllm_build
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Build a ROCm vLLM from source, so GRPO stops waiting on someone else's .sif.
#
#   STAGE=gate1 sbatch scripts/build_vllm_rocm.sbatch     # torch 2.9 rocm6.4 runs here?
#   STAGE=build sbatch scripts/build_vllm_rocm.sbatch     # compile vLLM for gfx90a
#
# This does NOT contradict the container verdict in CLAUDE.md. That verdict covers
# routes to a prebuilt IMAGE (docker:// unpack, OCI-SIF, docker group) and still
# holds. Building from source was written off without being checked, and the checks
# now say otherwise: pypi / repo.radeon.com / github are all reachable through the
# site proxy, hipcc and amdclang++ are present, and verl 0.7.1 accepts vllm <= 0.12.0.
#
# WHY A NEW VENV, standalone, not another overlay:
#   vLLM 0.11.2 is the first release with Qwen3-VL, and its ROCm build wants torch
#   2.9.0 + rocm6.4. The existing stack is torch 2.7.1+rocm6.2.4 living inside the
#   SHARED qwen_backup conda env. Upgrading torch there would be both forbidden and
#   reckless, and shadowing it from .venv-train would silently change the SFT path
#   that currently works. So this is additive: a separate env nothing else imports.
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"
STAGE=${STAGE:-gate1}
VENV=${VENV:-$REPO/.venv-vllm}
VLLM_TAG=${VLLM_TAG:-v0.11.2}
SRC=${SRC:-$REPO/.tmp_work/vllm-src}
BASE_PY=${BASE_PY:-/vast/users/mohammad.yaqub/miniconda3/envs/qwen_backup/bin/python}

# ROCm trap 1: SLURM exports ROCR_VISIBLE_DEVICES and torch ROCm hard-errors on it.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
# ROCm trap 3: the MIOpen DB under ~/.config is read-only on VAST.
export MIOPEN_USER_DB_PATH="$REPO/.tmp_work/miopen/vllmbuild_${SLURM_JOB_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH" logs "$REPO/.tmp_work"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PIP_CACHE_DIR="$REPO/.tmp_work/pip-cache"      # never /tmp
export TMPDIR="$REPO/.tmp_work/tmp"                   # pip/cmake scratch, never /tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -x "$VENV/bin/python" ]]; then
    # NO --system-site-packages: this env must not see qwen_backup's torch 2.7.
    "$BASE_PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -U pip setuptools wheel
fi
PY="$VENV/bin/python"

case "$STAGE" in
gate1)
    # The make-or-break fact: the torch 2.9 rocm6.4 wheel bundles ROCm USERSPACE,
    # but the kernel amdgpu/KFD driver here is 6.3.3. If that pairing does not run,
    # the whole source-build route dies -- there is no torch 2.9 rocm6.3 wheel, and
    # no vLLM before 0.11 knows Qwen3-VL. Do not debug a driver mismatch; fall back.
    "$PY" -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/rocm6.4
    "$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("device", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
c = (a @ b).float()
torch.cuda.synchronize()
print("matmul ok, mean", c.mean().item())
print("GATE1 PASS")
PYEOF
    ;;
build)
    [[ -d "$SRC" ]] || git clone --depth 1 --branch "$VLLM_TAG" \
        https://github.com/vllm-project/vllm.git "$SRC"
    # System cmake is 3.22; vLLM needs >= 3.26.1.
    "$PY" -m pip install ninja "cmake>=3.26.1,<4"
    "$PY" -m pip install -r "$SRC/requirements/rocm-build.txt"
    # gfx90a only. Building every arch is the difference between one hour and six,
    # and MI210 is the only GPU this cluster has.
    export PYTORCH_ROCM_ARCH=gfx90a
    export VLLM_TARGET_DEVICE=rocm
    export MAX_JOBS=${MAX_JOBS:-32}
    # --no-build-isolation so the build sees the torch we just installed instead of
    # pulling a fresh CUDA one into an isolated build env.
    ( cd "$SRC" && "$PY" -m pip install --no-build-isolation -v . )
    "$PY" -m pip install -r "$SRC/requirements/rocm.txt"
    "$PY" - <<'PYEOF'
import vllm, torch
print("vllm", vllm.__version__, "torch", torch.__version__)
import vllm._rocm_C          # the import that the CUDA wheel cannot satisfy
from vllm.platforms import current_platform
print("platform", current_platform.device_name)
print("BUILD PASS")
PYEOF
    ;;
*)
    echo "unknown STAGE=$STAGE (gate1|build)"; exit 2 ;;
esac
