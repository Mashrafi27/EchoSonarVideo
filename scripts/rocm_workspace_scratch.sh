#!/usr/bin/env bash
# Source inside a Slurm job. VAST rejects ':' in ROCm compiler filenames.
# A private ext4 image keeps every temporary file inside the project workspace.
# Call rocm_scratch_cleanup from the caller's EXIT trap after workers terminate.
rocm_scratch_setup() {
    local scratch_root="${1:?scratch root required}"
    local scratch_size="${2:-8G}"
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ROCm scratch requires a Slurm allocation' >&2; return 1; }
    mkdir -p "$scratch_root"
    scratch_root=$(realpath "$scratch_root")
    ROCM_SCRATCH_MOUNT="$scratch_root/scratch_${SLURM_JOB_ID}_${HOSTNAME}"
    ROCM_SCRATCH_IMAGE="${ROCM_SCRATCH_MOUNT}.img"
    [[ ! -e "$ROCM_SCRATCH_IMAGE" ]] || { echo "Scratch already exists: $ROCM_SCRATCH_IMAGE" >&2; return 1; }
    mkdir -p "$ROCM_SCRATCH_MOUNT"
    truncate -s "$scratch_size" "$ROCM_SCRATCH_IMAGE"
    mkfs.ext4 -q -F -m 0 -E "root_owner=$(id -u):$(id -g)" "$ROCM_SCRATCH_IMAGE"
    fuse2fs "$ROCM_SCRATCH_IMAGE" "$ROCM_SCRATCH_MOUNT"
    mountpoint -q "$ROCM_SCRATCH_MOUNT" || { echo 'Private scratch mount failed' >&2; return 1; }
    # Unix sockets have a 108-byte path limit. This short alias still points to
    # the workspace image; the owning Slurm shell keeps the descriptor open.
    exec {ROCM_SCRATCH_FD}<"$ROCM_SCRATCH_MOUNT"
    export ROCM_SCRATCH_ALIAS="/proc/$$/fd/$ROCM_SCRATCH_FD"
    export TMPDIR="$ROCM_SCRATCH_ALIAS/tmp" TMP="$ROCM_SCRATCH_ALIAS/tmp" TEMP="$ROCM_SCRATCH_ALIAS/tmp"
    export AMD_COMGR_CACHE_DIR="$ROCM_SCRATCH_MOUNT/comgr"
    export MIOPEN_USER_DB_PATH="$ROCM_SCRATCH_MOUNT/miopen"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
    export PYTORCH_KERNEL_CACHE_PATH="$ROCM_SCRATCH_MOUNT/torch_kernels"
    export TRITON_CACHE_DIR="$ROCM_SCRATCH_MOUNT/triton"
    export TORCHINDUCTOR_CACHE_DIR="$ROCM_SCRATCH_MOUNT/inductor"
    export VLLM_CACHE_ROOT="$ROCM_SCRATCH_MOUNT/vllm"
    mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" "$MIOPEN_USER_DB_PATH" \
      "$PYTORCH_KERNEL_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
    touch "$TMPDIR/gfx90a:sramecc+:xnack-.o"
    rm "$TMPDIR/gfx90a:sramecc+:xnack-.o"
    python3 - <<'SOCKET_CHECK'
import os, socket
p=os.path.join(os.environ['TMPDIR'],'socket_check')
with socket.socket(socket.AF_UNIX) as s:
    s.bind(p)
os.unlink(p)
SOCKET_CHECK
    echo "ROCM_WORKSPACE_SCRATCH=$ROCM_SCRATCH_MOUNT"
}
rocm_scratch_cleanup() {
    if [[ -n "${ROCM_SCRATCH_FD:-}" ]]; then exec {ROCM_SCRATCH_FD}<&-; fi
    if [[ -n "${ROCM_SCRATCH_MOUNT:-}" ]] && mountpoint -q "$ROCM_SCRATCH_MOUNT"; then
        fusermount -u "$ROCM_SCRATCH_MOUNT" || fusermount -uz "$ROCM_SCRATCH_MOUNT"
    fi
}
