"""Opt-in ROCm worker setup; fail startup if required workspace safeguards fail."""
import os
from pathlib import Path


def _configure(root):
    from echo_rocm_runtime import sync_visible_devices
    from echo_rocm_runtime.sockets import install

    sync_visible_devices()
    cache = Path(root) / 'miopen' / str(os.getpid())
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MIOPEN_USER_DB_PATH'] = str(cache)
    os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = str(cache)
    install()


root = os.environ.get('ECHO_ROCM_SCRATCH_ROOT')
if root:
    try:
        _configure(root)
    except Exception as exc:
        # site.py normally prints and ignores Exception from sitecustomize.
        # SystemExit prevents continuing with unredirected hardcoded IPC paths.
        raise SystemExit(f'Echo ROCm runtime setup failed: {exc}') from exc
