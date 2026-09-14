"""Supply the GPU identity method missing from vLLM 0.17's ROCm platform."""
import ctypes
import os
from pathlib import Path
from vllm.platforms import rocm as _rocm
from vllm.platforms.rocm import RocmPlatform
from . import sync_visible_devices


def _wrap_visibility_sync(original):
    def sync():
        # VeRL also changes CUDA visibility after importing the platform, in
        # the HTTP server itself. Normalize immediately before vLLM spawns.
        sync_visible_devices()
        original()
    sync._echo_rocm_sync = True
    return sync


if not getattr(_rocm._sync_hip_cuda_env_vars, '_echo_rocm_sync', False):
    _rocm._sync_hip_cuda_env_vars = _wrap_visibility_sync(_rocm._sync_hip_cuda_env_vars)


class EchoRocmPlatform(RocmPlatform):
    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        # HIP accepts logical device indices after visibility filtering. This
        # avoids treating an AMD SMI physical index as a per-process device index.
        lib = ctypes.CDLL(str(Path(os.environ['ROCM_PATH']) / 'lib/libamdhip64.so'))
        uuid = (ctypes.c_ubyte * 16)()
        fn = lib.hipDeviceGetUuid
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = fn(ctypes.byref(uuid), int(device_id))
        if rc or not any(uuid):
            raise RuntimeError(f'hipDeviceGetUuid({device_id}) failed: rc={rc}, uuid={bytes(uuid).hex()}')
        return 'GPU-' + bytes(uuid).hex()
