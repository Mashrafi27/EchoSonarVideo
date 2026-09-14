"""Opt-in vLLM ROCm compatibility, without changes to upstream checkouts."""
import os


def register():
    if os.environ.get('ECHO_ROCM_SCRATCH_ROOT'):
        return 'echo_rocm_runtime.platform.EchoRocmPlatform'
    return None


def sync_visible_devices():
    """Honor VeRL's per-replica CUDA selection before a new HIP process starts."""
    hip = os.environ.get('HIP_VISIBLE_DEVICES')
    cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    if hip and cuda and hip != cuda:
        allowed = set(os.environ['ECHO_ROCM_ALLOWED_DEVICES'].split(','))
        if not set(hip.split(',')) <= allowed or not set(cuda.split(',')) <= allowed:
            raise RuntimeError('Visibility remapping would exceed the Slurm allocation')
        os.environ['HIP_VISIBLE_DEVICES'] = cuda
