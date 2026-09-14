"""Redirect VeRL's two hardcoded IPC endpoints before any socket file is made."""
from functools import wraps
import hashlib
import os
from pathlib import Path

PREFIX = 'ipc:///tmp/rl-colocate-zmq-'


def workspace_endpoint(endpoint):
    binary = isinstance(endpoint, bytes)
    value = endpoint.decode() if binary else endpoint
    if not isinstance(value, str) or not value.startswith(PREFIX):
        return endpoint
    root = Path(os.environ['ROCM_SCRATCH_ALIAS']) / 'ipc'
    root.mkdir(exist_ok=True)
    name = hashlib.sha256(value.encode()).hexdigest()[:32] + '.sock'
    path = str(root / name)
    if len(path.encode()) >= 108:
        raise RuntimeError(f'IPC socket path exceeds Unix limit: {path}')
    redirected = 'ipc://' + path
    return redirected.encode() if binary else redirected


def install():
    import zmq
    if getattr(zmq.Socket, '_echo_workspace_ipc', False):
        return
    for name in ('bind', 'connect', 'unbind', 'disconnect'):
        original = getattr(zmq.Socket, name)

        def wrap(fn):
            @wraps(fn)
            def method(self, addr, *args, **kwargs):
                return fn(self, workspace_endpoint(addr), *args, **kwargs)
            return method

        setattr(zmq.Socket, name, wrap(original))
    zmq.Socket._echo_workspace_ipc = True
