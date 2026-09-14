"""Exercise UUID identity across visibility mappings and workspace IPC between processes."""
import json
import os
import subprocess
import sys
import torch
import zmq
from vllm.platforms import current_platform
from echo_rocm_runtime.sockets import install, workspace_endpoint

assert current_platform.is_rocm()
torch.cuda.init()
count = torch.cuda.device_count()
uuids = [current_platform.get_device_uuid(i) for i in range(count)]
assert len(set(uuids)) == count and count > 0, uuids
visible = os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
physical = visible.split(',') if visible else [str(i) for i in range(count)]
# Reproduce VeRL's server launch: it narrows CUDA visibility but leaves HIP unchanged.
child_env = dict(os.environ, HIP_VISIBLE_DEVICES=visible, CUDA_VISIBLE_DEVICES=physical[-1])
child = subprocess.run([sys.executable, '-c', 'import torch; torch.cuda.init(); from vllm.platforms import current_platform; print(current_platform.get_device_uuid(0))'], env=child_env, capture_output=True, text=True, timeout=60, check=True)
assert child.stdout.strip().splitlines()[-1] == uuids[-1], child.stdout
# Also reproduce the HTTP server's in-process change after platform import.
code = """import os, sys
from vllm.platforms import current_platform
from vllm.utils.system_utils import _sync_visible_devices_env_vars
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
_sync_visible_devices_env_vars()
assert os.environ['HIP_VISIBLE_DEVICES'] == sys.argv[1]
import torch
torch.cuda.init()
print(current_platform.get_device_uuid(0))
"""
child = subprocess.run([sys.executable, '-c', code, physical[-1]], capture_output=True, text=True, timeout=60, check=True)
assert child.stdout.strip().splitlines()[-1] == uuids[-1], child.stdout
install()
assert workspace_endpoint('tcp://127.0.0.1:1234') == 'tcp://127.0.0.1:1234'
endpoint = 'ipc:///tmp/rl-colocate-zmq-' + uuids[0] + '.sock'
with zmq.Context() as context:
    with context.socket(zmq.PAIR) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)
        bound = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        assert bound.startswith('ipc://' + os.environ['ROCM_SCRATCH_ALIAS'] + '/ipc/'), bound
        code = '''import sys, zmq
from echo_rocm_runtime.sockets import install
install()
with zmq.Context() as context:
    with context.socket(zmq.PAIR) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(sys.argv[1])
        socket.send(b'workspace-ipc')
        assert socket.poll(15000), 'reply timeout'
        assert socket.recv() == b'ok'
'''
        process = subprocess.Popen([sys.executable, '-c', code, endpoint])
        try:
            assert socket.poll(15000), 'child connection timeout'
            assert socket.recv() == b'workspace-ipc'
            socket.send(b'ok')
            assert process.wait(timeout=20) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        socket.unbind(endpoint)
print('ROCM_COMPAT_PREFLIGHT_PASS', json.dumps({'devices': count, 'uuids': uuids, 'visibility_remap': True, 'workspace_ipc': True}), flush=True)
