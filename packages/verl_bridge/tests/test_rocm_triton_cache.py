"""Verify that spawned workers cannot inherit another worker's compiler cache."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


STARTUP = Path(__file__).resolve().parents[3] / 'scripts/rocm_python_startup/sitecustomize.py'


class TritonCacheIsolationTest(unittest.TestCase):
    def test_spawned_processes_use_distinct_caches_under_job_scratch(self):
        code = '''
import json, os, runpy, sys
setup = runpy.run_path(sys.argv[1])
setup['_configure_triton_cache'](sys.argv[2])
print(json.dumps({'pid': os.getpid(), 'cache': os.environ['TRITON_CACHE_DIR']}))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            env = dict(os.environ)
            env.pop('ECHO_ROCM_SCRATCH_ROOT', None)
            env['TRITON_CACHE_DIR'] = str(root / 'triton' / 'parent')
            workers = [subprocess.Popen(
                [sys.executable, '-c', code, str(STARTUP), str(root)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) for _ in range(2)]
            results = []
            for worker in workers:
                stdout, stderr = worker.communicate(timeout=30)
                self.assertEqual(worker.returncode, 0, stderr)
                result = json.loads(stdout)
                cache = Path(result['cache'])
                self.assertEqual(cache, root / 'triton' / str(result['pid']))
                self.assertTrue(cache.is_dir())
                (cache / 'kernel.source').write_text(str(result['pid']))
                results.append(result)
            self.assertNotEqual(results[0]['cache'], results[1]['cache'])
            for result in results:
                self.assertEqual(
                    (Path(result['cache']) / 'kernel.source').read_text(), str(result['pid']))


if __name__ == '__main__':
    unittest.main()
