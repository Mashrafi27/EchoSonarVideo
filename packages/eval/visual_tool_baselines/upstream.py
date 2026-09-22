"""Load pinned upstream protocol code without importing training frameworks."""
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

PINS = {
    'chain_of_focus': '431ff0dd703f83d8a624ec1ba1d2ef6a834b78bc',
    'mini_o3': '2c5a0dedb5279eff2c0e6049aac05de97bf7a2b3',
    'video_com': 'aa07959e6c1be94caca9ca0dd4e65a2051077114',
}


def verify(root, name):
    root = Path(root)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINS[name]:
        raise ValueError(f'Unexpected {name} upstream revision: {commit}')
    if subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only']).strip():
        raise ValueError('Upstream source must remain unmodified')
    return commit


def load_cof(root):
    path = Path(root) / 'eval/vstar/vllm_inference.py'
    tree = ast.parse(path.read_text())
    # Only substitute the serving import. All original functions and constants remain intact.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'vllm')]
    namespace = dict(__name__='pinned_cof', SamplingParams=lambda **kw: SimpleNamespace(**kw))
    exec(compile(tree, str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def load_definitions(path, names, namespace=None):
    """Compile selected original functions, keeping their source bytes in the pinned clone."""
    path = Path(path)
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in tree.body} != set(names):
        raise ValueError(f'Missing upstream definitions in {path}')
    namespace = dict(namespace or {})
    exec(compile(tree, str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)
