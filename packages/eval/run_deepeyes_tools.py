"""Run echo questions through a pinned DeepEyes evaluation function."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from eval.run_deepeyes_baseline import sha256

UPSTREAM_COMMIT = '11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1'


def load_upstream(source, api_url, out_dir):
    """Import the original script without executing its benchmark driver."""
    argv = sys.argv
    sys.argv = [str(source), '--api_url', api_url, '--eval_model_name', 'deepeyes',
                '--save_path', str(out_dir), '--model_name', 'upstream']
    try:
        spec = importlib.util.spec_from_file_location('deepeyes_vstar_inference', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    # The original loop is unchanged. Never execute model-generated Python.
    module.eval = ast.literal_eval
    if source.name == 'eval_hrbench.py':
        # The pinned HRBench demo uses copy.deepcopy without importing copy.
        module.copy = copy
    # Echo QA is open-ended: remove the benchmark's multiple-choice options.
    module.instruction_prompt_before = 'Question: {question}\n' + module.USER_PROMPT_V2
    return module


class RecordingClient:
    """Record each actual model request/response and its exact image bytes."""

    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turns = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **params):
        saved = copy.deepcopy(params)
        for message in saved['messages']:
            if not isinstance(message['content'], list):
                continue
            for item in message['content']:
                if item['type'] == 'image_url':
                    raw = base64.b64decode(item['image_url']['url'].split(',', 1)[1])
                    image = Path('images') / (hashlib.sha256(raw).hexdigest() + '.png')
                    target = self.directory / image
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    item['image_url']['url'] = str(image)
        turn = {'request': saved}
        self.turns.append(turn)
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**params)
            turn.update(raw_output=response.choices[0].message.content,
                        finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else None)
            return response
        except Exception as exc:
            turn['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            turn['seconds'] = time.monotonic() - start
            (self.directory / 'turns.json').write_text(json.dumps(self.turns, indent=2) + '\n')


def run_record(module, rec, index, directory, client=None):
    directory = Path(directory)
    inputs = directory / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    name = f'{index:02d}.jpg'
    # V* expects a .jpg name; a symlink preserves our PNG's original pixels.
    frame = Path(rec['overview']['views'][0]['frame']).resolve()
    (inputs / name).symlink_to(frame)
    # This placeholder satisfies the benchmark output schema. It is never prompted.
    (inputs / f'{index:02d}.json').write_text(json.dumps(
        {'question': rec['question'], 'options': ['']}) + '\n')
    trace = RecordingClient(client or module.client, directory / 'trace')
    previous = module.client
    module.client = trace
    start = time.monotonic()
    try:
        if Path(module.__file__).name == 'eval_hrbench.py':
            # HRBench expects a dataframe row. No reference is prompted.
            annotation = dict(image=base64.b64encode(frame.read_bytes()).decode(),
                              question=rec['question'], answer='A', A='', B='', C='', D='',
                              category=rec['question_type'])
            upstream = module.process((0, SimpleNamespace(iloc=[annotation])))
        else:
            upstream = module.process((name, str(inputs)))
    finally:
        module.client = previous
    turns = trace.turns
    raw = turns[-1].get('raw_output', '') if turns else ''
    answer = (raw.split('<answer>', 1)[1].split('</answer>', 1)[0].strip()
              if '<answer>' in raw and '</answer>' in raw else None)
    observations = 0
    for turn in turns:
        delivered = sum(
            any(item.get('text') == '<tool_response>' for item in msg['content'])
            for msg in turn['request']['messages']
            if msg['role'] == 'user' and isinstance(msg['content'], list))
        observations = max(observations, delivered)
    return dict(index=index, question_type=rec['question_type'], question=rec['question'],
                gold_answer=rec['answer'], answer=answer, raw_output=raw,
                status=upstream['status'], upstream_result=upstream,
                tool_calls_requested=sum('<tool_call>' in t.get('raw_output', '') for t in turns),
                tool_observations_delivered=observations, turns=turns,
                truncated=any(t.get('finish_reason') == 'length' for t in turns),
                seconds=time.monotonic() - start)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-jsonl', required=True)
    ap.add_argument('--train-jsonl', default='build/rl.jsonl')
    ap.add_argument('--eval-jsonl', default='build/eval.jsonl')
    ap.add_argument('--upstream-root', default='external/DeepEyes')
    ap.add_argument('--protocol', choices=['vstar', 'hrbench'], default='vstar')
    ap.add_argument('--api-url', default='http://127.0.0.1:9/v1')
    ap.add_argument('--backend', choices=['vllm_api', 'transformers'], default='vllm_api')
    ap.add_argument('--model-path', default='checkpoints/deepeyes_7b')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--revision-file', default='build/deepeyes_baseline_20260922/model_revision.txt')
    args = ap.parse_args()
    from urllib.parse import urlsplit
    if urlsplit(args.api_url).hostname not in ('127.0.0.1', 'localhost'):
        ap.error('Inference server must be local to the inference machine')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.jsonl').exists():
        raise FileExistsError('Use a fresh output directory')
    records = [json.loads(s) for s in Path(args.sample_jsonl).read_text().splitlines() if s.strip()]
    if not records or len({r['study_uuid'] for r in records}) != len(records):
        raise ValueError('Sample must contain distinct studies')
    train = {json.loads(s)['study_uuid'] for s in Path(args.train_jsonl).open() if s.strip()}
    test = {json.loads(s)['study_uuid'] for s in Path(args.eval_jsonl).open() if s.strip()}
    from PIL import Image
    for rec in records:
        assert rec['study_uuid'] in test and rec['study_uuid'] not in train
        assert len(rec['overview']['views']) == 1
        with Image.open(rec['overview']['views'][0]['frame']) as image:
            image.verify()
    upstream_root = Path(args.upstream_root)
    commit = subprocess.check_output(['git', '-C', str(upstream_root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == UPSTREAM_COMMIT, 'Unexpected DeepEyes revision'
    source = upstream_root / f'eval/eval_{args.protocol}.py'
    pinned = subprocess.check_output(['git', '-C', str(upstream_root), 'show', f'HEAD:eval/eval_{args.protocol}.py'])
    assert source.read_bytes() == pinned, 'Upstream inference file has local edits'
    module = load_upstream(source, args.api_url, out)
    if args.backend == 'transformers':
        from eval.deepeyes_transformers_client import TransformersClient
        module.client = TransformersClient(args.model_path)
    metadata = dict(status='running', model_id='ChenShawn/DeepEyes-7B',
                    model_revision=Path(args.revision_file).read_text().strip(),
                    upstream_commit=commit, upstream_sha256=sha256(source),
                    runner_sha256=sha256(__file__), sample_sha256=sha256(args.sample_jsonl),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    command=[sys.executable, *sys.argv], selected_rows=len(records),
                    selected_studies=len(records), train_studies=len(train), test_studies=len(test),
                    train_test_overlap=0, system_prompt=module.instruction_prompt_system,
                    user_prompt=module.instruction_prompt_before, temperature=0.0,
                    protocol=args.protocol, backend=args.backend, model_path=args.model_path,
                    max_tokens_per_call=8192 if args.protocol == 'hrbench' else 10240,
                    max_calls=11, stop=['<|im_end|>', '</tool_call>'] if args.protocol == 'hrbench' else ['<|im_end|>'],
                    initial_image_resize=args.protocol == 'hrbench',
                    adaptations=['open-ended question without options', 'ast.literal_eval instead of eval'],
                    slurm_job_id=os.getenv('SLURM_JOB_ID'))
    if args.protocol == 'hrbench':
        metadata['adaptations'].append('supply missing upstream copy module import')
    if args.backend == 'transformers':
        metadata['adaptations'].append('local Transformers generation instead of vLLM API')
        metadata['backend_sha256'] = sha256(Path(__file__).with_name('deepeyes_transformers_client.py'))
    from importlib.metadata import version, PackageNotFoundError
    metadata['runtime_versions'] = {}
    for name in ('vllm', 'torch', 'transformers', 'openai', 'pillow'):
        try:
            metadata['runtime_versions'][name] = version(name)
        except PackageNotFoundError:
            metadata['runtime_versions'][name] = None
    shutil.copy2(args.sample_jsonl, out / 'sample.jsonl')
    (out / 'source').mkdir(exist_ok=True)
    shutil.copy2(source, out / 'source' / source.name)
    shutil.copy2(__file__, out / 'source/run_deepeyes_tools.py')
    import wandb
    run = wandb.init(project='echo-eval', entity='anaatef9-mbzuai',
                     name=f'deepeyes_original_tools_{os.getenv("SLURM_JOB_ID", "local")}',
                     dir=str(out.resolve()), mode='offline', config=metadata)
    metadata.update(wandb_mode='offline', wandb_local_dir=run.dir, wandb_url=None)
    results = []
    try:
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with (out / 'predictions.jsonl').open('x') as stream:
            for index, rec in enumerate(records):
                result = run_record(module, rec, index, out / f'example_{index:02d}')
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                results.append(result)
                run.log({k: result[k] for k in ('tool_calls_requested', 'tool_observations_delivered',
                                               'truncated', 'seconds')})
                print(f'EXAMPLE {index + 1}/{len(records)} status={result["status"]} '
                      f'tool_requests={result["tool_calls_requested"]} '
                      f'observations={result["tool_observations_delivered"]} '
                      f'answer={result["answer"] is not None}', flush=True)
        metadata['status'] = 'completed'
        metadata['summary'] = dict(examples=len(results),
            tool_episodes=sum(r['tool_calls_requested'] > 0 for r in results),
            tool_calls_requested=sum(r['tool_calls_requested'] for r in results),
            tool_observations_delivered=sum(r['tool_observations_delivered'] for r in results),
            answers=sum(r['answer'] is not None for r in results),
            errors=sum(r['status'] != 'success' for r in results),
            truncated=sum(r['truncated'] for r in results),
            examples_with_addcriterion=sum(any('addCriterion' in t.get('raw_output', '')
                                              for t in r['turns']) for r in results))
        run.summary.update(metadata['summary'])
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        run.finish(exit_code=0 if metadata['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
