"""Replay held-out echo frames through pinned visual-reasoning protocols."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from eval.run_deepeyes_tools import RecordingClient
from eval.visual_tool_baselines import image_protocols
from eval.visual_tool_baselines.upstream import verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['chain_of_focus', 'mini_o3', 'video_com'], required=True)
    ap.add_argument('--upstream-root', default='build/visual_baselines_20260922')
    ap.add_argument('--sample-jsonl', default='build/deepeyes_icardio_20260922/inputs/sample.jsonl')
    ap.add_argument('--train-jsonl', default='build/rl.jsonl')
    ap.add_argument('--eval-jsonl', default='build/eval.jsonl')
    ap.add_argument('--model-path')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--video-manifest')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metadata.json').exists():
        raise FileExistsError('Use a fresh run directory')
    root = Path(args.upstream_root)
    upstream = root / args.model
    pin = verify(upstream, args.model)
    identity = json.loads((root / (args.model + '_model.json')).read_text())
    records = [json.loads(s) for s in Path(args.sample_jsonl).read_text().splitlines() if s.strip()]
    train = {json.loads(s)['study_uuid'] for s in Path(args.train_jsonl).open() if s.strip()}
    test = {json.loads(s)['study_uuid'] for s in Path(args.eval_jsonl).open() if s.strip()}
    assert len({r['study_uuid'] for r in records}) == len(records)
    from PIL import Image
    for rec in records:
        assert rec['study_uuid'] in test and rec['study_uuid'] not in train
        assert len(rec['overview']['views']) == 1
        with Image.open(rec['overview']['views'][0]['frame']) as im:
            im.verify()
    model_path = args.model_path or str(Path('checkpoints') / args.model)
    metadata = dict(status='loading', model=args.model, **identity, model_path=model_path,
                    upstream_commit=pin, backend='transformers_bf16_sdpa', seed=1,
                    selected_studies=len(records), train_studies=len(train), test_studies=len(test),
                    train_overlap=0, include_view_context=True,
                    command=[sys.executable, *sys.argv],
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    runtime_versions={name: version(name) for name in ['torch', 'transformers', 'pillow']},
                    adaptations=['our open-ended questions with view/video-frame context',
                                 'Transformers BF16 SDPA backend; upstream serving environment differs',
                                 'offline inference only; no external judge or training'],
                    source_sha256={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *Path(__file__).with_name('visual_tool_baselines').glob('*.py')]})
    if args.model == 'mini_o3':
        metadata['adaptations'].append('standalone rollout adapter with original prompts/crop helpers and safe bbox literal parsing')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copy2(args.sample_jsonl, out / 'sample.jsonl')
    if args.model == 'video_com':
        from eval.visual_tool_baselines.video_com import VideoBackend, VideoRecordingClient
        if not args.video_manifest:
            ap.error('--video-manifest is required for Video-CoM')
        backend = VideoBackend(model_path, upstream)
        recorder_class = VideoRecordingClient
        metadata['adaptations'] += ['16 initial frames at stride 2 and seeded random start, as requested',
                                    'ordered PNG recordings with nominal 2 fps container; acquisition timing unavailable',
                                    'open-ended action-syntax prompt; original standalone evaluation prompt unavailable',
                                    'greedy decoding; stop after final kept answer instead of training-only dummy rounds']
    else:
        from eval.deepeyes_transformers_client import TransformersClient
        backend = TransformersClient(model_path)
        recorder_class = RecordingClient
    import wandb
    run = wandb.init(project='echo-eval', name=args.model + '_view_context', mode='offline',
                     dir=str(out.resolve()), config=metadata)
    metadata.update(status='running', wandb_mode='offline', wandb_local_dir=run.dir, wandb_url=None)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    results = []
    try:
        with (out / 'predictions.jsonl').open('x') as stream:
            for index, rec in enumerate(records):
                client = recorder_class(backend, out / f'example_{index:02d}' / 'trace')
                start = time.monotonic()
                result = dict(index=index, study_uuid=rec['study_uuid'], question=rec['question'],
                              question_type=rec['question_type'], gold_answer=rec['answer'],
                              view=rec['overview']['views'][0]['view'])
                try:
                    if args.model == 'video_com':
                        from eval.visual_tool_baselines.video_com import run_video
                        result.update(run_video(upstream, rec, index, client, args.video_manifest, out))
                    else:
                        result.update(getattr(image_protocols, args.model)(upstream, rec, client))
                except Exception as exc:
                    result.update(protocol_status='error', error=f'{type(exc).__name__}: {exc}')
                turns = client.turns
                raw = turns[-1].get('raw_output', '') if turns else ''
                answer = re.findall(r'<answer>(.*?)</answer>', raw, re.DOTALL)
                if args.model == 'video_com':
                    result['answer'] = raw.split('FINAL_ANSWER:', 1)[-1].strip() if 'FINAL_ANSWER:' in raw else None
                else:
                    result['answer'] = answer[-1].strip() if answer else None
                result.update(turns=turns, seconds=time.monotonic() - start,
                              truncated=any(t.get('finish_reason') == 'length' for t in turns),
                              addcriterion_count=sum(t.get('raw_output', '').count('addCriterion') for t in turns))
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                run.log(dict(example=index + 1, calls=len(turns), seconds=result['seconds'],
                             answer_present=result['answer'] is not None, truncated=result['truncated']))
                print(f"EXAMPLE {index+1}/{len(records)} status={result['protocol_status']} calls={len(turns)} answer={result['answer'] is not None}", flush=True)
        metadata['status'] = 'completed'
        metadata['summary'] = dict(examples=len(results), answers=sum(r['answer'] is not None for r in results),
                                   model_calls=sum(len(r['turns']) for r in results),
                                   errors=sum(r['protocol_status'] != 'success' for r in results),
                                   truncated=sum(r['truncated'] for r in results),
                                   examples_with_addcriterion=sum(r['addcriterion_count'] > 0 for r in results),
                                   total_seconds=sum(r['seconds'] for r in results))
        run.summary.update(metadata['summary'])
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        run.finish(exit_code=0 if metadata['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
