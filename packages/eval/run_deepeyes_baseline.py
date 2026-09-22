"""Small, reproducible DeepEyes QA baseline. --prepare-only needs no GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

SYSTEM_PROMPT = (
    'You are shown one preview image per available view of a cardiac ultrasound '
    'study. Answer the question based only on these images. '
    'Give your final answer inside <answer>...</answer> tags.'
)
SINGLE_FRAME_SYSTEM_PROMPT = (
    'You are shown a single frame from one view of a cardiac ultrasound study. '
    'Answer the question based only on this image. '
    'Give your final answer inside <answer>...</answer> tags.'
)
TYPES = ('abnormality_classification', 'abnormality_list', 'structure_description',
         'conclusion', 'full_report')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_records(records, per_type, seed):
    records = list(records)
    random.Random(seed).shuffle(records)
    counts, studies, selected = Counter(), set(), []
    for rec in records:
        kind = rec['question_type']
        if kind not in TYPES or counts[kind] >= per_type or rec['study_uuid'] in studies:
            continue
        selected.append(rec)
        counts[kind] += 1
        studies.add(rec['study_uuid'])
    if any(counts[k] != per_type for k in TYPES):
        raise ValueError(f'Not enough distinct studies per question type: {dict(counts)}')
    return selected


def sample_single_frame(records, seed, per_type=None):
    """Choose one frame per QA pair, optionally across a balanced batch."""
    rng = random.Random(seed)
    chosen = ([rng.choice(records)] if per_type is None
              else sample_records(records, per_type, seed))
    selected = []
    for rec in chosen:
        view = rng.choice(rec['overview']['views'])
        frames = sorted(Path(view['frame']).parent.glob('*.png'), key=lambda p: int(p.stem))
        if not frames:
            raise FileNotFoundError(f"No frames for selected view: {view['view']}")
        frame_index = rng.randrange(len(frames))
        selected_view = dict(view, frame=str(frames[frame_index]),
                             frame_index=frame_index, frame_count=len(frames))
        selected.append(dict(rec, overview=dict(rec['overview'], views=[selected_view])))
    return selected


def messages_for(rec, system_prompt=SYSTEM_PROMPT):
    content = []
    for view in rec['overview']['views']:
        content.extend([{'type': 'text', 'text': 'View: ' + view['view']},
                        {'type': 'image'}])
    content.append({'type': 'text', 'text': rec['question']})
    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content}]


def extract_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    if matches:
        return matches[-1].strip()
    # Never turn unfinished reasoning or a tool request into a clinical answer.
    if '<tool_call>' in text or text.count('<think>') != text.count('</think>'):
        return None
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip() or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval-jsonl', default='build/eval.jsonl')
    ap.add_argument('--train-jsonl', default='build/rl.jsonl')
    ap.add_argument('--model', default='checkpoints/deepeyes_7b')
    ap.add_argument('--revision-file', default='build/deepeyes_baseline_20260922/model_revision.txt')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-type', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    frame_mode = ap.add_mutually_exclusive_group()
    frame_mode.add_argument('--single-frame', action='store_true',
                            help='Use one random QA pair, one random view, and one random frame')
    frame_mode.add_argument('--single-frame-batch', action='store_true',
                            help='Use --per-type QA pairs per type, each with one random frame')
    ap.add_argument('--max-new-tokens', type=int, default=2048)
    ap.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if args.per_type < 1 or args.max_new_tokens < 1:
        ap.error('per-type and max-new-tokens must be positive')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.jsonl').exists():
        raise FileExistsError('Use a fresh out-dir to preserve previous predictions')
    with open(args.eval_jsonl) as f:
        pool = [json.loads(line) for line in f if line.strip()]
    with open(args.train_jsonl) as f:
        train_studies = {json.loads(line)['study_uuid'] for line in f if line.strip()}
    test_studies = {r['study_uuid'] for r in pool}
    if train_studies & test_studies:
        raise ValueError('Train/test study overlap')
    single_frame = args.single_frame or args.single_frame_batch
    records = (sample_single_frame(pool, args.seed,
                                  args.per_type if args.single_frame_batch else None) if single_frame
               else sample_records(pool, args.per_type, args.seed))
    system_prompt = SINGLE_FRAME_SYSTEM_PROMPT if single_frame else SYSTEM_PROMPT
    from PIL import Image
    image_metadata = []
    for rec in records:
        if not rec['overview']['views']:
            raise ValueError('Selected study has no images')
        for view in rec['overview']['views']:
            path = Path(view['frame'])
            with Image.open(path) as img:
                size = img.size
                img.verify()
            image_metadata.append({'path': str(path), 'sha256': sha256(path), 'size': size})
    metadata = dict(vars(args), model_id='ChenShawn/DeepEyes-7B',
                    mode=('single_random_frame_batch' if args.single_frame_batch else
                          'single_random_frame' if args.single_frame else 'direct_answers_preview_frames'),
                    system_prompt=system_prompt,
                    eval_rows=len(pool), eval_studies=len(test_studies),
                    train_studies=len(train_studies), train_test_overlap=0,
                    selected_rows=len(records), selected_studies=len(records),
                    question_types=dict(Counter(r['question_type'] for r in records)),
                    eval_sha256=sha256(args.eval_jsonl), train_sha256=sha256(args.train_jsonl),
                    images=image_metadata, temperature=0.0, processor_use_fast=True,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    runner_sha256=sha256(__file__), command=[sys.executable, *sys.argv], status='prepared')
    metadata['git_status'] = subprocess.check_output(['git', 'status', '--short'], text=True)
    (out / 'sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    def save_metadata():
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_metadata()
    print(f'PREFLIGHT_PASS: {len(records)} QA pairs, {len(image_metadata)} images, zero train/test study overlap', flush=True)
    if args.prepare_only:
        return
    metadata['model_revision'] = Path(args.revision_file).read_text().strip()
    config = json.loads((Path(args.model) / 'config.json').read_text())
    if config['model_type'] != 'qwen2_5_vl':
        raise ValueError('Expected the original Qwen2.5-VL DeepEyes model')
    import torch
    import transformers
    import wandb
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; run through Slurm')
    torch.manual_seed(args.seed)
    probe = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    if not torch.isfinite(probe @ probe.T).all().item():
        raise RuntimeError('GPU arithmetic preflight failed')
    del probe
    metadata.update(torch=torch.__version__, transformers=transformers.__version__,
                    device=torch.cuda.get_device_name(0), slurm_job_id=os.getenv('SLURM_JOB_ID'))
    run = wandb.init(project='echo-eval', entity=os.getenv('WANDB_ENTITY', 'anaatef9-mbzuai'),
                     name=f'deepeyes_basic_qa_{os.getenv("SLURM_JOB_ID", "local")}',
                     dir=str(out.resolve()), config={k: v for k, v in metadata.items() if k != 'images'})
    metadata['wandb_mode'] = run.settings.mode
    metadata['wandb_url'] = run.url if run.settings.mode == 'online' else None
    metadata['wandb_local_dir'] = run.dir
    print('WANDB_LOG=' + (metadata['wandb_url'] or run.dir), flush=True)
    save_metadata()
    succeeded = False
    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                                  max_pixels=args.max_pixels, use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='sdpa').eval()
        metadata['status'] = 'running'
        save_metadata()
        with (out / 'predictions.jsonl').open('x') as f:
            for index, rec in enumerate(records):
                images = []
                for view in rec['overview']['views']:
                    with Image.open(view['frame']) as img:
                        images.append(img.convert('RGB'))
                messages = messages_for(rec, system_prompt)
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=images, return_tensors='pt').to(model.device)
                input_tokens = inputs['input_ids'].shape[1]
                start = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                tokens = output[0, input_tokens:].tolist()
                raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                eos = model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                truncated = len(tokens) >= args.max_new_tokens and tokens[-1] not in eos_ids
                result = dict(study_uuid=rec['study_uuid'], question_type=rec['question_type'],
                              question=rec['question'], gold_answer=rec['answer'],
                              frames=rec['overview']['views'], messages=messages, prompt=prompt,
                              raw_output=raw, answer=extract_answer(raw),
                              input_tokens=input_tokens, output_tokens=len(tokens),
                              truncated=truncated, finish_reason='length' if truncated else 'eos',
                              tool_request_unexecuted='<tool_call>' in raw,
                              seconds=time.monotonic() - start)
                f.write(json.dumps(result) + '\n')
                f.flush()
                run.log({'examples_completed': index + 1, 'output_tokens': len(tokens),
                         'truncated': int(truncated), 'seconds': result['seconds']})
                print(f'EXAMPLE {index+1}/{len(records)} {rec["question_type"]}: {result["answer"]!r}', flush=True)
        metadata['status'] = 'completed'
        run.summary['examples_completed'] = len(records)
        succeeded = True
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_metadata()
        run.finish(exit_code=0 if succeeded else 1)


if __name__ == '__main__':
    main()
