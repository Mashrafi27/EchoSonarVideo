"""Render completed visual-tool runs with exact prompts, images and raw turns."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


def media_items(message):
    content = message['content']
    if not isinstance(content, list):
        return []
    result = []
    for item in content:
        if item['type'] == 'image_url':
            result.append(('image', [item['image_url']['url']]))
        elif item['type'] == 'image':
            result.append(('image', [item['image']]))
        elif item['type'] == 'video':
            result.append(('video', item['video']))
    return result


def audit_row(row, root, model):
    turns = row['turns']
    observations = sum(len(media_items(t['request']['messages'][-1])) for t in turns[1:])
    raw = [t.get('raw_output', '') for t in turns]
    result = dict(delivered_media_observations=observations,
                  repeated_identical_responses=len(raw) - len(set(raw)),
                  completed_answer=bool(row['answer']) and row['protocol_status'] == 'success',
                  invalid_bbox_observations=0)
    if model == 'deepeyes':
        final = raw[-1] if raw else ''
        result['trailing_tool_request_after_answer'] = '</answer>' in final and '<tool_call>' in final.split('</answer>', 1)[-1]
    if model == 'chain_of_focus':
        result['completed_answer'] = bool(row.get('upstream_result', {}).get('pred_ans')) and bool(row['answer'])
        if not result['completed_answer'] and len(turns) == 6:
            result['termination'] = 'turn_limit'
        if turns:
            name = media_items(turns[0]['request']['messages'][1])[0][1][0]
            with Image.open(root / f"example_{row['index']:02d}/trace" / name) as im:
                width, height = im.size
            for turn in turns[:-1]:
                try:
                    match = re.search(r'<tool_call>(.*?)</tool_call>', turn['raw_output'], re.DOTALL)
                    bbox = json.loads(match.group(1))['arguments']['bbox_2d']
                    x0, y0, x1, y1 = bbox
                    valid = (isinstance(bbox, list) and all(isinstance(x, (int, float)) for x in bbox)
                             and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
                except (ValueError, TypeError, KeyError, AttributeError):
                    valid = False
                result['invalid_bbox_observations'] += not valid
    return result


TITLES = dict(deepeyes='DeepEyes', chain_of_focus='Chain-of-Focus', mini_o3='Mini-o3', video_com='Video-CoM')


def run_model(metadata):
    if metadata.get('model_id') == 'ChenShawn/DeepEyes-7B':
        return 'deepeyes'
    return metadata.get('model')


def load_rows(root, metadata):
    """Predictions with one field vocabulary; DeepEyes rows carry upstream names and no sample fields."""
    rows = [json.loads(s) for s in (root / 'predictions.jsonl').read_text().splitlines() if s.strip()]
    if run_model(metadata) == 'deepeyes':
        sample = [json.loads(s) for s in (root / 'sample.jsonl').read_text().splitlines() if s.strip()]
        assert len(sample) == len(rows)
        for row, rec in zip(rows, sample):
            row.setdefault('protocol_status', row['status'])
            row.setdefault('study_uuid', rec['study_uuid'])
            row.setdefault('view', rec['overview']['views'][0]['view'])
            row.setdefault('addcriterion_count', sum(t.get('raw_output', '').count('addCriterion') for t in row['turns']))
    return rows


def render_run(root):
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    assert metadata['status'] == 'completed', metadata['status']
    rows = load_rows(root, metadata)
    assert len(rows) == metadata['selected_studies']
    model = run_model(metadata)
    domain = metadata.get('domain', 'echo')
    title = TITLES[model]
    reference_label = 'Expected answer (hand-written, unverified)' if domain == 'natural' else 'Dataset reference'
    audits = [audit_row(r, root, model) for r in rows]
    summary = dict(selected=len(rows), attempted=sum(bool(r['turns']) for r in rows),
                   completed_answers=sum(a['completed_answer'] for a in audits),
                   runtime_errors=sum(r['protocol_status'] == 'error' for r in rows),
                   incomplete_attempts=sum(bool(r['turns']) and not a['completed_answer'] for r, a in zip(rows, audits)),
                   skipped=sum(not r['turns'] for r in rows),
                   delivered_media_observations=sum(a['delivered_media_observations'] for a in audits),
                   invalid_bbox_observations=sum(a['invalid_bbox_observations'] for a in audits),
                   repeated_identical_responses=sum(a['repeated_identical_responses'] for a in audits),
                   truncated_examples=sum(r['truncated'] for r in rows),
                   examples_with_addcriterion=sum(r['addcriterion_count'] > 0 for r in rows),
                   seconds=sum(r['seconds'] for r in rows))
    (root / 'audit.json').write_text(json.dumps(dict(summary=summary, examples=audits), indent=2) + '\n')
    if domain == 'natural':
        heading = f'# {title}: natural sample inference'
        caveat = ('These counts describe inference and tool execution. Expected answers were written by hand after viewing '
                  'each file and are shown only for inspection; they are never prompted and were not independently verified.')
    else:
        heading = f'# {title}: held-out echo inference with view context'
        caveat = ('These counts describe inference and tool execution, not clinical accuracy. Dataset answers concern the study; '
                  'the model receives evidence from only one randomly selected view. Reference answers are never prompted.')
    parts = [heading, '',
             f"**{summary['completed_answers']}/{summary['attempted']} attempted examples produced completed answers.** "
             f"{len(rows) - summary['attempted']} selected examples were skipped. "
             f"{summary['delivered_media_observations']} image/video observations were supplied in follow-up model calls.", '',
             caveat, '',
             '## Execution', '', '| Measure | Result |', '| --- | --- |',
             *[f'| {k.replace("_", " ")} | {v:.2f} |' if isinstance(v, float) else f'| {k.replace("_", " ")} | {v} |' for k, v in summary.items()], '',
             'Timing comes from a shared RTX A6000 and is not a throughput benchmark. W&B logging is offline; no public run link exists.', '',
             '## Protocol and adaptations', '', *['- ' + s for s in metadata['adaptations']], '']
    if model == 'chain_of_focus':
        parts += ['The original six-call loop marks exhausted episodes as `success`, even without an answer. '
                  'The completed-answer count above additionally requires its `pred_ans` to be nonempty. '
                  'Invalid bounding boxes can produce a full-image fallback rather than the requested crop; these are counted separately. '
                  'A tool result created after the last allowed call is not counted as delivered.', '']
    if model == 'mini_o3':
        parts += ['The stop `</grounding>` is retained in returned text. Crops use the original relative-coordinate parser '
                  'and can target the original image or an earlier observation. The loop permits at most 12 calls and 12 images. '
                  'Repeated crops remain visible below. The earlier uncached partial run is retained separately in `../mini_o3_run/`.', '']
    if model == 'deepeyes':
        parts += [f"The pinned `{metadata['protocol']}` loop is used with its published stop list `{metadata['stop']}`, "
                  f"{metadata['max_tokens_per_call']} tokens per call and at most {metadata['max_calls']} calls. "
                  'Multiple-choice options are omitted for open-ended questions. Final answers can be followed by an undelivered '
                  f"tool request; {sum(a.get('trailing_tool_request_after_answer', False) for a in audits)} such responses occurred.", '']
    if model == 'video_com' and domain == 'natural':
        parts += ['Initial input: 16 frames spread evenly over the whole clip, at the clip’s real frame rate. '
                  'The author tools may request additional frames or segments from the same annotated clip. '
                  'Frame/segment labels and the mp4v overlay encoder come from the original preprocessing helper.', '',
                  '**This is an adapted standalone inference run.** The release contains training and manipulation code, '
                  'but no standalone evaluation prompt was found. The action-syntax instruction below was reconstructed; '
                  'it is not represented as the authors’ exact evaluation prompt. The run uses greedy decoding, five calls '
                  'and 512 new tokens per call. Returned segments use the released code’s 16-frame setting.', '']
    elif model == 'video_com':
        parts += ['Initial input: 16 distinct frames, a seeded random valid start, stride 2. Short recordings are skipped. '
                  'The author tools may request additional frames from the same full recording. Source PNGs have no acquisition FPS; '
                  'the nominal 2-FPS container is only a processing convention. No physiological timing is inferred. '
                  'Frame/segment labels and the mp4v overlay encoder come from the original preprocessing helper.', '',
                  '**This is an adapted standalone inference run.** The release contains training and manipulation code, '
                  'but no standalone evaluation prompt was found. The action-syntax instruction below was reconstructed; '
                  'it is not represented as the authors’ exact evaluation prompt. The run uses greedy decoding, five calls '
                  'and 512 new tokens per call. Returned segments use the released code’s 16-frame setting.', '']
    parts += ['## Exact prompts and outputs', '']
    for row, audit in zip(rows, audits):
        n = row['index']
        trace = Path(f'example_{n:02d}/trace')
        parts += [f"### {n + 1}. {row['view']} · {row['question_type'].replace('_', ' ')}", '',
                  '**Question:** ' + row['question'], '',
                  '**Model answer:** ' + (row['answer'] or 'No complete answer.'), '',
                  f'**{reference_label}:** ' + row['gold_answer'], '',
                  f"Status: `{row['protocol_status']}`; audited completion: `{audit['completed_answer']}`; "
                  f"{len(row['turns'])} calls; {row['seconds']:.2f} seconds.", '']
        if audit['invalid_bbox_observations']:
            parts += [f"**Invalid crop requests followed by fallback images:** {audit['invalid_bbox_observations']}.", '']
        if audit['repeated_identical_responses']:
            parts += [f"**Repeated identical model responses:** {audit['repeated_identical_responses']}.", '']
        if 'video_input' in row:
            inp = row['video_input']
            parts += [f"Source frames: {inp['source_frame_count']}; stride: {inp['stride']}; "
                      f"selected zero-based indices: `{inp['frame_indices']}`.", '']
            if inp.get('source_fps'):
                parts += [f"Source frame rate: {inp['source_fps']:.3f} fps; duration: {inp['duration_seconds']:.2f} s; "
                          f"initial frames passed at {inp['sample_fps']:.3f} fps.", '']
            if inp.get('reason'):
                parts += [inp['reason'], '']
        if row.get('error'):
            parts += ['**Error:** ' + row['error'], '']
        for number, turn in enumerate(row['turns'], 1):
            request = turn['request']
            parts += [f'#### Call {number}', '']
            if number == 1:
                for msg in request['messages']:
                    content = msg['content']
                    texts = [content] if isinstance(content, str) else [i['text'] for i in content if i['type'] == 'text']
                    if texts:
                        parts += [f"**{msg['role'].capitalize()} prompt (exact text):**", '', '```text', '\n'.join(texts), '```', '']
            for kind, names in media_items(request['messages'][-1]):
                assert isinstance(names, list)
                parts += [f'**{kind.capitalize()} supplied before this call:**', '']
                for frame_number, name in enumerate(names, 1):
                    path = trace / name
                    assert not Path(name).is_absolute() and '..' not in Path(name).parts
                    assert (root / path).is_file(), path
                    with Image.open(root / path) as im:
                        im.verify()
                    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == Path(name).stem
                    parts += [f'![Input {frame_number}]({path.as_posix()})', '']
            raw_output = turn.get('raw_output', '')
            parts += [f"Finish: `{turn.get('finish_reason')}`; generated tokens: "
                      f"{(turn.get('usage') or {}).get('completion_tokens', 'unknown')}.", '']
            if len(raw_output) > 4000:
                parts += [f'<details><summary>Full raw response ({len(raw_output):,} characters)</summary>', '']
            parts += ['```text', raw_output, '```', '']
            if len(raw_output) > 4000:
                parts += ['</details>', '']
            parts += ['<details><summary>Exact full request, including conversation history</summary>', '',
                      '```json', json.dumps(request, indent=2), '```', '', '</details>', '']
    parts += ['## Provenance', '', '[Raw predictions](predictions.jsonl) · [Metadata](metadata.json) · [Execution audit](audit.json)', '',
              '```json', json.dumps(metadata, indent=2), '```', '']
    report = '\n'.join(parts)
    (root / 'report.md').write_text(report)
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file() and 'wandb' not in p.parts and p.name != 'checksums.json'}
    (root / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    return summary


def render_comparison(root):
    root = Path(root)
    titles = dict(chain_of_focus='Chain-of-Focus', mini_o3='Mini-o3', video_com='Video-CoM')
    runs = {}
    domains = set()
    for path in sorted(root.glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        name = run_model(metadata)
        if name in TITLES and metadata['status'] == 'completed':
            if name in runs:
                raise ValueError(f'Multiple completed runs for {name}; choose a comparison directory explicitly')
            audit = render_run(path.parent)
            rows = load_rows(path.parent, metadata)
            runs[name] = (path.parent.name, audit, rows)
            domains.add(metadata.get('domain', 'echo'))
    assert len(domains) == 1, domains
    if domains == {'natural'}:
        return render_natural_comparison(root, runs)
    assert set(runs) == set(titles), 'All three completed runs are required'
    reference = runs['chain_of_focus'][2]
    for _, _, rows in runs.values():
        assert [(r['study_uuid'], r['question'], r['view']) for r in rows] == [(r['study_uuid'], r['question'], r['view']) for r in reference]
    parts = ['# Visual-tool inference comparison', '',
             'Same ten held-out question/view pairs, with view context. Open each model report for its exact '
             'system/user prompts, input images, returned tool evidence and every raw response.', '',
             '| Model report | Completed answers / attempted | Delivered tool observations | Inference seconds |',
             '| --- | --- | --- | --- |']
    for name, title in titles.items():
        folder, audit, _ = runs[name]
        parts.append(f"| [{title}]({folder}/report.md) | {audit['completed_answers']}/{audit['attempted']} | "
                     f"{audit['delivered_media_observations']} | {audit['seconds']:.1f} |")
    parts += ['', 'These are execution counts, not clinical accuracy. Chain-of-Focus and Mini-o3 see one frame; '
              'Video-CoM initially sees 16 frames at stride 2 and skips four recordings with fewer than 31 frames. '
              'Its tools can request additional evidence from the same recording.', '',
              'All three use an explicit Transformers backend adaptation. Video-CoM additionally uses a reconstructed '
              'standalone prompt because a released evaluation prompt was not found. None of these runs is claimed '
              'to reproduce the authors’ numerical benchmark results.', '',
              'The reports distinguish exhausted loops, invalid crop fallbacks, repeated text and truncated generations. '
              'W&B remains offline; no public run links exist.', '', '## Answers by question', '']
    for i, row in enumerate(reference):
        parts += [f"### {i + 1}. {row['view']}", '', row['question'], '',
                  '**Dataset reference:** ' + row['gold_answer'], '']
        for name, title in titles.items():
            result = runs[name][2][i]
            parts += [f'**{title}:** ' + (result['answer'] or f"No complete answer (`{result['protocol_status']}`)."), '']
    (root / 'report.md').write_text('\n'.join(parts))
    return {name: run[1] for name, run in runs.items()}


def render_natural_comparison(root, runs):
    """Image models share one photo sample; Video-CoM answers its own clip sample."""
    image_models = [m for m in ('deepeyes', 'chain_of_focus', 'mini_o3') if m in runs]
    missing = [TITLES[m] for m in TITLES if m not in runs]
    reference = runs[image_models[0]][2] if image_models else []
    for name in image_models:
        rows = runs[name][2]
        assert [(r['study_uuid'], r['question']) for r in rows] == [(r['study_uuid'], r['question']) for r in reference], name
    parts = ['# Visual-tool inference on natural samples', '',
             'Ten sample photographs with one hand-written detail question each, passed to the released image models '
             'without any echo framing. Video-CoM receives three short sample clips instead. Open each model report for '
             'exact prompts, input images, returned tool evidence and every raw response.', '',
             '| Model report | Completed answers / attempted | Delivered tool observations | Inference seconds |',
             '| --- | --- | --- | --- |']
    for name in TITLES:
        if name not in runs:
            continue
        folder, audit, _ = runs[name]
        parts.append(f"| [{TITLES[name]}]({folder}/report.md) | {audit['completed_answers']}/{audit['attempted']} | "
                     f"{audit['delivered_media_observations']} | {audit['seconds']:.1f} |")
    if missing:
        parts += ['', 'Not run in this comparison: ' + ', '.join(missing) + '.']
    parts += ['', 'These are execution counts. Expected answers below were written by hand after viewing each file, are '
              'shown only for inspection and were not independently verified. All runs use an explicit Transformers '
              'backend adaptation; none is claimed to reproduce the authors’ benchmark numbers.', '',
              'W&B remains offline; no public run links exist.', '']
    if image_models:
        parts += ['## Photo answers by question', '']
        for i, row in enumerate(reference):
            parts += [f"### {i + 1}. {row['study_uuid']}", '', row['question'], '',
                      '**Expected (hand-written):** ' + row['gold_answer'], '']
            for name in image_models:
                result = runs[name][2][i]
                parts += [f'**{TITLES[name]}:** ' + (result['answer'] or f"No complete answer (`{result['protocol_status']}`)."), '']
    if 'video_com' in runs:
        parts += ['## Clip answers (Video-CoM)', '']
        for i, row in enumerate(runs['video_com'][2]):
            parts += [f"### {i + 1}. {row['study_uuid']}", '', row['question'], '',
                      '**Expected (hand-written):** ' + row['gold_answer'], '',
                      '**Video-CoM:** ' + (row['answer'] or f"No complete answer (`{row['protocol_status']}`)."), '']
    (root / 'report.md').write_text('\n'.join(parts))
    return {name: run[1] for name, run in runs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir')
    ap.add_argument('--comparison', action='store_true', help='Render all three completed runs under this directory')
    args = ap.parse_args()
    print(json.dumps((render_comparison if args.comparison else render_run)(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
