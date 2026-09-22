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


def render_run(root):
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    assert metadata['status'] == 'completed', metadata['status']
    rows = [json.loads(s) for s in (root / 'predictions.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows) == metadata['selected_studies']
    model = metadata['model']
    title = dict(chain_of_focus='Chain-of-Focus', mini_o3='Mini-o3', video_com='Video-CoM')[model]
    audits = [audit_row(r, root, model) for r in rows]
    summary = dict(selected=len(rows), attempted=sum(bool(r['turns']) for r in rows),
                   completed_answers=sum(a['completed_answer'] for a in audits),
                   delivered_media_observations=sum(a['delivered_media_observations'] for a in audits),
                   invalid_bbox_observations=sum(a['invalid_bbox_observations'] for a in audits),
                   repeated_identical_responses=sum(a['repeated_identical_responses'] for a in audits),
                   truncated_examples=sum(r['truncated'] for r in rows),
                   examples_with_addcriterion=sum(r['addcriterion_count'] > 0 for r in rows),
                   seconds=sum(r['seconds'] for r in rows))
    (root / 'audit.json').write_text(json.dumps(dict(summary=summary, examples=audits), indent=2) + '\n')
    parts = [f'# {title}: held-out echo inference with view context', '',
             f"**{summary['completed_answers']}/{summary['attempted']} attempted examples produced completed answers.** "
             f"{len(rows) - summary['attempted']} selected examples were skipped. "
             f"{summary['delivered_media_observations']} image/video observations were supplied in follow-up model calls.", '',
             'These counts describe inference and tool execution, not clinical accuracy. Dataset answers concern the study; '
             'the model receives evidence from only one randomly selected view. Reference answers are never prompted.', '',
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
    if model == 'video_com':
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
                  '**Dataset reference:** ' + row['gold_answer'], '',
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
            parts += [f"Finish: `{turn.get('finish_reason')}`; generated tokens: "
                      f"{(turn.get('usage') or {}).get('completion_tokens', 'unknown')}.", '',
                      '```text', turn.get('raw_output', ''), '```', '',
                      '<details><summary>Exact full request, including conversation history</summary>', '',
                      '```json', json.dumps(request, indent=2), '```', '', '</details>', '']
    parts += ['## Provenance', '', '[Raw predictions](predictions.jsonl) · [Metadata](metadata.json) · [Execution audit](audit.json)', '',
              '```json', json.dumps(metadata, indent=2), '```', '']
    report = '\n'.join(parts)
    (root / 'report.md').write_text(report)
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file() and 'wandb' not in p.parts and p.name != 'checksums.json'}
    (root / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir')
    args = ap.parse_args()
    print(json.dumps(render_run(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
