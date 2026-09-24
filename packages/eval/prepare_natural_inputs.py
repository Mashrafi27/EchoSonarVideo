"""Build natural-domain sample inputs (photos and short clips) in the baseline record schema."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from eval.visual_tool_baselines.upstream import verify

QUESTIONS = Path(__file__).with_name('natural_samples_20260924.json')
PROVENANCE = 'hand-written expected answer after viewing the file; not a benchmark label'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uniform_indices(frame_count, count=16):
    """Sixteen distinct frame indices spread evenly over the whole clip."""
    if frame_count < count:
        return None
    return sorted({round(i * (frame_count - 1) / (count - 1)) for i in range(count)})


def image_records(spec, images_dir):
    from PIL import Image
    records = []
    for item in spec['images']:
        path = (Path(images_dir) / item['file']).resolve()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            size = im.size
        records.append(dict(study_uuid='natural/' + item['id'], domain='natural', question=item['question'],
                            answer=item['expected'], answer_provenance=PROVENANCE, question_type='natural_detail',
                            source={k: item[k] for k in ('picsum_id', 'source_url', 'download_url')},
                            overview=dict(views=[dict(view='photo', frame=str(path), frame_count=1,
                                                     size=list(size), sha256=sha256(path))])))
    return records


def video_records(spec, videos_dir, upstream, out):
    import cv2
    pin = verify(upstream, 'video_com')
    source = Path(upstream) / 'train/data_utils.py'
    module_spec = importlib.util.spec_from_file_location('video_com_labels', source)
    labels = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(labels)
    manifest = dict(upstream_commit=pin, initial_frames=16, sampling='16 evenly spaced frames over the whole clip',
                    short_clip_policy='skip', examples=[])
    records = []
    for n, item in enumerate(spec['videos']):
        src = (Path(videos_dir) / item['file']).resolve()
        directory = out / f'example_{n:02d}'
        directory.mkdir()
        copied = directory / ('source' + src.suffix)
        shutil.copy2(src, copied)
        cap = cv2.VideoCapture(str(copied))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        annotated = directory / 'annotated.mp4'
        labels.overlay_segment_frame_labels_all_frames(str(copied), str(annotated))
        cap = cv2.VideoCapture(str(annotated))
        annotated_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected = uniform_indices(annotated_count)
        sampled = []
        try:
            for frame_index in selected or []:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                assert ok, frame_index
                path = directory / f'frame_{frame_index:04d}.png'
                assert cv2.imwrite(str(path), frame)
                sampled.append(str(path))
        finally:
            cap.release()
        uuid = 'natural/' + item['id']
        row = dict(index=n, study_uuid=uuid, view='clip', source_file=str(src), source_sha256=sha256(src),
                   source_fps=fps, source_frame_count=count, annotated_frame_count=annotated_count,
                   duration_seconds=count / fps if fps else None, stride='uniform', frame_indices=selected,
                   status='ready' if selected else 'skipped_short_recording')
        if selected:
            row.update(annotated_video=str(annotated), sample_fps=len(selected) / (count / fps),
                       original_size=[width, height], displayed_frame_numbers=[i + 1 for i in selected],
                       sampled_annotated_frames=sampled, sampled_sha256=[sha256(p) for p in sampled])
        manifest['examples'].append(row)
        records.append(dict(study_uuid=uuid, domain='natural', question=item['question'], answer=item['expected'],
                            answer_provenance=PROVENANCE, question_type='natural_detail',
                            source={k: item[k] for k in ('download_url', 'license')},
                            overview=dict(views=[dict(view='clip', frame=sampled[0] if sampled else None,
                                                     frame_count=count, sha256=sha256(src))])))
    return manifest, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--videos-dir')
    ap.add_argument('--upstream', help='Pinned Video-CoM clone; required with --videos-dir')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--questions', default=str(QUESTIONS))
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'sample.jsonl').exists():
        raise FileExistsError(out / 'sample.jsonl')
    spec = json.loads(Path(args.questions).read_text())
    images = image_records(spec, args.images_dir)
    (out / 'sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in images))
    summary = dict(images=len(images), questions_sha256=sha256(args.questions))
    if args.videos_dir:
        if not args.upstream:
            ap.error('--upstream is required with --videos-dir')
        video_dir = out / 'video_inputs'
        video_dir.mkdir()
        manifest, records = video_records(spec, args.videos_dir, args.upstream, video_dir)
        (video_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (out / 'video_sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        summary.update(videos=len(records), ready=sum(r['status'] == 'ready' for r in manifest['examples']))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
