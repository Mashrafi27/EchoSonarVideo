"""Prepare auditable 16-frame Video-CoM inputs from ordered local recordings."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random

from eval.visual_tool_baselines.upstream import verify


def sample_indices(frame_count, rng, stride=2, count=16):
    """Choose uniformly among valid starts; never pad or wrap a short recording."""
    span = (count - 1) * stride + 1
    if frame_count < span:
        return None
    start = rng.randrange(frame_count - span + 1)
    return list(range(start, start + span, stride))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-sample', required=True)
    ap.add_argument('--dicom-index', required=True)
    ap.add_argument('--upstream', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--short-clip-policy', choices=['skip', 'stride_one'], default='skip')
    args = ap.parse_args()
    pin = verify(args.upstream, 'video_com')
    import cv2
    source = Path(args.upstream) / 'train/data_utils.py'
    spec = importlib.util.spec_from_file_location('video_com_labels', source)
    labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labels)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    records = [json.loads(s) for s in Path(args.source_sample).read_text().splitlines() if s.strip()]
    index = json.loads(Path(args.dicom_index).read_text())
    rng = random.Random(args.seed)
    manifest = dict(seed=args.seed, upstream_commit=pin, initial_frames=16, requested_stride=2,
                    short_clip_policy=args.short_clip_policy, nominal_container_fps=2,
                    acquisition_fps=None, encoding='FFV1 source container; original mp4v overlay helper',
                    sampling='uniform valid start in the entire selected view, then fixed stride', examples=[])
    for n, record in enumerate(records):
        view = record['overview']['views'][0]
        dicom_id = Path(view['frame']).parent.name.split('_', 1)[0]
        folder = Path(index[dicom_id])
        frames = sorted((p for p in folder.glob('*.png') if p.stem.isdigit()), key=lambda p: int(p.stem))
        assert frames and [int(p.stem) for p in frames] == list(range(len(frames))), folder
        assert len(frames) == view['frame_count'], (dicom_id, len(frames), view['frame_count'])
        stride = 1 if len(frames) < 31 and args.short_clip_policy == 'stride_one' else 2
        selected = sample_indices(len(frames), rng, stride=stride)
        row = dict(index=n, study_uuid=record['study_uuid'], view=view['view'], dicom_id=dicom_id,
                   source_directory=str(folder), source_frame_count=len(frames), stride=stride,
                   frame_indices=selected, status='ready' if selected else 'skipped_short_recording')
        manifest['examples'].append(row)
        if selected is None:
            row['reason'] = '16 distinct frames at stride 2 require at least 31 source frames'
            continue
        directory = out / f'example_{n:02d}'
        directory.mkdir()
        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        video_path = directory / 'source.avi'
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'FFV1'), 2, (width, height))
        assert writer.isOpened(), 'FFV1 writer unavailable'
        hashes = []
        try:
            for path in frames:
                frame = cv2.imread(str(path))
                assert frame is not None and frame.shape == first.shape, path
                writer.write(frame)
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        finally:
            writer.release()
        annotated = directory / 'annotated.mp4'
        labels.overlay_segment_frame_labels_all_frames(str(video_path), str(annotated))
        cap = cv2.VideoCapture(str(annotated))
        sampled = []
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
            for frame_index in selected:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                assert ok, frame_index
                path = directory / f'frame_{frame_index:04d}.png'
                assert cv2.imwrite(str(path), frame)
                sampled.append(str(path))
        finally:
            cap.release()
        row.update(annotated_video=str(annotated), sample_fps=2 / stride,
                   source_frame_sha256=hashes, original_size=[width, height],
                   displayed_frame_numbers=[i + 1 for i in selected], sampled_annotated_frames=sampled,
                   sampled_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sampled])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(manifest_path), ready=sum(r['status'] == 'ready' for r in manifest['examples']),
                         skipped=sum(r['status'] != 'ready' for r in manifest['examples']))))


if __name__ == '__main__':
    main()
