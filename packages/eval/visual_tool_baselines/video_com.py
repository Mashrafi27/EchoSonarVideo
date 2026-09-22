"""Video-CoM adapter using its original vision and manipulation helpers."""
import copy
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image

from eval.deepeyes_transformers_client import TransformersClient


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The release provides training/tool code but no standalone evaluation prompt.
# This open-ended instruction exposes the syntax consumed by the released parser.
USER_INSTRUCTION = '''Reason step by step using the visual evidence. If needed, request a manipulation and wait for its returned evidence before continuing:
FIND_SEGMENT(description) = [segment numbers]
FIND_FRAME(description) = [frame numbers]
SPATIAL_ZOOM(description) = [x1, y1, x2, y2]
Use the overlaid segment and frame numbers, not timestamps. Select a frame before spatially zooming into it. Finish with FINAL_ANSWER: followed by your answer.'''


class VideoBackend(TransformersClient):
    def __init__(self, model_path, root):
        super().__init__(model_path)
        self.vision = import_file('video_com_vision', Path(root) / 'train/vision_process.py')
        self.capture_dir = None

    def save_image(self, image):
        buf = BytesIO()
        image.save(buf, format='PNG')
        raw = buf.getvalue()
        name = 'images/' + hashlib.sha256(raw).hexdigest() + '.png'
        path = self.capture_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return name

    def create(self, **params):
        import torch
        import numpy as np
        images, videos, video_kwargs = self.vision.process_vision_info(params['messages'], return_video_kwargs=True)
        saved = copy.deepcopy(params)
        image_index = video_index = 0
        for message in saved['messages']:
            if not isinstance(message['content'], list):
                continue
            for item in message['content']:
                if item['type'] == 'image':
                    item['image'] = self.save_image(images[image_index])
                    image_index += 1
                elif item['type'] == 'video':
                    video = videos[video_index]
                    frame_paths = []
                    for frame in video:
                        if isinstance(frame, torch.Tensor):
                            frame = Image.fromarray(frame.permute(1, 2, 0).cpu().clamp(0, 255).byte().numpy())
                        elif not isinstance(frame, Image.Image):
                            frame = Image.fromarray(np.asarray(frame))
                        frame_paths.append(self.save_image(frame))
                    item.update(video=frame_paths, fps=video_kwargs['fps'][video_index])
                    video_index += 1
        self.last_request = saved
        prompt = self.processor.apply_chat_template(params['messages'], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, videos=videos, padding=True,
                                return_tensors='pt', do_sample_frames=False, **video_kwargs).to(self.model.device)
        self.last_video_grid = inputs.get('video_grid_thw').tolist() if 'video_grid_thw' in inputs else []
        if videos:
            assert [int(g[0]) * 2 for g in self.last_video_grid] == [len(v) for v in videos], 'Unexpected temporal resampling'
        prompt_tokens = inputs.input_ids.shape[1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=params['max_tokens'], do_sample=False)
        ids = output[0, prompt_tokens:].tolist()
        raw = self.processor.decode(ids, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = 'length' if len(ids) >= params['max_tokens'] and ids[-1] not in eos else 'stop'
        usage = dict(prompt_tokens=prompt_tokens, completion_tokens=len(ids), total_tokens=prompt_tokens + len(ids))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=reason)],
                               usage=SimpleNamespace(model_dump=lambda: usage))


class VideoRecordingClient:
    def __init__(self, backend, directory):
        self.client = backend
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turns = []

    def create(self, **params):
        start = time.monotonic()
        self.client.capture_dir = self.directory
        self.client.last_request = None
        turn = dict(request=copy.deepcopy(params))
        self.turns.append(turn)
        try:
            response = self.client.create(**params)
            turn.update(request=self.client.last_request, raw_output=response.choices[0].message.content,
                        finish_reason=response.choices[0].finish_reason, usage=response.usage.model_dump(),
                        video_grid_thw=self.client.last_video_grid)
            return response
        except Exception as exc:
            turn['error'] = f'{type(exc).__name__}: {exc}'
            if self.client.last_request is not None:
                turn['request'] = self.client.last_request
            raise
        finally:
            turn['seconds'] = time.monotonic() - start
            (self.directory / 'turns.json').write_text(json.dumps(self.turns, indent=2) + '\n')


def run_video(root, record, index, client, manifest_path, out):
    row = json.loads(Path(manifest_path).read_text())['examples'][index]
    assert row['study_uuid'] == record['study_uuid']
    if row['status'] != 'ready':
        return dict(protocol_status='skipped_short_recording', video_input=row)
    # The original module uses these paths only for generated local tool media.
    os.environ['OUTPUT_DIR'] = str((out / f'example_{index:02d}' / 'tools').resolve())
    os.environ['DATA_FOLDER'] = str(Path(row['annotated_video']).parent)
    sys.path.insert(0, str(Path(root) / 'train'))
    try:
        module = import_file('video_com_manipulations', Path(root) / 'train/manipulation_model.py')
    finally:
        sys.path.pop(0)
    first_video = dict(type='video', video=row['sampled_annotated_frames'], fps=row['sample_fps'], max_pixels=360*420)
    question = (record['question'] + '\n\nThese are 16 sampled frames from one echocardiography video.\nView: '
                + record['overview']['views'][0]['view'] + '\nAcquisition timing is unavailable; frame order is preserved.\n\n' + USER_INSTRUCTION)
    messages = [dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content=[first_video, dict(type='text', text=question)])]
    base_media = [dict(type='video', video=row['annotated_video'])]
    frame_number = None
    events = []
    status = 'turn_limit'
    for turn in range(5):
        response = client.create(messages=messages, model='video_com', temperature=0, max_tokens=512, stop=[])
        raw = response.choices[0].message.content
        messages.append(dict(role='assistant', content=raw))
        if 'FINAL_ANSWER' in raw:
            status = 'success' if 'FINAL_ANSWER:' in raw and response.choices[0].finish_reason != 'length' else 'invalid_final'
            break
        if turn == 4:
            break
        followup, frame_number = module.get_next_user_input(raw, base_media, frame_number, None)
        for item in followup:
            if item.get('text') == 'Choose the final answer from the provided options.':
                item['text'] = 'Answer the question in your own words using FINAL_ANSWER:.'
        events.append(dict(turn=turn+1, frame_number=frame_number,
                           manipulations=[name for name in module.manipulations if name in raw],
                           returned_media=sum(c['type'] in ('image', 'video') for c in followup)))
        messages.append(dict(role='user', content=followup))
    return dict(protocol_status=status, video_input=row, tool_events=events,
                system_prompt='You are a helpful assistant.', user_instruction=USER_INSTRUCTION)
