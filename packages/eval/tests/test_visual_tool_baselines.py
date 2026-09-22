"""Real pinned protocol loops with synthetic model responses and real image crops."""
import base64
import copy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from eval.visual_tool_baselines.image_protocols import chain_of_focus, mini_o3

ROOT = Path(__file__).resolve().parents[3] / 'build/visual_baselines_20260922'


def record(tmp_path):
    path = tmp_path / 'frame.png'
    image = Image.new('RGB', (224, 224))
    image.putdata([(x, y, 0) for y in range(224) for x in range(224)])
    image.save(path)
    return dict(question='Question sentinel?', answer='Never-prompt-this-reference',
                overview=dict(views=[dict(frame=str(path), view='A4C')]))


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s: list(s.encode())),
                                    image_processor=lambda *a, **k: {'image_grid_thw': np.array([[1, 8, 8]])})
        self.client = SimpleNamespace(processor=processor)

    def create(self, **params):
        self.requests.append(copy.deepcopy(params))
        assert 'Never-prompt-this-reference' not in str(params)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.outputs)),
                                                       finish_reason='stop')],
                               usage=SimpleNamespace(model_dump=lambda: dict(total_tokens=100)))


def require_source(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip('Fetch pinned upstream source to run protocol checks')
    return path


def decode_image(item):
    return Image.open(BytesIO(base64.b64decode(item['image_url']['url'].split(',', 1)[1])))


def test_cof_original_loop_delivers_real_zoom_and_keeps_view_context(tmp_path):
    source = require_source('chain_of_focus')
    from eval.visual_tool_baselines.upstream import load_cof
    raw = '<think>Inspect.</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[56,56,112,112]}}</tool_call>'
    client = FakeClient([raw, '<answer>Result.</answer>'])
    rec = record(tmp_path)
    result = chain_of_focus(source, rec, client)
    assert result['protocol_status'] == 'success'
    assert len(client.requests) == 2
    first, second = client.requests
    assert first['max_tokens'] == 512
    assert 'View: A4C' in str(first['messages'])
    crop = decode_image(second['messages'][-1]['content'][0])
    expected = load_cof(source).do_crop(Image.open(rec['overview']['views'][0]['frame']), raw,
                                       scaleup_factor=2, enlarge_factor=1.5, min_pixels=112**2)
    assert crop.size == (168, 168)
    assert crop.tobytes() == expected.tobytes()
    assert second['messages'][-2]['content'][0]['text'] == raw


def test_mini_o3_relative_crop_observation_can_be_cropped_again(tmp_path):
    source = require_source('mini_o3')
    client = FakeClient([
        '<grounding>{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "source": "original_image"}</grounding>',
        '<grounding>{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "source": "observation_1"}</grounding>',
        '<answer>Result.</answer>'])
    result = mini_o3(source, record(tmp_path), client)
    assert result['protocol_status'] == 'success'
    assert len(client.requests) == 3
    assert [e['pixel_bbox'] for e in result['tool_events']] == [[56, 56, 168, 168], [28, 28, 84, 84]]
    assert all(r['include_stop_str_in_output'] for r in client.requests)
    assert all(r['stop'] == ['</grounding>'] for r in client.requests)
    last = client.requests[-1]['messages'][-1]['content']
    assert 'Observation 2' in last[0]['text']
    crop = decode_image(last[1])
    assert crop.getpixel((0, 0)) == (84, 84, 0)


def test_mini_o3_tool_error_is_returned_without_executing_model_code(tmp_path):
    source = require_source('mini_o3')
    marker = tmp_path / 'must_not_exist'
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    client = FakeClient(['<grounding>{"bbox_2d": ' + payload + ', "source": "original_image"}</grounding>',
                         '<answer>Result.</answer>'])
    result = mini_o3(source, record(tmp_path), client)
    assert not marker.exists()
    assert result['tool_events'][0]['status'] == 'tool_error'
    assert 'ERROR occurs during grounding' in client.requests[1]['messages'][-1]['content']
