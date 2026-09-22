"""Exercise the original DeepEyes tool loop with synthetic server responses."""
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from eval.run_deepeyes_tools import load_upstream, run_record


@pytest.fixture
def upstream(tmp_path):
    source = Path(__file__).resolve().parents[3] / 'external/DeepEyes/eval/eval_vstar.py'
    if not source.exists():
        pytest.skip('Initialize the pinned DeepEyes submodule for protocol tests')
    return load_upstream(source, 'http://127.0.0.1:9/v1', tmp_path)


def record(tmp_path):
    frame = tmp_path / 'frame.png'
    image = Image.new('RGB', (112, 112))
    image.putdata([(x, y, 0) for y in range(112) for x in range(112)])
    image.save(frame)
    return dict(question='Question sentinel?', answer='Reference sentinel',
                question_type='structure_description',
                overview=dict(views=[dict(view='A4C', frame=str(frame))]))


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **params):
        import copy
        self.requests.append(copy.deepcopy(params))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(self.outputs)), finish_reason='stop')],
            usage=SimpleNamespace(model_dump=lambda: {'completion_tokens': 10}))


def test_original_tool_loop_sends_real_crop_then_continues(upstream, tmp_path):
    rec = record(tmp_path)
    tool = ('<think>Inspect.</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":{"bbox_2d":[28,28,84,84],"label":"region"}}</tool_call>')
    client = FakeClient([tool, '<think>Done.</think><answer>Example answer.</answer>'])
    result = run_record(upstream, rec, 0, tmp_path / 'result', client)
    assert result['status'] == 'success'
    assert result['answer'] == 'Example answer.'
    assert result['tool_calls_requested'] == result['tool_observations_delivered'] == 1
    assert len(client.requests) == 2
    for request in client.requests:
        assert request['temperature'] == 0.0
        assert request['max_tokens'] == 10240
        assert request['stop'] == ['<|im_end|>']
        assert request['messages'][0]['content'] == upstream.instruction_prompt_system
        assert 'Reference sentinel' not in str(request)
    first = client.requests[0]['messages'][1]['content']
    assert first[1]['text'] == 'Question: Question sentinel?\n' + upstream.USER_PROMPT_V2
    continuation = client.requests[1]['messages'][-1]
    assert continuation['role'] == 'user'
    assert continuation['content'][0]['text'] == '<tool_response>'
    assert continuation['content'][-1]['text'] == '</tool_response>'
    assert continuation['content'][2]['text'] == upstream.USER_PROMPT_V2
    raw = base64.b64decode(continuation['content'][1]['image_url']['url'].split(',', 1)[1])
    with Image.open(BytesIO(raw)) as crop, Image.open(rec['overview']['views'][0]['frame']) as original:
        expected = original.crop((28, 28, 84, 84))
        assert crop.size == expected.size
        assert crop.tobytes() == expected.tobytes()


def test_model_generated_python_is_not_executed(upstream, tmp_path):
    marker = tmp_path / 'must_not_exist'
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    client = FakeClient(['<tool_call>' + payload + '</tool_call>'])
    result = run_record(upstream, record(tmp_path), 0, tmp_path / 'result', client)
    assert result['status'] == 'error'
    assert result['tool_observations_delivered'] == 0
    assert not marker.exists()


def test_original_retry_limit_and_no_fabricated_final_answer(upstream, tmp_path):
    client = FakeClient(['<think>Still considering.</think>'] * 11)
    result = run_record(upstream, record(tmp_path), 0, tmp_path / 'result', client)
    assert len(client.requests) == 11
    assert result['answer'] is None
    assert result['tool_calls_requested'] == result['tool_observations_delivered'] == 0


def test_hrbench_stop_boundary_returns_crop_before_next_generation(tmp_path):
    source = Path(__file__).resolve().parents[3] / 'external/DeepEyes/eval/eval_hrbench.py'
    if not source.exists():
        pytest.skip('Initialize the pinned DeepEyes submodule for protocol tests')
    module = load_upstream(source, 'http://127.0.0.1:9/v1', tmp_path)
    # The API excludes its matched stop string from the returned content.
    client = FakeClient(['<think>Inspect.</think><tool_call>{"name":"image_zoom_in_tool",'
                         '"arguments":{"bbox_2d":[28,28,84,84],"label":"region"}}',
                         '<think>Done.</think><answer>Example answer.</answer>'])
    rec = record(tmp_path)
    result = run_record(module, rec, 0, tmp_path / 'result', client)
    assert result['answer'] == 'Example answer.'
    assert result['tool_observations_delivered'] == 1
    assert len(client.requests) == 2
    for request in client.requests:
        assert request['stop'] == ['<|im_end|>', '</tool_call>']
        assert request['max_tokens'] == 8192
        assert 'Reference sentinel' not in str(request)
    content = client.requests[1]['messages'][-1]['content']
    raw = base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1])
    with Image.open(BytesIO(raw)) as crop, Image.open(rec['overview']['views'][0]['frame']) as original:
        assert crop.tobytes() == original.crop((28, 28, 84, 84)).tobytes()
