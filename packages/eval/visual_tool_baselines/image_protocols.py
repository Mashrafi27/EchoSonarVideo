"""Adapters for pinned Chain-of-Focus and Mini-o3 image protocols."""
import ast
import base64
import importlib.util
from io import BytesIO
from pathlib import Path
import re
from types import SimpleNamespace

from PIL import Image

from .upstream import load_cof, load_definitions


def image_item(image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    return dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()))


def question_with_view(record):
    if record.get('domain') == 'natural':
        # Natural sample photos: the plain question, no echo framing.
        return record['question']
    view = record['overview']['views'][0]['view']
    return record['question'] + '\n\nThis image is a frame from an echocardiography video.\nView: ' + view


def chain_of_focus(root, record, client, system_prompt_suffix=''):
    module = load_cof(root)
    if system_prompt_suffix:
        module.run_inference_loop.__globals__['SYSTEM_PROMPT'] += '\n\n' + system_prompt_suffix
        module.SYSTEM_PROMPT = module.run_inference_loop.__globals__['SYSTEM_PROMPT']

    class ServingAdapter:
        def chat(self, messages, sampling_params):
            response = client.create(messages=messages, model='chain_of_focus',
                                     temperature=sampling_params.temperature,
                                     max_tokens=sampling_params.max_tokens, stop=['<|im_end|>'])
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response.choices[0].message.content,
                                                              token_ids=None)])]

    frame = Path(record['overview']['views'][0]['frame']).resolve()
    data = dict(image=frame.name, text=question_with_view(record), label='')
    args = SimpleNamespace(max_new_tokens=512, vstar_bench_path=str(frame.parent),
                           scaleup_factor=2, enlarge_factor=1.5)
    result = module.run_inference_loop(ServingAdapter(), [data], args, 112 ** 2)[0]
    return dict(upstream_result=result, protocol_status=result['status'],
                system_prompt=module.SYSTEM_PROMPT, user_suffix=module.USER_PROMPT)


def mini_o3(root, record, client, system_prompt_suffix=''):
    root = Path(root)
    path = root / 'verl/trainer/constants.py'
    constants = {}
    exec(compile(path.read_text(), str(path), 'exec'), constants)
    if system_prompt_suffix:
        constants['TOOL_CROP_SYSTEM_PROMPT'] += '\n\n' + system_prompt_suffix
    tools = load_definitions(root / 'verl/workers/rollout/vllm_rollout/function_tools.py',
                             ['crop_image', 'prepare_grounding_inputs_multi_turn'], dict(Image=Image, re=re))
    resize = load_definitions(root / 'verl/utils/dataset/rl_dataset.py', ['process_image']).process_image
    original = Image.open(record['overview']['views'][0]['frame']).convert('RGB')
    first = resize(original, max_pixels=2000000, min_pixels=40000)
    observations, sizes = [original], [first.size]
    messages = [dict(role='system', content=constants['TOOL_CROP_SYSTEM_PROMPT']),
                dict(role='user', content=[image_item(first), dict(type='text', text='\n' + question_with_view(record))])]
    events = []
    status = 'turn_limit'
    for turn in range(12):
        response = client.create(messages=messages, model='mini_o3', temperature=0,
                                 max_tokens=8192, stop=['</grounding>'], include_stop_str_in_output=True,
                                 use_cache=True, decode_token_id_ceiling=151664)
        raw = response.choices[0].message.content
        messages.append(dict(role='assistant', content=raw))
        if not re.findall(r'<grounding>(.*?)</grounding>', raw, re.DOTALL):
            status = ('success' if re.match(r'.*<answer>.*</answer>$', raw, re.DOTALL)
                      and response.choices[0].finish_reason != 'length' else 'invalid_final')
            break
        if len(observations) >= 12 or turn == 11:
            break
        usage = response.usage.model_dump()
        if usage['total_tokens'] >= 32768 - 2000:
            status = 'context_limit'
            break
        try:
            # Preserve the upstream parser; only replace execution of the bbox literal with safe parsing.
            match = re.match(r'.*<grounding>{"bbox_2d": (.*),.*"source": [\'\"](.*)[\'\"]}</grounding>', raw, re.DOTALL)
            bbox, source = match.group(1), match.group(2)
            arguments = dict(bbox_2d=ast.literal_eval(bbox), source=source)
            _, args = tools.prepare_grounding_inputs_multi_turn([arguments], observations, sizes, True)
            crop = tools.crop_image(args[0], args[1], sizes, resize=1)
            resized = resize(crop, max_pixels=2000000, min_pixels=40000)
            prefix, suffix = constants['TOOL_CALL_CROP_MULTI_TRUN_PROMPT'].format(
                action_turn=turn, observation_turn=turn + 1).split('<|vision_start|><|image_pad|><|vision_end|>')
            followup = dict(role='user', content=[dict(type='text', text=prefix), image_item(resized),
                                                  dict(type='text', text=suffix)])
            # The original rollout refuses a tool turn if its resulting context crosses the budget.
            extra_text = '<|im_end|>\n<|im_start|>user\n' + prefix + '<|vision_start|><|image_pad|><|vision_end|>' + suffix + '<|im_end|>\n<|im_start|>assistant\n'
            backend = client.client
            grid = backend.processor.image_processor([resized], return_tensors='pt')['image_grid_thw'][0]
            extra = len(backend.processor.tokenizer.encode(extra_text)) + int(grid.prod()) // 4 - 1
            if usage['total_tokens'] + extra >= 32768 - 2000:
                status = 'context_limit'
                break
            observations.append(crop)
            sizes.append(resized.size)
            messages.append(followup)
            events.append(dict(turn=turn + 1, arguments=arguments, pixel_bbox=args[1], status='crop_delivered'))
        except Exception as exc:
            error = str(exc)
            messages.append(dict(role='user', content='ERROR occurs during grounding. Error Information: ' + error + '.\n' + constants['ERROR_INFO_MULTI_TURN_PROMPT']))
            events.append(dict(turn=turn + 1, status='tool_error', error=error))
    return dict(protocol_status=status, tool_events=events,
                system_prompt=constants['TOOL_CROP_SYSTEM_PROMPT'],
                observation_prompt=constants['TOOL_CALL_CROP_MULTI_TRUN_PROMPT'])
