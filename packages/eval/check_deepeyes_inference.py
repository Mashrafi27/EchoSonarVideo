"""Replay one saved DeepEyes request through Transformers and the vLLM API."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import time


class ProgressRecorder:
    """Save generated-token progress without changing generation or decoding."""

    def __init__(self, path):
        self.path = Path(path)
        self.prompt_seen = False
        self.ids = []
        self.start = time.monotonic()

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.ids.extend(value.reshape(-1).tolist())
        if len(self.ids) % 16 == 0:
            self.end()

    def end(self):
        state = dict(generated_tokens=len(self.ids), seconds=time.monotonic() - self.start,
                     output_ids=self.ids)
        self.path.write_text(json.dumps(state) + '\n')
        print(f'GENERATION_PROGRESS tokens={len(self.ids)} seconds={state["seconds"]:.1f}', flush=True)


def load_request(trace_dir, turn):
    trace_dir = Path(trace_dir)
    saved = json.loads((trace_dir / 'turns.json').read_text())[turn]
    request = copy.deepcopy(saved['request'])
    images = []
    from PIL import Image
    hf_messages = copy.deepcopy(request['messages'])
    for api_message, hf_message in zip(request['messages'], hf_messages):
        if not isinstance(api_message['content'], list):
            continue
        for api_item, hf_item in zip(api_message['content'], hf_message['content']):
            if api_item['type'] == 'image_url':
                path = trace_dir / api_item['image_url']['url']
                raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == path.stem
                images.append(Image.open(path).convert('RGB'))
                api_item['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
                hf_item.clear()
                hf_item.update(type='image')
    return request, hf_messages, images, saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['transformers', 'vllm_api'], required=True)
    ap.add_argument('--model', default='checkpoints/deepeyes_7b')
    ap.add_argument('--trace-dir', required=True)
    ap.add_argument('--turn', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--api-url')
    ap.add_argument('--device', choices=['cuda:0', 'cpu'], default='cuda:0')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.backend + '.json')
    if target.exists():
        raise FileExistsError(target)
    request, messages, images, saved = load_request(args.trace_dir, args.turn)
    request['max_tokens'] = args.max_tokens
    start = time.monotonic()
    result = dict(backend=args.backend, device=args.device, trace_dir=args.trace_dir, turn=args.turn,
                  diagnostic_max_tokens=args.max_tokens, original_usage=saved['usage'])
    if args.backend == 'transformers':
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import os
        torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
        torch.manual_seed(1)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=True)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, padding=True, return_tensors='pt')
        result.update(prompt=prompt, input_ids=inputs.input_ids[0].tolist(),
                      image_grid_thw=inputs.image_grid_thw.tolist(),
                      pixel_values_shape=list(inputs.pixel_values.shape),
                      pixel_values_sha256=hashlib.sha256(inputs.pixel_values.numpy().tobytes()).hexdigest())
        torch.save(dict(inputs), out / 'transformers_inputs.pt')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True,
            torch_dtype=torch.float32 if args.device == 'cpu' else torch.bfloat16,
            attn_implementation='sdpa', device_map=args.device).eval()
        inputs = inputs.to(model.device)
        progress = ProgressRecorder(out / 'transformers_progress.json')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                       do_sample=False, return_dict_in_generate=True,
                                       streamer=progress)
        ids = generated.sequences[0, inputs.input_ids.shape[1]:].tolist()
        result.update(output_ids=ids, raw_output=processor.decode(ids, skip_special_tokens=True),
                      output_with_special_tokens=processor.decode(ids, skip_special_tokens=False),
                      generation_config=model.generation_config.to_dict(),
                      input_tokens=inputs.input_ids.shape[1], output_tokens=len(ids))
    else:
        from urllib.parse import urlsplit
        from openai import OpenAI
        import requests
        if urlsplit(args.api_url).hostname not in ('127.0.0.1', 'localhost'):
            ap.error('The inference API must be local')
        client = OpenAI(base_url=args.api_url, api_key='EMPTY')
        response = client.chat.completions.create(**request)
        result.update(raw_output=response.choices[0].message.content,
                      finish_reason=response.choices[0].finish_reason,
                      usage=response.usage.model_dump())
        boundary_request = copy.deepcopy(request)
        boundary_request['stop'].append('</tool_call>')
        boundary = client.chat.completions.create(**boundary_request)
        boundary_text = boundary.choices[0].message.content
        result['tool_boundary'] = dict(raw_output=boundary_text,
            finish_reason=boundary.choices[0].finish_reason, usage=boundary.usage.model_dump(),
            stop=boundary_request['stop'], prefix_matches=result['raw_output'].startswith(boundary_text))
        # /tokenize before generation can populate only the sender multimodal
        # cache in vLLM 0.17, leaving the engine without the expected image.
        tokenized = requests.post(args.api_url.removesuffix('/v1') + '/tokenize',
                                  json={'model': request['model'], 'messages': request['messages']},
                                  timeout=90)
        result['tokenize_status'] = tokenized.status_code
        result['tokenize_result'] = tokenized.json()
    result['seconds'] = time.monotonic() - start
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k in (
        'backend', 'input_tokens', 'output_tokens', 'seconds', 'usage', 'finish_reason')}), flush=True)
    print('addCriterion occurrences:', result['raw_output'].count('addCriterion'), flush=True)


if __name__ == '__main__':
    main()
