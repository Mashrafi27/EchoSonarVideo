"""Local Transformers backend for the original DeepEyes evaluation functions."""
from __future__ import annotations

import base64
import copy
from io import BytesIO
from types import SimpleNamespace


class TransformersClient:
    def __init__(self, model_path):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        torch.manual_seed(1)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map='cuda:0').eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **params):
        import torch
        from PIL import Image
        if params['temperature'] != 0:
            raise ValueError('This reference backend implements greedy decoding only')
        messages = copy.deepcopy(params['messages'])
        images = []
        for message in messages:
            if not isinstance(message['content'], list):
                continue
            for item in message['content']:
                if item['type'] == 'image_url':
                    url = item['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise ValueError('Expected an inline image from the original evaluation loop')
                    with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
                        images.append(image.convert('RGB'))
                    item.clear()
                    item['type'] = 'image'
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors='pt').to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=params['max_tokens'], do_sample=False,
                                         stop_strings=params['stop'], tokenizer=self.processor.tokenizer)
        ids = output[0, input_tokens:].tolist()
        raw = self.processor.decode(ids, skip_special_tokens=True)
        # OpenAI-compatible servers omit a matched stop string from content.
        positions = [raw.find(stop) for stop in params['stop'] if stop in raw]
        matched = bool(positions)
        if positions:
            raw = raw[:min(positions)]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = bool(ids and ids[-1] in eos_ids) or matched
        reason = 'length' if len(ids) >= params['max_tokens'] and not ended else 'stop'
        usage = dict(prompt_tokens=input_tokens, completion_tokens=len(ids), total_tokens=input_tokens + len(ids))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=reason)],
                               usage=SimpleNamespace(model_dump=lambda: usage))
