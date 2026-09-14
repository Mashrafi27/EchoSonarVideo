"""Regression preflight for VeRL tool responses containing multiple images."""
import argparse
from PIL import Image
from transformers import AutoProcessor
from verl_bridge.agent_loop import expand_tool_image_placeholders

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
args = ap.parse_args()
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
red = Image.new('RGB', (160, 160), 'red')
blue = Image.new('RGB', (160, 160), 'blue')
source = [{'role':'tool','content':[{'type':'image'}, {'type':'text','text':'Two preview frames.'}]}]
repaired = expand_tool_image_placeholders(source, 2)
assert len(source[0]['content']) == 2, 'Input messages were mutated'
assert [item['type'] for item in repaired[0]['content']] == ['image', 'image', 'text']
# Include an existing image to exercise the next-turn accumulated-image contract.
history = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':'Inspect another view.'}]},
           {'role':'assistant','content':'<tool_call>{"name":"echo","arguments":{"op":"select_view","view_name":"A4C"}}</tool_call>'}]
prompt = processor.apply_chat_template(history + repaired, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[red, red, blue], return_tensors='pt', do_sample_frames=False)
marker = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
assert int((inputs['input_ids'] == marker).sum()) == 3
assert len(inputs['image_grid_thw']) == 3
try:
    expand_tool_image_placeholders(source + source, 3)
except ValueError:
    pass
else:
    raise AssertionError('Ambiguous parallel image mapping was accepted')
print('ECHO_MULTI_IMAGE_PLACEHOLDER_PASS existing_images=1 returned_images=2 markers=3', flush=True)
