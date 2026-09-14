"""GPU runtime validation: text, images, and a tool-response conversation.

Run under Slurm with scripts/rocm_workspace_scratch.sh. Inputs are synthetic;
this is an infrastructure test, not an echocardiography accuracy experiment.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import re
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {'stage': 'qwen_vllm_inference', 'model': args.model,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': platform.node(),
              'synthetic_inputs': True, 'checks': [], 'pass': False}
    run = None
    try:
        import wandb
        run = wandb.init(project='echo-runtime-validation', entity='anaatef9-mbzuai',
                         name=f'qwen3vl8b-rocm-inference-{result["slurm_job_id"]}',
                         dir=str(out_dir), config={k:v for k,v in result.items() if k != 'checks'},
                         settings=wandb.Settings(init_timeout=60))
        result['wandb_url'] = run.url
        print('WANDB_URL=' + run.url, flush=True)
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform
        from transformers import AutoProcessor
        from PIL import Image
        result.update(torch=torch.__version__, hip=torch.version.hip,
                      vllm=vllm.__version__, device=torch.cuda.get_device_name(0),
                      architecture=torch.cuda.get_device_properties(0).gcnArchName,
                      platform=type(current_platform).__name__)
        assert torch.version.hip and current_platform.is_rocm(), result
        x = torch.ones(256, device='cuda')
        assert x.sum().item() == 256
        a = torch.randn(128,128,dtype=torch.bfloat16,device='cuda')
        assert torch.isfinite(a @ a.T).all().item()
        conv = torch.nn.Conv2d(3,16,16,stride=16).to(device='cuda',dtype=torch.bfloat16)
        assert torch.isfinite(conv(torch.ones(1,3,256,256,device='cuda',dtype=torch.bfloat16))).all().item()
        result['loaded_rocm_libraries']=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if any(name in line for name in ('libamdhip64','libhsa-runtime64','libamd_comgr','libMIOpen','librocblas','libhipblas','librccl'))))
        runtime=Path(os.environ['ROCM_PATH']).resolve()
        assert result['loaded_rocm_libraries']
        assert all(Path(p).resolve().is_relative_to(runtime) for p in result['loaded_rocm_libraries']), result['loaded_rocm_libraries']
        result['private_rocm_runtime_verified']=True
        print('PRIVATE_ROCM_LIBRARIES',json.dumps(result['loaded_rocm_libraries']),flush=True)
        del a, x, conv
        torch.cuda.empty_cache()
        result['checks'].append({'name':'torch_and_vision_convolution','pass':True})
        print('TORCH_GPU_GATE_PASS', flush=True)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        engine_args = dict(model=args.model, dtype='bfloat16', tensor_parallel_size=1,
                           gpu_memory_utilization=0.70, max_model_len=4096,
                           max_num_seqs=2, max_num_batched_tokens=4096,
                           enforce_eager=True, enable_prefix_caching=False,
                           limit_mm_per_prompt={'image':4,'video':0},
                           attention_backend='TRITON_ATTN', mm_encoder_attn_backend='TORCH_SDPA',
                           disable_custom_all_reduce=True, seed=0)
        result['engine_args'] = engine_args
        start=time.monotonic()
        llm=LLM(**engine_args)
        result['model_load_seconds']=time.monotonic()-start
        print('VLLM_ENGINE_READY',flush=True)
        def generate(name, messages, images, expected):
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            request={'prompt':prompt}
            if images: request['multi_modal_data']={'image':images}
            start=time.monotonic()
            response=llm.generate([request],SamplingParams(temperature=0,max_tokens=128),use_tqdm=False)[0]
            text=response.outputs[0].text
            check={'name':name,'prompt':prompt,'response':text,
                   'generated_tokens':len(response.outputs[0].token_ids),
                   'seconds':time.monotonic()-start,'pass':bool(expected(text.lower()))}
            result['checks'].append(check)
            print(json.dumps(check),flush=True)
            run.log({f'{name}/pass':int(check['pass']),f'{name}/seconds':check['seconds'],
                     f'{name}/generated_tokens':check['generated_tokens']})
            assert check['pass'], f'{name} failed: {text!r}'
            return text
        generate('text',[{'role':'user','content':'What is 2 + 2? Reply with only the number.'}],[],lambda s:bool(re.search(r'\b4\b',s)))
        red=Image.new('RGB',(256,256),(255,0,0)); blue=Image.new('RGB',(256,256),(0,0,255))
        messages=[{'role':'user','content':[{'type':'image'},{'type':'image'},
                  {'type':'text','text':'Name the solid color of the first image and then the second image.'}]}]
        generate('two_images',messages,[red,blue],lambda s:'red' in s and 'blue' in s and s.index('red')<s.index('blue'))
        messages=[{'role':'system','content':'You inspect images returned by tools. Answer the latest user question.'},
                  {'role':'user','content':[{'type':'image'},{'type':'text','text':'Inspect the next view using echo, then tell me the color of the returned image.'}]},
                  {'role':'assistant','content':'<tool_call>\n{"name":"echo","arguments":{"op":"select_view","view_name":"A4C"}}\n</tool_call>'},
                  {'role':'user','content':[{'type':'text','text':'<tool_response>Selected A4C view:'},{'type':'image'},
                     {'type':'text','text':'</tool_response>\nWhat solid color is the newly returned image?'}]}]
        generate('tool_response_images',messages,[red,blue],lambda s:'blue' in s)
        result['pass']=True
        print('QWEN_VLLM_INFERENCE_PASS',flush=True)
    except Exception as exc:
        result['error']=repr(exc)
        traceback.print_exc()
    finally:
        (out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if run:
            run.summary.update({k:v for k,v in result.items() if k not in ('checks','engine_args')})
            run.summary['checks_passed']=sum(c['pass'] for c in result['checks'])
            run.finish(exit_code=0 if result['pass'] else 1)
    return 0 if result['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
