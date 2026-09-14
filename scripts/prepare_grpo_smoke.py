"""Select four real TRAIN examples for an infrastructure-only GRPO smoke run."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from eval.agentic_loop import SYSTEM_PROMPT

ap=argparse.ArgumentParser()
ap.add_argument('--source',default='build/rl_train.parquet')
ap.add_argument('--output',required=True)
args=ap.parse_args()
source=Path(args.source)
rows=pq.read_table(source).to_pylist()
selected=[]; counts={'yes':0,'no':0}; studies=set()
for row in rows:
    key=json.loads(row['reward_model']['ground_truth'])
    label=str(key.get('target','')).strip().lower()
    study=row['extra_info']['index']
    if key.get('kind')!='yesno' or label not in counts or counts[label]>=2 or study in studies:
        continue
    assert row.get('agent_name')=='tool_agent'
    row['agent_name']='echo_tool_agent'
    assert not row.get('videos'), 'Echo opening context must use images'
    for image in row['images']:
        assert Path(image['image']).is_file(), 'Missing source image'
        image['max_pixels']=160*160
    row['prompt']=[{'role':'system','content':SYSTEM_PROMPT}]+row['prompt']
    counts[label]+=1; studies.add(study); selected.append(row)
    if len(selected)==4: break
assert counts=={'yes':2,'no':2}, counts
# Exercise the real tool/session contract and the Qwen helper before model loading.
async def check_tools():
    from verl.tools.utils.tool_registry import initialize_tools_from_config
    from qwen_vl_utils import process_vision_info
    tool = initialize_tools_from_config('packages/verl_bridge/configs/echo_tool_config.yaml')[0]
    checks = []
    for row in selected:
        kwargs = row['extra_info']['tools_kwargs']['echo']
        instance_id, _ = await tool.create(**kwargs)
        try:
            views = tool._instances[instance_id].manifest.view_names()
            assert views, 'Study has no available views'
            response, reward, info = await tool.execute(instance_id, {'op':'select_view','view_name':views[0]})
            assert info.get('success') and response.image, 'Echo tool failed to return images'
            images, videos = process_vision_info([{'role':'user','content':[{'type':'image','image':im} for im in response.image]}],image_patch_size=16,return_video_metadata=True)
            assert len(images)==len(response.image) and videos is None
            checks.append({'available_views':len(views),'returned_images':len(images),'pass':True})
        finally:
            await tool.release(instance_id)
    return checks

tool_checks = asyncio.run(check_tools())
print('ECHO_TOOL_IMAGE_PREFLIGHT_PASS', json.dumps(tool_checks), flush=True)
output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
pq.write_table(pa.Table.from_pylist(selected),output)
manifest={'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'rows':4,'distinct_studies':4,'labels':counts,'image_counts':[len(r['images']) for r in selected],
          'tool_preflight':tool_checks,'image_max_pixels':25600,'system_prompt':SYSTEM_PROMPT,'purpose':'Infrastructure validation only; not a clinical accuracy experiment'}
output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='system_prompt'}),flush=True)
