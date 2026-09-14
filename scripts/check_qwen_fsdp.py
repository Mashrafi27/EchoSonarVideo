"""Two-GPU regression gate for Qwen vision under FSDP2 CPU offload."""
import copy
import os
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
import verl_bridge.fsdp_compat  # Select the same wrapping policy as the actual trainer.
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict


class VisionProbe(nn.Module):
    _no_split_modules = ['Qwen3VLVisionBlock']

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='qwen3_vl', tie_word_embeddings=False)
        config = Qwen3VLVisionConfig(depth=2, hidden_size=64, intermediate_size=128,
            out_hidden_size=128, num_heads=4, patch_size=16, temporal_patch_size=2,
            spatial_merge_size=2, num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1])
        config._attn_implementation = 'sdpa'
        self.model = nn.Module()
        self.model.visual = Qwen3VLVisionModel(config)

    def forward(self, pixels, grid):
        output, deepstack = self.model.visual(pixels, grid_thw=grid)
        return output + sum(deepstack)


rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
mesh = init_device_mesh('cuda', (dist.get_world_size(),))
torch.manual_seed(20260914)
base = VisionProbe().cuda()
sharded = copy.deepcopy(base).cpu()
pixels = torch.randn(16, 3 * 2 * 16 * 16, device='cuda', dtype=torch.bfloat16)
grid = torch.tensor([[1, 4, 4]], device='cuda')
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    expected = base(pixels, grid).float()
del base
full_state = sharded.state_dict()
offload = CPUOffloadPolicy(pin_memory=True)
apply_fsdp2(sharded, dict(mesh=mesh,
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    offload_policy=offload, reshard_after_forward=True), {})
fsdp2_load_full_state_dict(sharded, full_state, mesh, offload)
del full_state
optimizer = torch.optim.AdamW(sharded.parameters(), lr=1e-3)
for step in (1, 2):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = sharded(pixels, grid).float()
    if step == 1:
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
    loss = result.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    position = sharded.model.visual.pos_embed.weight
    grad = position.grad.to_local()
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    before = position.to_local().detach().clone()
    optimizer.step()
    assert not torch.equal(before, position.to_local()), 'Optimizer did not change vision position weights'
    if rank == 0:
        print(f'QWEN_FSDP_OFFLOAD_STEP_PASS step={step} loss={loss.item():.6f}', flush=True)
dist.barrier()
dist.destroy_process_group()
