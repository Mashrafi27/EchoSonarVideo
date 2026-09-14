"""The frozen EchoPrime video encoder (mViT-v2-S), loaded from local weights.

Ported from /home/mashrafimonon/Mashrafi/EchoPrime's load_for_finetuning.py -- confirmed
working under this project's conda env (`/data/ahmedaly/mashrafi_echogrpo/conda_env`,
torchvision 0.25.0, checkpoint loads cleanly, forward pass on a dummy (1,3,16,224,224)
tensor produces the expected (1, 512) output -- 512-dim, NOT the 768 the EchoSonar-R paper
states; trust the actual checkpoint over the paper's number).
"""
import os

import torch
import torchvision

# Default matches this box's local reference install; override via env var so this isn't
# hardcoded to one machine/user's path.
DEFAULT_WEIGHTS_DIR = "/home/mashrafimonon/Mashrafi/EchoPrime/model_data/weights"
EMBED_DIM = 512


def load_frozen_encoder(weights_dir: str = None, device=None) -> torch.nn.Module:
    """The mViT-v2-S video encoder, weights loaded, eval mode, every param requires_grad=False.

    Caller is responsible for keeping it frozen (no optimizer should ever see its params) --
    this function does not return an optimizer or param groups, just the frozen module.
    """
    weights_dir = weights_dir or os.environ.get("ECHOPRIME_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
    ckpt_path = os.path.join(weights_dir, "echo_prime_encoder.pt")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = torch.nn.Linear(encoder.head[-1].in_features, EMBED_DIM)
    encoder.load_state_dict(checkpoint)
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def embed_videos(encoder: torch.nn.Module, stack_of_videos: torch.Tensor,
                  bin_size: int = 8) -> torch.Tensor:
    """(N, 3, 16, 224, 224) -> (N, 512), binned to fit GPU memory (mirrors EchoPrime's own
    embed_videos, which bins at 50 for their hardware; 8 is more conservative for a shared box
    also running an LM -- tune via caller if needed)."""
    device = next(encoder.parameters()).device
    feats = []
    for start in range(0, stack_of_videos.shape[0], bin_size):
        chunk = stack_of_videos[start:start + bin_size].to(device)
        feats.append(encoder(chunk).cpu())
    return torch.cat(feats, dim=0)
