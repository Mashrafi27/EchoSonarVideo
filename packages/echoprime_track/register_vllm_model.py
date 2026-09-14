"""Registers `EchoPrimeQwen3ForCausalLMVLLM` with vLLM's model registry under the SAME
architecture name HF's `save_pretrained` already writes to our checkpoint's `config.json`
(`"architectures": ["EchoPrimeQwen3ForCausalLM"]`) -- no config.json changes needed, the class
name and the registered arch name just need to match exactly.

Must run inside whatever process actually builds the vLLM engine, before that happens --
mirrors `verl/experimental/vla/fsdp_workers.py:248-250`'s
`from verl.experimental.vla.models import register_vla_models; register_vla_models()` pattern.
"""
from vllm import ModelRegistry

from echoprime_track.vllm_model import EchoPrimeQwen3ForCausalLMVLLM

ARCH_NAME = "EchoPrimeQwen3ForCausalLM"


def register_echoprime_vllm_model() -> None:
    ModelRegistry.register_model(ARCH_NAME, EchoPrimeQwen3ForCausalLMVLLM)
