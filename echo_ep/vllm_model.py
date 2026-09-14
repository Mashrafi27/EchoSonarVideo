"""vLLM-native model class for the frozen-EchoPrime + Qwen3-8B-text GRPO rollout.

NOT the same class as `echo_ep/modeling.py`'s `EchoPrimeQwen3ForCausalLM` -- vLLM models have a
completely different forward/KV-cache contract (paged attention, `positions` instead of an
`attention_mask`, weights loaded via `AutoWeightsLoader` instead of `state_dict`). That HF class
stays exactly as-is for the FSDP actor's training-time forward pass; this one exists only to let
verl's real vLLM rollout serve the same architecture for generation.

Reuses vLLM's own "image" modality end-to-end (not a custom modality key) because it already has
a real precomputed-embedding path for exactly our shape of input -- see
`vllm/model_executor/models/llava.py`'s `LlavaImageEmbeddingInputs`/`image_embeds` handling: a
caller passing a raw tensor under `multi_modal_data={"image": tensor}` gets treated as
`ImageEmbeddingItems` (`vllm/multimodal/parse.py`) automatically, with NO custom
`MultiModalDataParser` subclass needed. Each EchoPrime view is one "image" item producing exactly
one soft token (`echo_ep.modeling.VIEW_TOKEN`) -- unlike llava (many tokens per image, from a
patch grid), our per-item token count is always 1, which is the simplest possible case for vLLM's
placeholder-replacement and embedding-merge machinery.

Unlike llava's `image_embeds` path (which assumes the caller's tensor is ALREADY in the LM's
hidden size, bypassing llava's own projector too), our `image_embeds` input is EchoPrime's raw
512-dim feature -- our projector still needs to run inside `embed_multimodal`, matching llava's
`pixel_values` code path's shape (project, then hand back to be spliced in) rather than its
`image_embeds` code path's shape (already-projected, spliced in verbatim).
"""
from collections.abc import Mapping, Sequence
from typing import Iterable

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from echo_ep.modeling import VIEW_TOKEN

VIDEO_EMBED_DIM = 512


class _EchoPrimeProcessor:
    """A minimal stand-in for a real HF `ProcessorMixin`, which we don't have (Qwen3-8B's own
    "processor" is just a plain tokenizer, not a combined text+vision `ProcessorMixin` -- vLLM's
    default processor resolution (`BaseProcessingInfo.get_hf_processor` ->
    `self.ctx.get_hf_processor`) does a strict `isinstance(..., ProcessorMixin)` check that a
    bare tokenizer fails). `EchoPrimeProcessingInfo.get_hf_processor` returns this directly,
    bypassing that AutoProcessor-based resolution entirely -- vLLM's own calling code
    (`_call_hf_processor`) only ever calls it as a plain `processor(text=..., **mm_data, ...)`
    callable, so it doesn't need to actually inherit from `ProcessorMixin`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str, image=None, **kwargs) -> BatchFeature:
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if image is not None:
            data["image_embeds"] = image
        return BatchFeature(data=data)


class EchoPrimeProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_hf_processor(self, **kwargs: object) -> _EchoPrimeProcessor:
        return _EchoPrimeProcessor(self.get_tokenizer())

    def _get_expected_hidden_size(self) -> int | None:
        # Default (with enable_mm_embeds=True) validates that `image_embeds` arrives already
        # projected to the LM's hidden size -- true for llava's OWN `image_embeds` path (which
        # bypasses its projector entirely), but not for us: our `image_embeds` is EchoPrime's
        # raw 512-dim feature, and `embed_multimodal` runs OUR projector on it inside the model
        # (so the projector stays a normal trainable/loadable/syncable model weight, not
        # something computed outside it). Skip this validation -- we control the shape.
        return None


class EchoPrimeDummyInputsBuilder(BaseDummyInputsBuilder["EchoPrimeProcessingInfo"]):
    """Builds synthetic inputs vLLM uses once at startup to profile peak memory (sizing the KV
    cache) -- never seen by real requests. No real HF processor needed for either piece here:
    our "text" is just N view-token placeholders, our "image data" is N zero-vectors of the
    right dim, matching what `EchoPrimeMultiModalProcessor` below expects to receive."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return VIEW_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options
    ) -> MultiModalDataDict:
        num_views = mm_counts.get("image", 0)
        # (num_items, feature_size, hidden) is the shape `is_embeddings()`
        # (vllm/multimodal/parse.py) requires for a plain tensor to be treated as
        # ImageEmbeddingItems rather than ImageProcessorItems (which needs a real pixel
        # decoder) -- feature_size is 1 since each view is exactly one soft token.
        return {"image": torch.zeros(num_views, 1, VIDEO_EMBED_DIM)}


class EchoPrimeMultiModalProcessor(BaseMultiModalProcessor[EchoPrimeProcessingInfo]):
    """Deliberately much smaller than llava's own processor: llava needs `_get_prompt_updates` to
    compute a resolution-dependent token count per image (a real HF vision encoder's patch grid
    varies with image size), which needs a real HF processor/config round-trip. We don't -- one
    view is always exactly one token, and the caller (our dataset's `process_vision_info`
    override) already builds the prompt with the correct number of `VIEW_TOKEN` placeholders
    before this ever runs. This is a straight identity replacement, just here to satisfy vLLM's
    (still-required, even for pre-tokenized `TokensPrompt` requests) processing pipeline, which
    validates placeholder counts against actual per-item feature sizes."""

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(image_embeds=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        view_token_id = self.info.get_hf_config().view_token_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
            return [view_token_id] * images.get_feature_size(item_idx)

        return [
            PromptReplacement(modality="image", target=[view_token_id],
                               replacement=get_replacement),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    EchoPrimeMultiModalProcessor,
    info=EchoPrimeProcessingInfo,
    dummy_inputs=EchoPrimeDummyInputsBuilder,
)
class EchoPrimeQwen3ForCausalLMVLLM(nn.Module, SupportsMultiModal, SupportsLoRA):
    # Our checkpoint's HF-side weight names (`echo_ep/modeling.py`'s `self.lm` is a real
    # `transformers.Qwen3ForCausalLM`, saved with the `lm.` prefix from being a submodule of
    # OUR wrapper) don't match vLLM's own naming for its `Qwen3ForCausalLM` (`language_model.
    # model.*` / `language_model.lm_head.*`) -- `AutoWeightsLoader` handles this declaratively
    # via a prefix rename, matching llava's own `hf_to_vllm_mapper` pattern exactly.
    # `projector.*` needs no rename -- same attribute name on both sides.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm.model.": "language_model.model.",
            "lm.lm_head.": "language_model.lm_head.",
        }
    )

    # GRPO trains LoRA adapters on the frozen base (paper spec) -- verl syncs them into this
    # vLLM instance for rollout generation, which requires `SupportsLoRA`. We delegate attention
    # /MLP linear layers entirely to `self.language_model` (a real, unmodified vLLM
    # `Qwen3ForCausalLM`), so its own LoRA-relevant class attributes describe our model's
    # actual linear-layer structure correctly too -- mirrored here rather than left at
    # `SupportsLoRA`'s empty defaults, which would mishandle Qwen3's fused qkv/gate_up
    # projections.
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping
    embedding_modules = Qwen3ForCausalLM.embedding_modules

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.language_model = Qwen3ForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        hidden = config.text_config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(VIDEO_EMBED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return VIEW_TOKEN
        return None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_embeds = kwargs.pop("image_embeds", None)
        if image_embeds is None:
            return []
        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        # Raw EchoPrime features in (VIDEO_EMBED_DIM), not yet in the LM's hidden size --
        # unlike llava's OWN `image_embeds` path (which assumes the caller already projected),
        # ours still needs the projector, matching llava's `pixel_values` path's shape instead.
        # Shape in is (total_items, 1, VIDEO_EMBED_DIM) -- feature_size=1 per view (see
        # EchoPrimeDummyInputsBuilder) -- projector acts on the last dim regardless of leading
        # dims. Do NOT squeeze the feature_size=1 dim: vLLM expects `MultiModalEmbeddings` as a
        # sequence of 2D (feature_size, hidden) tensors, one per item -- squeezing collapses
        # each item's embedding to 1D, which fails validation ("Expected multimodal embeddings
        # to be a sequence of 2D tensors, but got tensors with shapes [torch.Size([hidden])]").
        dtype = self.projector[0].weight.dtype
        return self.projector(image_embeds.to(dtype))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
