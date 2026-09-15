"""vLLM-native model class for the frozen-EchoPrime + Qwen3-8B-text GRPO rollout.

NOT the same class as `echoprime_track/modeling.py`'s `EchoPrimeQwen3ForCausalLM` -- vLLM models have a
completely different forward/KV-cache contract (paged attention, `positions` instead of an
`attention_mask`, weights loaded via `AutoWeightsLoader` instead of `state_dict`). That HF class
stays exactly as-is for the FSDP actor's training-time forward pass; this one exists only to let
verl's real vLLM rollout serve the same architecture for generation.

Registers TWO custom modalities, "clip" (768-dim) and "detr" (256-dim), one per EchoPrime
detection head -- matching Task 2's dual-projector split on the HF side (`modeling.py`'s
`clip_projector`/`detr_projector`). UNLIKE the old single-modality version of this file, this
does NOT get "clip"/"detr" recognized for free: that old version's "treat a raw tensor as
`ImageEmbeddingItems`" trick only worked because it reused the literal, hardcoded key "image"
(one of exactly four modality names `vllm.multimodal.parse.MultiModalDataParser` recognizes
out of the box). "clip"/"detr" are genuinely custom names, and a raw tensor under either key is
REJECTED by vLLM's default parser (`ValueError: Unsupported modality: clip`) unless something
else is registered to handle them -- that's what `_EchoPrimeDataParser` below does, wired in via
`EchoPrimeProcessingInfo.get_data_parser()`. See that class's own docstring, and the Step 1/
Step 4 investigation notes further down, for what was actually confirmed against the real
installed vLLM 0.17.0 source (including a second, more subtle bug this uncovered) before this
shape was settled on -- don't take this summary paragraph's word for the mechanism, the sections
below cite real source and job numbers. Each EchoPrime view produces exactly one soft token per
modality present (`echoprime_track.modeling.CLIP_TOKEN`/`DETR_TOKEN`) -- unlike llava (many
tokens per image, from a patch grid), our per-item token count is always 1.

Unlike llava's `image_embeds` path (which assumes the caller's tensor is ALREADY in the LM's
hidden size, bypassing llava's own projector too), our `clip_embeds`/`detr_embeds` inputs are
EchoPrime's raw 768-dim / 256-dim features -- our projectors still need to run inside
`embed_multimodal`, matching llava's `pixel_values` code path's shape (project, then hand back
to be spliced in) rather than its `image_embeds` code path's shape (already-projected, spliced
in verbatim).

Step 1 investigation (this task, on the AMD MI210 box, `.tmp_work/rocm_validation_20260914/env`,
vLLM 0.17.0+rocm700 -- jobs 190870/190872, real source read directly from the installed
`vllm` package):

- `embed_multimodal(self, **kwargs) -> MultiModalEmbeddings` IS the real, current hook name on
  `SupportsMultiModal` in this vLLM version (the old single-modality code's name was already
  correct -- no rename needed, contrary to the brief's flagged possibility).
- The merge step is `vllm.model_executor.models.utils._merge_multimodal_embeddings` (now
  PRIVATE -- renamed from the old public `merge_multimodal_embeddings`). Irrelevant to this
  file either way: neither the old nor the new code calls it directly. It's invoked internally
  by `SupportsMultiModal.embed_input_ids` (the base class's own orchestration method, not
  `get_input_embeddings` -- that name doesn't exist on this interface in 0.17.0), which the
  model runner calls after collecting this class's `embed_multimodal` output. We never need to
  touch it.
- Confirmed from the real caller, `vllm/v1/worker/gpu_model_runner.py:2478-2497`, and
  `vllm/multimodal/utils.py`'s `group_mm_kwargs_by_modality` (its own docstring: "To simplify
  the implementation of `embed_multimodal`, we add another restriction that the items in a
  batch must belong to the same modality"): `embed_multimodal` is called ONCE PER MODALITY
  GROUP, never once with every registered modality's kwargs merged together. So on any real
  call here, `kwargs` contains at most ONE of `clip_embeds`/`detr_embeds`, never both at once.
  The brief's Step 2 draft assumed both could arrive in the same call and concatenated
  defensively -- that defensive shape is harmless (matches the real dual-modality precedent,
  `llava_onevision.py`'s own `embed_multimodal`, which loops over whichever modality keys are
  actually present) and needs no structural change, just this corrected description: in
  practice exactly one of the two `if` branches below ever fires per call, never both.

Step 4 manual verification against a real checkpoint (this task, jobs 190932/190934/190935 on
the same AMD box) surfaced a SECOND, structural gap the brief's Step 2 draft didn't anticipate
and this file's old single-modality docstring got wrong: the "any raw tensor under a
`multi_modal_data` key is auto-parsed as `ImageEmbeddingItems`" trick does NOT work for
arbitrary custom modality names. `vllm.multimodal.parse.MultiModalDataParser._get_subparsers`
hardcodes exactly four recognized keys -- "audio", "image", "video", "vision_chunk" -- and
`parse_mm_data` raises `ValueError: Unsupported modality: clip` for anything else (confirmed
directly, job 190935's stderr). The OLD single-modality code happened to work only because it
literally reused the name "image", one of those four hardcoded keys -- not because vLLM
generically accepts any custom modality name. "clip"/"detr" need a real subparser mapping, which
is what `_EchoPrimeDataParser` below adds. (The first fix attempted here delegated to the same
built-in `_parse_image_data` the "image" key already uses -- that turned out to be WRONG too,
a second bug documented in `_EchoPrimeDataParser`'s own docstring below; don't copy that
approach.) The current (0.17.0) extension point for this is `BaseProcessingInfo.get_data_parser()` --
`BaseMultiModalProcessor._get_data_parser` (what an older version of this pattern might use) was
removed and now raises `ValueError` on class construction, pointing at this replacement (see
`vllm/multimodal/processing/processor.py`'s `BaseMultiModalProcessor.__init__`, "TODO: Remove in
v0.18" / "has been moved to `BaseProcessingInfo.build_data_parser` in v0.16").

Separately, this environment's `aiter` package (installed but with no HIP/C++ compiler
toolchain -- see `docs/ROCM_VALIDATION.md`) crashes on the mere `import aiter` regardless of
`VLLM_ROCM_USE_AITER=0` (that flag gates aiter *usage*, not vLLM's own unconditional import of
`vllm._aiter_ops` at ROCm platform init, which chains into `import aiter` and trips aiter's
module-level JIT enum build). Worked around in the Step 4 smoke script only
(`sys.modules["aiter"] = None` before importing vllm, the standard way to make
`importlib.util.find_spec("aiter")` report "not found"), not in this file -- irrelevant to the
class registration itself.
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
from vllm.multimodal.parse import (
    EmbeddingItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from echoprime_track.modeling import CLIP_TOKEN, DETR_TOKEN

CLIP_EMBED_DIM = 768
DETR_EMBED_DIM = 256


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

    def __call__(self, text: str, clip=None, detr=None, **kwargs) -> BatchFeature:
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if clip is not None:
            data["clip_embeds"] = clip
        if detr is not None:
            data["detr_embeds"] = detr
        return BatchFeature(data=data)


class _EchoPrimeDataParser(MultiModalDataParser):
    """`MultiModalDataParser._get_subparsers` hardcodes exactly four recognized modality keys
    ("audio", "image", "video", "vision_chunk") -- a raw tensor under any OTHER key (e.g. our
    "clip"/"detr") raises `ValueError: Unsupported modality: <k>` in the base parser (confirmed
    directly against the real vLLM 0.17.0 install this task, job 190935). The old
    single-modality version of this file worked only because it reused the literal name
    "image", one of those four -- not because any custom key is auto-supported. Per
    `BaseProcessingInfo.get_data_parser`'s own docstring ("You can support additional modalities
    by creating a subclass of `MultiModalDataParser` that has additional subparsers"), this adds
    "clip"/"detr" subparsers.

    Deliberately NOT routed through the built-in `_parse_image_data` (which is what the old
    single-modality file's docstring assumed would generalize): `_parse_image_data`'s embeddings
    branch builds an `ImageEmbeddingItems`, whose `__init__` HARDCODES `modality="image"` in its
    call to `EmbeddingItems.__init__` (`vllm/multimodal/parse.py`'s `ImageEmbeddingItems`) --
    regardless of which dict key the tensor arrived under. Confirmed directly this task (job
    190936): routing "clip" data through `_parse_image_data` silently tags the resulting items
    as modality "image", so `EmbeddingItems.get_passthrough_data()` (`{modality}_embeds`) emits
    `image_embeds` instead of `clip_embeds`, and downstream validation fails with "Expected
    there to be 1 clip items ... but only found 0". Fix: construct the modality-parametrized
    base class, `EmbeddingItems(data, modality, expected_hidden_size)`, directly -- our
    clip/detr inputs are always precomputed embeddings (never raw pixel data needing the real
    image decoder path `_parse_image_data` also handles), so we don't need that branch anyway.
    """

    def _parse_embedding_data(
        self, modality: str, data: object
    ) -> EmbeddingItems | None:
        if data is None:
            return None
        return EmbeddingItems(data, modality, self.expected_hidden_size)

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        parsers = dict(super()._get_subparsers())
        parsers["clip"] = lambda data: self._parse_embedding_data("clip", data)
        parsers["detr"] = lambda data: self._parse_embedding_data("detr", data)
        return parsers


class EchoPrimeProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"clip": None, "detr": None}

    def get_hf_processor(self, **kwargs: object) -> _EchoPrimeProcessor:
        return _EchoPrimeProcessor(self.get_tokenizer())

    def get_data_parser(self) -> MultiModalDataParser:
        # Current (0.17.0) extension point for custom modality keys -- NOT
        # `BaseMultiModalProcessor._get_data_parser`, which was removed and now raises
        # `ValueError` on class construction pointing back at this method (see
        # `vllm/multimodal/processing/processor.py`'s `BaseMultiModalProcessor.__init__`).
        return _EchoPrimeDataParser(expected_hidden_size=self._get_expected_hidden_size())

    def _get_expected_hidden_size(self) -> int | None:
        # Default (with enable_mm_embeds=True) validates that `*_embeds` arrives already
        # projected to the LM's hidden size -- true for llava's OWN `image_embeds` path (which
        # bypasses its projector entirely), but not for us: our `clip_embeds`/`detr_embeds` are
        # EchoPrime's raw 768-dim/256-dim features, and `embed_multimodal` runs OUR projectors
        # on them inside the model (so the projectors stay normal trainable/loadable/syncable
        # model weights, not something computed outside it). Skip this validation -- we control
        # the shape.
        return None


class EchoPrimeDummyInputsBuilder(BaseDummyInputsBuilder["EchoPrimeProcessingInfo"]):
    """Builds synthetic inputs vLLM uses once at startup to profile peak memory (sizing the KV
    cache) -- never seen by real requests. No real HF processor needed for either piece here:
    our "text" is just N clip-token and M detr-token placeholders, our "data" is zero-vectors of
    the right dims, matching what `EchoPrimeMultiModalProcessor` below expects to receive."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return CLIP_TOKEN * mm_counts.get("clip", 0) + DETR_TOKEN * mm_counts.get("detr", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options
    ) -> MultiModalDataDict:
        n_clip = mm_counts.get("clip", 0)
        n_detr = mm_counts.get("detr", 0)
        # (num_items, feature_size, hidden) is the shape `is_embeddings()`
        # (vllm/multimodal/parse.py) requires for a plain tensor to be treated as
        # ImageEmbeddingItems rather than ImageProcessorItems (which needs a real pixel
        # decoder) -- feature_size is 1 since each view is exactly one soft token per modality.
        return {
            "clip": torch.zeros(n_clip, 1, CLIP_EMBED_DIM),
            "detr": torch.zeros(n_detr, 1, DETR_EMBED_DIM),
        }


class EchoPrimeMultiModalProcessor(BaseMultiModalProcessor[EchoPrimeProcessingInfo]):
    """Deliberately much smaller than llava's own processor: llava needs `_get_prompt_updates` to
    compute a resolution-dependent token count per image (a real HF vision encoder's patch grid
    varies with image size), which needs a real HF processor/config round-trip. We don't -- one
    view is always exactly one token per modality, and the caller (our dataset's
    `process_vision_info` override) already builds the prompt with the correct number of
    `CLIP_TOKEN`/`DETR_TOKEN` placeholders before this ever runs. This is a straight identity
    replacement, just here to satisfy vLLM's (still-required, even for pre-tokenized
    `TokensPrompt` requests) processing pipeline, which validates placeholder counts against
    actual per-item feature sizes."""

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            clip_embeds=MultiModalFieldConfig.batched("clip"),
            detr_embeds=MultiModalFieldConfig.batched("detr"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        clip_token_id = hf_config.clip_token_id
        detr_token_id = hf_config.detr_token_id

        def get_clip_replacement(item_idx: int):
            items = mm_items.get_items("clip", EmbeddingItems)
            return [clip_token_id] * items.get_feature_size(item_idx)

        def get_detr_replacement(item_idx: int):
            items = mm_items.get_items("detr", EmbeddingItems)
            return [detr_token_id] * items.get_feature_size(item_idx)

        return [
            PromptReplacement(modality="clip", target=[clip_token_id],
                               replacement=get_clip_replacement),
            PromptReplacement(modality="detr", target=[detr_token_id],
                               replacement=get_detr_replacement),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    EchoPrimeMultiModalProcessor,
    info=EchoPrimeProcessingInfo,
    dummy_inputs=EchoPrimeDummyInputsBuilder,
)
class EchoPrimeQwen3ForCausalLMVLLM(nn.Module, SupportsMultiModal, SupportsLoRA):
    # Our checkpoint's HF-side weight names (`echoprime_track/modeling.py`'s `self.lm` is a real
    # `transformers.Qwen3ForCausalLM`, saved with the `lm.` prefix from being a submodule of
    # OUR wrapper) don't match vLLM's own naming for its `Qwen3ForCausalLM` (`language_model.
    # model.*` / `language_model.lm_head.*`) -- `AutoWeightsLoader` handles this declaratively
    # via a prefix rename, matching llava's own `hf_to_vllm_mapper` pattern exactly.
    # `clip_projector.*`/`detr_projector.*` need no rename -- same attribute names on both sides
    # (matching `modeling.py`'s `EchoPrimeQwen3ForCausalLM.clip_projector`/`.detr_projector`
    # exactly), so `AutoWeightsLoader`'s default name-based matching handles them with no
    # additional `orig_to_new_prefix` entries.
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
        # LayerNorm, Linear, GELU, Linear -- matching `modeling.py`'s HF-side
        # `clip_projector`/`detr_projector` layer-for-layer (an earlier version of this plan
        # said 2 layers, based on truncated grep output; corrected during Task 2's review
        # against the real trained checkpoint, `report_generation/sft_thinking/model.py:67-79`).
        # This is a SEPARATE set of weights from the HF-side projectors -- synced from the FSDP
        # actor at training time via verl's normal weight-sync mechanism, which matches by
        # parameter name, hence these attribute names must match `modeling.py`'s exactly.
        self.clip_projector = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM), nn.Linear(CLIP_EMBED_DIM, hidden),
            nn.GELU(), nn.Linear(hidden, hidden))
        self.detr_projector = nn.Sequential(
            nn.LayerNorm(DETR_EMBED_DIM), nn.Linear(DETR_EMBED_DIM, hidden),
            nn.GELU(), nn.Linear(hidden, hidden))

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "clip":
            return CLIP_TOKEN
        if modality == "detr":
            return DETR_TOKEN
        return None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Confirmed on the AMD box against the real installed vLLM 0.17.0 source (see module
        # docstring): vLLM's `group_mm_kwargs_by_modality` guarantees every real call to this
        # method carries kwargs for exactly ONE modality, so in practice at most one of the two
        # branches below ever fires per call -- never both in the same call. Handling both
        # defensively (rather than asserting exactly one) matches the real dual-modality
        # precedent shipped in this vLLM version, `llava_onevision.py`'s own `embed_multimodal`,
        # which loops over whichever modality keys are actually present in its parsed kwargs.
        results = []
        clip_embeds = kwargs.pop("clip_embeds", None)
        if clip_embeds is not None:
            if isinstance(clip_embeds, list):
                clip_embeds = torch.cat(clip_embeds, dim=0)
            # Shape in is (total_items, 1, CLIP_EMBED_DIM) -- feature_size=1 per view (see
            # EchoPrimeDummyInputsBuilder) -- projector acts on the last dim regardless of
            # leading dims. Do NOT squeeze the feature_size=1 dim: vLLM expects
            # `MultiModalEmbeddings` as a sequence of 2D (feature_size, hidden) tensors, one per
            # item -- squeezing collapses each item's embedding to 1D, which fails validation.
            dtype = self.clip_projector[1].weight.dtype  # index [1]: the first Linear, after
            # the LayerNorm at index [0] -- unlike the old single-modality projector (plain
            # nn.Sequential(Linear, GELU, Linear), dtype read off index [0]).
            results.append(self.clip_projector(clip_embeds.to(dtype)))
        detr_embeds = kwargs.pop("detr_embeds", None)
        if detr_embeds is not None:
            if isinstance(detr_embeds, list):
                detr_embeds = torch.cat(detr_embeds, dim=0)
            dtype = self.detr_projector[1].weight.dtype
            results.append(self.detr_projector(detr_embeds.to(dtype)))
        if not results:
            return []
        return torch.cat(results, dim=0) if len(results) > 1 else results[0]

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
