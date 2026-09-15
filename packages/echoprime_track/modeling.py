"""The trainable half of the cold-start architecture: a projector + Qwen3-8B (text-only).

Deliberately NOT bundling the frozen EchoPrime encoder into this class -- the encoder only
ever runs offline, once per study, via `echoprime_track.build_video_cache` (see that module's
docstring for why: it's frozen, so its output for a study is identical regardless of which
QA pair is training, running it once and caching is both correct and ~25x cheaper than once
per QA pair). This model's `forward()` takes PRECOMPUTED per-view (N, 512) embeddings as
input, not raw video -- at inference/eval time later, the same cache-then-feed flow applies:
run `echoprime_track.encoder` once per study, then this model consumes the result.

LLaVA-style embedding splice: the projector maps each view's 512-dim EchoPrime embedding to
one soft token in Qwen3-8B's embedding space (one token per view is the natural choice here --
`encode_study` already returns one global-pooled vector per view, not a patch grid, so there's
no finer spatial structure a multi-token-per-view resampler would actually be resampling).
Those soft tokens replace a placeholder token's embedding at the positions the collator put
placeholders in the tokenized sequence, in view order.

Packaged as a real `PreTrainedModel`/`PretrainedConfig` pair (not a bare nn.Module) so
`save_pretrained`/`from_pretrained` work normally and the checkpoint is later loadable via
`AutoModel.from_pretrained(path, trust_remote_code=True)` -- the `auto_map` entries in
`get_auto_map()` are what make that work once this is saved with a config.json that includes
them (see `save_as_custom_model` below).
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin

VIEW_TOKEN = "<|view_embed|>"  # legacy -- only echoprime_track/dataset.py's SFT collator
                                # (no active launcher references it, see plan's Global
                                # Constraints) still imports this; unused by the splice below.
CLIP_TOKEN = "<|clip_embed|>"
DETR_TOKEN = "<|detr_embed|>"


class EchoPrimeQwen3Config(PretrainedConfig):
    model_type = "echoprime_qwen3"

    def __init__(self, text_model_name: str = "Qwen/Qwen3-8B",
                 clip_embed_dim: int = 768, detr_embed_dim: int = 256,
                 clip_token: str = CLIP_TOKEN, detr_token: str = DETR_TOKEN,
                 clip_token_id: int = None, detr_token_id: int = None,
                 text_config=None, **kwargs):
        self.text_model_name = text_model_name
        self.clip_embed_dim = clip_embed_dim
        self.detr_embed_dim = detr_embed_dim
        self.clip_token = clip_token
        self.detr_token = detr_token
        self.clip_token_id = clip_token_id
        self.detr_token_id = detr_token_id
        # `PretrainedConfig.get_text_config(decoder=True)` -- which HF's own generation code
        # (`_prepare_cache_for_generation`) calls to size the KV cache -- auto-detects a
        # `text_config` attribute on ANY config and returns it in place of `self`; this is the
        # built-in hook for composite/wrapper configs like this one, whose OWN attributes
        # (video_embed_dim etc.) aren't a real decoder config and don't have `num_hidden_layers`
        # etc. Without this, `.generate()` crashes trying to read Qwen3's fields off of us.
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        if isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            # Deliberately NOT `AutoConfig.from_pretrained(text_model_name)` here (a network/
            # disk fetch) -- transformers' own internals sometimes construct a throwaway
            # `self.__class__()` instance of ANY config class purely for `to_diff_dict()`/
            # `__repr__` bookkeeping (hit in practice: crashed under HF_HUB_OFFLINE=1 with no
            # real request involved at all). This branch is always a placeholder anyway: on the
            # real construction path `from_cold_start` immediately overwrites it via
            # `model.config.text_config = model.lm.config`, and on a real checkpoint reload
            # `text_config` arrives as a dict (the `isinstance` branch above), never hitting
            # this fallback. An in-memory default costs nothing and needs no network/disk.
            text_config = Qwen3Config()
        self.text_config = text_config
        # `PretrainedConfig.__init__` unconditionally sets `self.eos_token_id = kwargs.pop(...,
        # None)` as a REAL top-level attribute -- this happens here, in super().__init__, so it
        # permanently shadows __getattr__'s delegation to text_config for this one field (
        # __getattr__ only fires when normal lookup fails, and a real `None` attribute makes
        # normal lookup "succeed"). Left alone, this saves as a MISSING top-level eos_token_id in
        # config.json (None values get pruned), which vLLM reads directly as raw JSON -- it never
        # walks into our nested text_config the way our own __getattr__ does, so it falls back to
        # no/wrong stop token and generation runs to the hard max_tokens ceiling every single
        # time. Confirmed this session: 100% response-length clip ratio, zero variance, at both
        # temperature 1.0 and 0.8, while an in-process HF .generate() call (whose live Python
        # object still had __getattr__ delegation working) stopped early on the same checkpoint.
        # Fix: pass the REAL decoder eos/pad/bos ids (and tie_word_embeddings, same shadowing
        # bug: PretrainedConfig.__init__ defaults it to True, silently wrong for Qwen3-8B which
        # is untied -- confirmed this session as the ACTUAL root cause of every garbage-output
        # rollout tonight, not eos_token_id/temperature/response-length as earlier suspected:
        # vLLM was projecting final hidden states through the tied embed_tokens matrix instead
        # of the real, separately-trained lm_head matrix, corrupting every single output token
        # regardless of sampling settings) through explicitly so they land in kwargs and get set
        # as genuine top-level attributes that actually serialize.
        for _id_field in ("eos_token_id", "pad_token_id", "bos_token_id", "tie_word_embeddings"):
            if kwargs.get(_id_field) is None:
                _text_value = getattr(text_config, _id_field, None)
                if _text_value is not None:
                    kwargs[_id_field] = _text_value
        super().__init__(**kwargs)

    def __getattr__(self, name):
        """Only called when normal attribute lookup fails. Delegates to the real Qwen3 config
        for any flat-decoder-config field (`vocab_size`, `hidden_size`, `num_hidden_layers`,
        ...) this wrapper config doesn't define itself. First needed for HF's own generation
        code (`get_text_config()`, handled explicitly above); vLLM's engine init hit a second,
        different case (`config.vocab_size` accessed directly, not through `get_text_config()`)
        -- rather than keep patching individual fields one caller at a time, delegate broadly.
        Guarded against recursion before `text_config` itself is set during __init__."""
        if name == "text_config" or "text_config" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["text_config"], name)


class EchoPrimeQwen3ForCausalLM(PreTrainedModel, GenerationMixin):
    """Frozen-EchoPrime-embeddings in, Qwen3-8B-text-with-spliced-soft-tokens out.

    `GenerationMixin` isn't mixed into `PreTrainedModel` by default in this transformers
    version -- without it, `.generate()` doesn't exist on this class at all."""

    config_class = EchoPrimeQwen3Config
    # These capability flags are checked on THIS class (not delegated to `self.lm`) by
    # transformers' `_check_and_adjust_attn_implementation` -- a bare custom PreTrainedModel
    # subclass defaults to unsupported for everything, so without declaring them explicitly,
    # requesting sdpa OR flash-attn both hard-error even though `self.lm` (a real
    # Qwen3ForCausalLM, which sets these same flags) handles attention correctly either way.
    # Mirrors `Qwen3PreTrainedModel`'s own flags exactly.
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_sdpa = True
    # Same reasoning as the attention flags above: `gradient_checkpointing_enable()` checks
    # this flag on THIS class before doing anything, but the actual propagation
    # (`self.apply(...)`) walks the WHOLE module tree regardless of nesting depth, so it
    # still correctly reaches `self.lm`'s real decoder layers once this is declared true.
    supports_gradient_checkpointing = True
    # verl's FSDP wrap-policy auto-detection reads `_no_split_modules` off the TOP-LEVEL model
    # (`verl/utils/fsdp_utils.py:get_fsdp_wrap_policy`: `getattr(module, "_no_split_modules",
    # None)`) to decide which submodule class to give its own FSDP unit. A bare custom
    # PreTrainedModel subclass doesn't have this, so without declaring it FSDP silently treats
    # the ENTIRE model as one unit instead of one per decoder layer -- confirmed by an OOM
    # trying to flatten/allocate ~30GB in one `torch.cat` (every parameter at once) instead of
    # a per-layer amount. Matches `Qwen3PreTrainedModel._no_split_modules` exactly, since the
    # real decoder layers live inside `self.lm`.
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: EchoPrimeQwen3Config):
        super().__init__(config)
        # Plain PreTrainedModel(config) construction is config-only (random init) per HF
        # convention -- real pretrained Qwen3-8B weights are loaded by `from_cold_start`
        # below, not here. This lets `AutoModel.from_pretrained(<our checkpoint>)` later
        # reconstruct the architecture correctly before loading OUR trained state_dict over
        # it (which does include the real LM weights, post-training).
        self.lm = AutoModelForCausalLM.from_config(config.text_config)
        hidden = self.lm.config.hidden_size
        # LayerNorm + Linear, matching report_generation/sft_thinking/model.py::EchoVLM's
        # clip_projector/detr_projector layer-for-layer -- save_sft_init_checkpoint.py (Task 3)
        # loads Darya's real trained weights into these, so the shapes/layer types must match
        # hers exactly or that state_dict load fails.
        self.clip_projector = nn.Sequential(
            nn.LayerNorm(config.clip_embed_dim),
            nn.Linear(config.clip_embed_dim, hidden),
        )
        self.detr_projector = nn.Sequential(
            nn.LayerNorm(config.detr_embed_dim),
            nn.Linear(config.detr_embed_dim, hidden),
        )

    @classmethod
    def from_cold_start(cls, config: EchoPrimeQwen3Config, dtype=torch.bfloat16):
        """The real entrypoint for starting SFT: config-only __init__ above, then swap in
        Qwen3-8B's actual pretrained weights (not a random init) for the LM half."""
        model = cls(config)
        model.lm = AutoModelForCausalLM.from_pretrained(config.text_model_name, dtype=dtype)
        # `model.lm.from_pretrained` builds its OWN fresh config object, orphaning the one
        # `config.text_config` was set to in __init__ (loaded once, before this real LM
        # existed). Re-point at the real one so they're the SAME object from here on: a later
        # `model.lm.resize_token_embeddings(...)` (e.g. after adding VIEW_TOKEN) mutates
        # `model.lm.config.vocab_size` in place, and without this line `config.text_config`
        # (what actually gets serialized by `save_pretrained`) would silently keep the stale
        # pre-resize vocab size, corrupting any checkpoint saved after a token was added.
        model.config.text_config = model.lm.config
        # __init__'s eos/pad/bos-id sync (see the comment there) only runs once, against
        # whatever text_config existed AT CONSTRUCTION TIME -- for this from_cold_start path
        # that's still the placeholder Qwen3Config() from __init__, not the real pretrained one
        # just loaded above. Re-sync now that model.config.text_config is the real thing, or the
        # top-level ids stay None/wrong in every checkpoint saved from this model.
        for _id_field in ("eos_token_id", "pad_token_id", "bos_token_id", "tie_word_embeddings"):
            _text_value = getattr(model.lm.config, _id_field, None)
            if _text_value is not None:
                setattr(model.config, _id_field, _text_value)
        # __init__'s plain nn.Sequential(nn.Linear(...)) defaults to float32 regardless of
        # what dtype the LM loads in -- .to(device) alone doesn't fix a dtype mismatch, only
        # a device one, so without this the projector's Linear weight stays fp32 while
        # inputs_embeds (from the now-bf16 LM's embedding table) is bf16 -> dtype crash at
        # the first forward call. Cast explicitly.
        model.clip_projector.to(dtype)
        model.detr_projector.to(dtype)
        return model

    @staticmethod
    def _splice_positions(inputs_embeds: torch.Tensor, input_ids: torch.Tensor,
                           token_id: int, projected: torch.Tensor, counts) -> torch.Tensor:
        """Returns `inputs_embeds` with each `token_id` position (in order, per example)
        replaced by the next row of `projected`. `projected` is (sum(counts), H) -- unpadded,
        concatenated across the batch in order, same convention the old
        `_splice_view_embeddings` used for a single modality, generalized to be called once
        per modality (clip, detr) instead of baking one modality in."""
        if torch.is_tensor(counts):
            counts = counts.flatten().tolist()
        offset = 0
        for b in range(input_ids.shape[0]):
            n = counts[b]
            if n == 0:
                continue
            positions = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
            assert len(positions) == n, (
                f"row {b}: {len(positions)} placeholders for token_id={token_id} but "
                f"{n} embeddings -- prompt construction and the embeddings passed into "
                f"forward() disagree on this example's count")
            inputs_embeds[b, positions] = projected[offset:offset + n]
            offset += n
        return inputs_embeds

    def _splice_vision_embeddings(self, input_ids: torch.Tensor,
                                   clip_embeddings, clip_counts,
                                   detr_embeddings, detr_counts) -> torch.Tensor:
        """inputs_embeds with clip- and detr-token positions replaced by their projected
        embeddings. Either modality may be absent (e.g. a study with no DETR detections at
        all) -- `clip_embeddings`/`detr_embeddings` None or empty just skips that splice."""
        # .clone() required, not optional: see the original _splice_view_embeddings'
        # docstring for why (in-place indexed assignment into a leaf-adjacent
        # requires-grad tensor errors during backward without it).
        inputs_embeds = self.lm.get_input_embeddings()(input_ids).clone()
        if clip_embeddings is not None and clip_embeddings.numel() > 0:
            clip_proj = self.clip_projector(clip_embeddings.to(inputs_embeds.dtype))
            inputs_embeds = self._splice_positions(
                inputs_embeds, input_ids, self.config.clip_token_id, clip_proj, clip_counts)
        if detr_embeddings is not None and detr_embeddings.numel() > 0:
            detr_proj = self.detr_projector(detr_embeddings.to(inputs_embeds.dtype))
            inputs_embeds = self._splice_positions(
                inputs_embeds, input_ids, self.config.detr_token_id, detr_proj, detr_counts)
        return inputs_embeds

    def forward(self, input_ids, attention_mask=None,
                clip_embeddings=None, clip_counts=None,
                detr_embeddings=None, detr_counts=None,
                labels=None, past_key_values=None, inputs_embeds=None, **kwargs):
        """`clip_embeddings is not None or detr_embeddings is not None` is the actual branch
        condition (mirrors the old `view_embeddings is not None` check) -- see the original
        docstring for why `inputs_embeds` is a named-but-discarded parameter."""
        if clip_embeddings is not None or detr_embeddings is not None:
            inputs_embeds = self._splice_vision_embeddings(
                input_ids, clip_embeddings, clip_counts, detr_embeddings, detr_counts)
            return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            labels=labels, past_key_values=past_key_values, **kwargs)
        return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        past_key_values=past_key_values, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                       clip_embeddings=None, clip_counts=None,
                                       detr_embeddings=None, detr_counts=None, **kwargs):
        model_inputs = self.lm.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs)
        if past_key_values is None:
            model_inputs["clip_embeddings"] = clip_embeddings
            model_inputs["clip_counts"] = clip_counts
            model_inputs["detr_embeddings"] = detr_embeddings
            model_inputs["detr_counts"] = detr_counts
        return model_inputs

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.lm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def get_auto_map(self) -> dict:
        """For a saved checkpoint's config.json, so a later `trust_remote_code=True` load
        (verl or plain transformers) can find this class -- see modeling docstring."""
        return {
            "AutoConfig": "modeling_echoprime_qwen3.EchoPrimeQwen3Config",
            "AutoModelForCausalLM": "modeling_echoprime_qwen3.EchoPrimeQwen3ForCausalLM",
        }
