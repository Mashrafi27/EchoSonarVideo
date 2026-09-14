"""One-off: save a `from_cold_start()` checkpoint (frozen EchoPrime + UNTUNED Qwen3-8B, no SFT
at all) to disk, so verl can load it like any other checkpoint
(`model.path=<dir>, trust_remote_code=true`) for the no-cold-start-SFT GRPO ablation.

Once the real SFT run finishes, point `model.path` at that checkpoint instead -- nothing else
about the GRPO config needs to change, since both are the same `EchoPrimeQwen3ForCausalLM`
architecture, just with different LM weights.
"""
import argparse
import os
import shutil

from echoprime_track.train_sft import build_model_and_tokenizer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    model, tok = build_model_and_tokenizer(args.text_model)
    model.config.auto_map = model.get_auto_map()
    # HF's auto-derived `model.generation_config` (a SEPARATE object from `model.config`, saved
    # to its own generation_config.json) did not pick up config.eos_token_id for this composite
    # model -- confirmed this session: config.json's top-level eos_token_id was correctly 151645
    # after modeling.py's from_cold_start fix, but generation_config.json still saved as
    # essentially empty ({"_from_model_config": true, ...}, no eos_token_id at all). vLLM reads
    # THIS file (not config.json directly) to know what token ends generation, so a real GRPO
    # rollout ran every response to the hard max_tokens ceiling every single time (100% clip
    # ratio, zero variance, at both temperature 1.0 and 0.8) while an in-process HF .generate()
    # call on the same checkpoint stopped early -- HF's generate() falls back to config.
    # eos_token_id directly when generation_config's own field is unset, vLLM does not. Set it
    # explicitly rather than trust the auto-derivation a second time.
    for _id_field in ("eos_token_id", "pad_token_id", "bos_token_id"):
        _value = getattr(model.config, _id_field, None)
        if _value is not None:
            setattr(model.generation_config, _id_field, _value)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # `auto_map` in config.json only points at a module name (`modeling_echoprime_qwen3`) --
    # trust_remote_code=True loading needs that actual file sitting next to config.json, not
    # just the pointer. modeling.py has no `echoprime_track.*` relative imports, so copying it standalone
    # works.
    shutil.copy(os.path.join(os.path.dirname(__file__), "modeling.py"),
                os.path.join(args.out_dir, "modeling_echoprime_qwen3.py"))

    print(f"[echoprime_track] cold-start checkpoint (no SFT) -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
