"""Validate or repair only stored system messages, retaining all dataset content/order."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from echoprime_track.prompts import SYSTEM_PROMPT, validate_tool_prompt


def repair_prompt_column(table):
    import pyarrow as pa

    prompts = copy.deepcopy(table["prompt"].to_pylist())
    for messages in prompts:
        systems = [m for m in messages if m.get("role") == "system"]
        if len(systems) != 1:
            raise ValueError("Expected exactly one system message per row")
        systems[0]["content"] = SYSTEM_PROMPT
    idx = table.schema.get_field_index("prompt")
    return table.set_column(idx, table.schema.field(idx),
                            pa.array(prompts, type=table.schema.field(idx).type))


def check_table(table, tokenizer=None, max_prompt_length=None):
    lengths = []
    for messages in table["prompt"].to_pylist():
        validate_tool_prompt(messages)
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False) + "<think>\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))
    if lengths and max_prompt_length is not None and max(lengths) > max_prompt_length:
        raise ValueError(f"Primed prompt length {max(lengths)} exceeds {max_prompt_length}")
    return {"rows": table.num_rows, "max_primed_prompt_tokens": max(lengths, default=None),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}


def check_file(path, *, repair=False, archive_dir=None, tokenizer=None, max_prompt_length=None):
    import pyarrow.parquet as pq

    path = Path(path)
    original = pq.read_table(path)
    updated = repair_prompt_column(original) if repair else original
    result = check_table(updated, tokenizer, max_prompt_length)
    result["path"] = str(path)
    if repair and not original.equals(updated):
        if archive_dir is None:
            raise ValueError("--archive-dir is required when repairing artifacts")
        # Retain the exact original artifact as provenance before replacing it.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        archive = Path(archive_dir) / digest / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(path, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError("Archived artifact hash mismatch")
        fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            pq.write_table(updated, temp)
            roundtrip = pq.read_table(temp)
            if not updated.equals(roundtrip):
                raise ValueError("Parquet round-trip changed data")
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        result.update(original_sha256=digest, archive=str(archive))
    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--archive-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-prompt-length", type=int, default=3584)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    for path in args.paths:
        print(json.dumps(check_file(path, repair=args.repair, archive_dir=args.archive_dir,
                                    tokenizer=tokenizer,
                                    max_prompt_length=args.max_prompt_length)), flush=True)


if __name__ == "__main__":
    main()
