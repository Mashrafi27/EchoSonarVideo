import json
import re

import pytest

from echoprime_track.prompts import SYSTEM_PROMPT, validate_tool_prompt


def test_stale_prompt_fails_before_rollout():
    with pytest.raises(ValueError, match="Stale EchoPrime"):
        validate_tool_prompt([{"role": "system", "content": "old prompt"}])


def test_tool_examples_are_executable_json():
    calls = [json.loads(s) for s in re.findall(r"<tool_call>(.*?)</tool_call>", SYSTEM_PROMPT)]
    assert {c["name"] for c in calls} == {"select_frames", "zoom"}
    for call in calls:
        assert call["arguments"]["frame_indices"] == [0, 1]
    assert "<answer>" not in SYSTEM_PROMPT


def test_repair_preserves_all_non_system_data_and_archives_original(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from echoprime_track.check_grpo_prompts import check_file

    path = tmp_path / "train.parquet"
    rows = [{"prompt": [{"role": "system", "content": "old"},
                        {"role": "user", "content": "A4C: <|clip_embed|> question"}],
             "reward_model": {"ground_truth": '{"target":"no"}'},
             "extra_info": {"dicom_uuids_by_view": '{"B":"b","A":"a"}'}}]
    original = pa.Table.from_pylist(rows)
    pq.write_table(original, path)
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="Stale"):
        check_file(path)
    result = check_file(path, repair=True, archive_dir=tmp_path / "provenance")
    from pathlib import Path
    assert Path(result["archive"]).read_bytes() == original_bytes
    fixed = pq.read_table(path).to_pylist()
    assert fixed[0]["prompt"][0]["content"] == SYSTEM_PROMPT
    fixed[0]["prompt"][0]["content"] = "old"
    assert fixed == rows
    assert check_file(path)["rows"] == 1
