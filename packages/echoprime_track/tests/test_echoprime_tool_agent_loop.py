import torch

from echoprime_track.echoprime_tool_agent_loop import resolve_tool_call


def _fake_grid():
    # (393, 768) grid where every token's first value is its own index, for easy assertion
    g = torch.zeros(393, 768)
    for i in range(393):
        g[i, 0] = i
    return g


def test_resolve_select_frames_returns_group_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A4C", "frame_indices": [0, 1]}},
        view_grids={"A4C": grid})
    assert tokens.shape == (49, 768)
    assert tokens[0, 0].item() == 1  # group 0 starts at absolute index 1
    assert "A4C" in text


def test_resolve_zoom_returns_spatial_subset():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "zoom", "arguments": {"view": "A4C", "bbox": [0.0, 0.0, 0.5, 0.5],
                                        "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert 0 < tokens.shape[0] < 49
    assert tokens.shape[1] == 768


def test_resolve_unknown_view_returns_error_text_no_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A2C", "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown view" in text


def test_resolve_unknown_tool_name_returns_error():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_view", "arguments": {"view": "A4C"}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown tool" in text


def test_tool_loop_continues_through_both_tools_and_masks_observations(tmp_path):
    """Exercise the real loop with synthetic model outputs, without claiming policy tool use."""
    import asyncio
    import h5py
    import json
    import re
    from types import SimpleNamespace
    import pytest
    import echoprime_track.echoprime_tool_agent_loop as module
    from echoprime_track.prompts import SYSTEM_PROMPT

    if not module._VERL_AVAILABLE:
        pytest.skip("full loop requires verl")

    class Tokenizer:
        def __call__(self, text, **kwargs):
            parts = re.split(r"(<\|clip_embed\|>|<\|detr_embed\|>)", text)
            ids = []
            for part in parts:
                if part in (module.CLIP_TOKEN, module.DETR_TOKEN):
                    ids.append(self.convert_tokens_to_ids(part))
                else:
                    ids.extend(ord(c) + 10 for c in part)
            return {"input_ids": ids}

        def convert_tokens_to_ids(self, token):
            return {module.CLIP_TOKEN: 1, module.DETR_TOKEN: 2}[token]

        def decode(self, ids, **kwargs):
            return "".join({1: module.CLIP_TOKEN, 2: module.DETR_TOKEN}.get(i, chr(i - 10))
                           if i >= 10 else {1: module.CLIP_TOKEN, 2: module.DETR_TOKEN}[i]
                           for i in ids)

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages) + "\nassistant\n"

    tok = Tokenizer()
    texts = [
        'Inspect frames.</think><tool_call>{"name":"select_frames",'
        '"arguments":{"view":"A4C","frame_indices":[0,1]}}</tool_call>',
        '<think>Inspect a region.</think><tool_call>{"name":"zoom",'
        '"arguments":{"view":"A4C","frame_indices":[0,1],'
        '"bbox":[0.0,0.0,0.5,0.5]}}</tool_call>',
        '<think>Finished.</think>No.',
    ]
    requests = []

    class Server:
        async def generate(self, **kwargs):
            requests.append(kwargs)
            ids = tok(texts[len(requests) - 1])["input_ids"]
            return SimpleNamespace(token_ids=ids, log_probs=[-0.5] * len(ids), num_preempted=0)

    with h5py.File(tmp_path / "clip.h5", "w") as clip, h5py.File(tmp_path / "detr.h5", "w") as detr:
        clip.create_group("synthetic-view").create_dataset("tokens", data=_fake_grid().numpy())
        loop = object.__new__(module.EchoPrimeToolAgentLoop)
        loop.tokenizer = tok
        loop.server_manager = Server()
        loop.response_length = 2048
        loop._clip_h5, loop._detr_h5 = clip, detr
        result = asyncio.run(loop.run({}, raw_prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "A4C:\n" + module.CLIP_TOKEN * 393 + "\nQuestion?"},
        ], extra_info={"study_uuid": "synthetic-study",
                       "dicom_uuids_by_view": json.dumps({"A4C": "synthetic-view"})}))

    assert len(requests) == 3
    assert result.num_turns == 4  # historical 2 + two tool continuations
    assert tok.decode(requests[0]["prompt_ids"]).endswith("<think>\n")
    assert "select_frames" in tok.decode(requests[0]["prompt_ids"])
    assert "<tool_response>" in tok.decode(requests[1]["prompt_ids"])
    assert requests[1]["clip_data"].shape[0] == 393 + 49
    assert 393 + 49 < requests[2]["clip_data"].shape[0] < 393 + 98
    assert result.multi_modal_data["clip_counts"] == requests[2]["clip_data"].shape[0]
    generated = [i for i, mask in zip(result.response_ids, result.response_mask) if mask]
    assert tok.decode(generated) == "".join(texts)
    assert all(p == 0 for p, mask in zip(result.response_logprobs, result.response_mask) if not mask)
