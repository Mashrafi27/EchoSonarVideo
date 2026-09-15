import torch

from echoprime_track.modeling import EchoPrimeQwen3ForCausalLM


def test_splice_positions_single_example():
    # 1 example, 5 tokens, token_id=99 at positions [1, 3]
    input_ids = torch.tensor([[0, 99, 0, 99, 0]])
    inputs_embeds = torch.zeros(1, 5, 4)
    projected = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0, 2.0]])
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected, counts=[2])
    assert torch.equal(out[0, 1], projected[0])
    assert torch.equal(out[0, 3], projected[1])
    assert torch.equal(out[0, 0], torch.zeros(4))  # untouched


def test_splice_positions_batch_with_zero_count_row():
    # row 0 has 1 placeholder, row 1 has none
    input_ids = torch.tensor([[99, 0, 0], [0, 0, 0]])
    inputs_embeds = torch.zeros(2, 3, 4)
    projected = torch.tensor([[9.0, 9.0, 9.0, 9.0]])
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected, counts=[1, 0])
    assert torch.equal(out[0, 0], projected[0])
    assert torch.equal(out[1], torch.zeros(3, 4))


def test_splice_positions_count_mismatch_asserts():
    input_ids = torch.tensor([[99, 99, 0]])  # 2 placeholders
    inputs_embeds = torch.zeros(1, 3, 4)
    projected = torch.zeros(1, 4)
    try:
        EchoPrimeQwen3ForCausalLM._splice_positions(
            inputs_embeds, input_ids, token_id=99, projected=projected, counts=[1])
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "placeholders" in str(e)


def test_splice_positions_accepts_tensor_counts():
    # verl's extract_multi_modal_inputs concatenates per-example (1,) tensors this way
    input_ids = torch.tensor([[99, 0]])
    inputs_embeds = torch.zeros(1, 2, 4)
    projected = torch.ones(1, 4)
    out = EchoPrimeQwen3ForCausalLM._splice_positions(
        inputs_embeds, input_ids, token_id=99, projected=projected,
        counts=torch.tensor([[1]]))
    assert torch.equal(out[0, 0], torch.ones(4))
