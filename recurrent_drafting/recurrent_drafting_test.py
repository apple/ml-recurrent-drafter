#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import pytest
import torch

import recurrent_drafting


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("beam_width", [4])
@pytest.mark.parametrize("beam_length", [5])
def test_greedy_prepare_next_input(batch_size: int, beam_width: int, beam_length: int) -> None:
    recurrent_drafting.rng.seed_pytorch(123)
    hidden_size = 32
    vocab_size = 128
    beams_by_llm = torch.randint(high=vocab_size, size=(batch_size, beam_width, beam_length))
    last_hidden_state = torch.rand(batch_size, beam_width, beam_length, hidden_size)
    seq_in_beam = torch.randint(high=beam_width, size=(batch_size,))
    n_tokens_in_seq = torch.randint(high=beam_length - 1, size=(batch_size,))
    last_token_state, next_tokens = (
        recurrent_drafting.recurrent_drafting._greedy_prepare_next_input(
            beams_by_llm, last_hidden_state, seq_in_beam, n_tokens_in_seq
        )
    )
    for bi in range(batch_size):
        assert next_tokens[bi] == beams_by_llm[bi][seq_in_beam[bi]][n_tokens_in_seq[bi]]
        assert torch.allclose(
            last_token_state[bi], last_hidden_state[bi][seq_in_beam[bi]][n_tokens_in_seq[bi]]
        )


@pytest.mark.parametrize(
    ["beams_by_drafter", "beams_by_llm", "expected_n_tokens_in_seq", "expected_seq_in_beam"],
    [
        pytest.param(
            torch.Tensor(
                [
                    [[1, 2, 3, 4], [1, 2, 3, 5], [1, 3, 4, 5]],
                    [[1, 2, 3, 4], [1, 2, 3, 5], [1, 3, 4, 5]],
                    [[1, 2, 3, 4], [1, 2, 3, 5], [1, 3, 4, 5]],
                ]
            ),
            torch.Tensor(
                [
                    [[2, 3, 4, 5], [2, 3, 4, 6], [3, 4, 6, 8]],
                    [[1, 3, 4, 6], [2, 5, 4, 5], [7, 4, 6, 8]],
                    [[8, 3, 4, 6], [8, 5, 4, 5], [8, 4, 6, 8]],
                ]
            ),
            torch.Tensor([3, 1, 0]),
            torch.Tensor([0, 1, 0]),
        )
    ],
)
def test_greedy_choose_from_candidates(
    beams_by_drafter: torch.Tensor,
    beams_by_llm: torch.Tensor,
    expected_n_tokens_in_seq: torch.Tensor,
    expected_seq_in_beam: torch.Tensor,
) -> None:
    n_tokens_in_seq, seq_in_beam = (
        recurrent_drafting.recurrent_drafting._greedy_choose_from_candidates(
            beams_by_drafter, beams_by_llm
        )
    )
    assert torch.all(n_tokens_in_seq == expected_n_tokens_in_seq)
    assert torch.all(seq_in_beam == expected_seq_in_beam)
