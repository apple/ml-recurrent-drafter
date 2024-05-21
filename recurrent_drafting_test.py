# Copyright © 2024 Apple Inc.

import unittest.mock
from typing import Tuple

import mlx.core as mx
import numpy
import pytest
import recurrent_drafting
import torch

import mlx_recurrent_drafting
import mlx_recurrent_drafting.recurrent_drafting
from mlx_recurrent_drafting.modeling_drafter import LOG_0


@pytest.mark.parametrize(
    "shape",
    [(7, 8), (7, 8, 9), (7, 8, 9, 10)],
)
def test_select_one_per_row(shape: Tuple[int]) -> None:
    numpy.random.seed(123)
    x = numpy.random.rand(*shape)
    batch_index = numpy.random.randint(low=0, high=shape[1], size=(shape[0],))  # type:ignore
    ref = recurrent_drafting.recurrent_drafting._select_one_per_row(
        torch.tensor(x), torch.tensor(batch_index)
    )

    mlx_output = mlx_recurrent_drafting.recurrent_drafting._select_one_per_row(
        mx.array(x), mx.array(batch_index)
    )

    assert mx.all(mx.allclose(mlx_output, mx.array(ref.numpy()), atol=1e-4, rtol=1e-4))


@pytest.mark.parametrize("batch_size", [1, 4, 13])
@pytest.mark.parametrize("beam_width", [1, 4, 13])
@pytest.mark.parametrize("beam_length", [1, 4, 13])
def test_greedy_prepare_next_input(batch_size: int, beam_width: int, beam_length: int) -> None:
    numpy.random.seed(123)
    vocab_size, hidden_size = 100, 100
    beams_by_llm = numpy.random.randint(
        low=0, high=vocab_size, size=(batch_size, beam_width, beam_length)
    )
    last_hidden_state = numpy.random.rand(batch_size, beam_width, beam_length, hidden_size)
    seq_in_beam = numpy.random.randint(low=0, high=beam_width, size=(batch_size,))
    n_tokens_in_seq = numpy.random.randint(low=0, high=beam_length, size=(batch_size,))

    ref_state, ref_tokens = recurrent_drafting.recurrent_drafting._greedy_prepare_next_input(
        torch.tensor(beams_by_llm),
        torch.tensor(last_hidden_state),
        torch.tensor(seq_in_beam),
        torch.tensor(n_tokens_in_seq),
    )

    mlx_state, mlx_tokens = mlx_recurrent_drafting.recurrent_drafting._greedy_prepare_next_input(
        mx.array(beams_by_llm),
        mx.array(last_hidden_state),
        mx.array(seq_in_beam),
        mx.array(n_tokens_in_seq),
    )

    assert mx.all(mx.allclose(mlx_state, mx.array(ref_state.numpy()), atol=1e-4, rtol=1e-4))
    assert mx.all(mlx_tokens == ref_tokens)


@pytest.mark.parametrize("batch_size", [1, 4, 13])
@pytest.mark.parametrize("beam_width", [1, 4, 13])
@pytest.mark.parametrize("beam_length", [2, 4, 13])
def test_prepare_next_input(batch_size: int, beam_width: int, beam_length: int) -> None:
    numpy.random.seed(123)
    vocab_size = hidden_size = 100
    log_probs_by_drafter = numpy.random.uniform(
        low=LOG_0, high=0, size=(batch_size, beam_width, beam_length - 1, vocab_size)
    )
    log_probs_by_llm = numpy.full(
        shape=(batch_size, beam_width, beam_length, vocab_size), fill_value=LOG_0
    )

    # Manually set the highest log probs to avoid the randomness of sampling
    d1 = numpy.arange(batch_size)[:, None, None]
    d2 = numpy.arange(beam_width)[None, :, None]
    d3 = numpy.arange(beam_length)[None, None, :]
    d1 = numpy.broadcast_to(d1, (batch_size, beam_width, beam_length))
    d2 = numpy.broadcast_to(d2, (batch_size, beam_width, beam_length))
    d3 = numpy.broadcast_to(d3, (batch_size, beam_width, beam_length))
    llm_argmax_indices = numpy.random.randint(
        low=0, high=vocab_size, size=(batch_size, beam_width, beam_length)
    )
    log_probs_by_llm[d1, d2, d3, llm_argmax_indices] = 0

    last_hidden_state = numpy.random.rand(batch_size, beam_width, beam_length, hidden_size)

    seq_in_beam = numpy.random.randint(low=0, high=beam_width, size=(batch_size,))
    n_tokens_in_seq = numpy.random.randint(low=0, high=beam_length, size=(batch_size,))

    ref_state, ref_tokens = recurrent_drafting.recurrent_drafting._prepare_next_input(
        torch.tensor(log_probs_by_drafter),
        torch.tensor(log_probs_by_llm),
        torch.tensor(last_hidden_state),
        torch.tensor(seq_in_beam),
        torch.tensor(n_tokens_in_seq),
    )

    mlx_state, mlx_tokens = mlx_recurrent_drafting.recurrent_drafting._prepare_next_input(
        mx.array(log_probs_by_drafter),
        mx.array(log_probs_by_llm),
        mx.array(last_hidden_state),
        mx.array(seq_in_beam),
        mx.array(n_tokens_in_seq),
    )

    assert mx.all(mx.allclose(mlx_state, mx.array(ref_state.numpy()), atol=1e-4, rtol=1e-4))
    assert mx.all(mlx_tokens == mx.array(ref_tokens.numpy()))


@pytest.mark.parametrize("batch_size", [1, 4, 13])
@pytest.mark.parametrize("beam_width", [1, 4, 13])
@pytest.mark.parametrize("beam_length", [2, 4, 13])
def test_greedy_choose_from_candidates(batch_size: int, beam_width: int, beam_length: int) -> None:
    numpy.random.seed(123)
    vocab_size = 2  # set a small number for more matches
    beams_by_drafter = numpy.random.randint(
        low=0, high=vocab_size, size=(batch_size, beam_width, beam_length)
    )
    beams_by_llm = numpy.random.randint(
        low=0, high=vocab_size, size=(batch_size, beam_width, beam_length)
    )

    ref_n_tokens_in_seq, _ = recurrent_drafting.recurrent_drafting._greedy_choose_from_candidates(
        torch.tensor(beams_by_drafter), torch.tensor(beams_by_llm)
    )
    mlx_n_tokens_in_seq, mlx_seq_in_beam = (
        mlx_recurrent_drafting.recurrent_drafting._greedy_choose_from_candidates(
            mx.array(beams_by_drafter), mx.array(beams_by_llm)
        )
    )

    assert mx.all(mlx_n_tokens_in_seq == mx.array(ref_n_tokens_in_seq.numpy()))
    for batch_i in range(batch_size):
        selected_beam_by_drafter_by_mx = mx.array(
            beams_by_drafter[batch_i][mlx_seq_in_beam[batch_i]]
        )
        selected_beam_by_llm_by_mx = mx.array(beams_by_llm[batch_i][mlx_seq_in_beam[batch_i]])
        assert mx.all(
            selected_beam_by_drafter_by_mx[1 : mlx_n_tokens_in_seq[batch_i].item() + 1]
            == selected_beam_by_llm_by_mx[: mlx_n_tokens_in_seq[batch_i].item()]
        )


@pytest.mark.parametrize("batch_size", [1, 4, 13])
@pytest.mark.parametrize("beam_width", [1, 4, 13])
@pytest.mark.parametrize("beam_length", [2, 4, 13])
@unittest.mock.patch("mlx.core.random.uniform")
@unittest.mock.patch("torch.rand")
def test_choose_from_candidates(
    torch_rand: unittest.mock.MagicMock,  # order matters
    mx_rand: unittest.mock.MagicMock,  # order of patches should be reversed
    batch_size: int,
    beam_width: int,
    beam_length: int,
) -> None:
    numpy.random.seed(123)
    vocab_size = 2
    beams = numpy.random.randint(low=0, high=vocab_size, size=(batch_size, beam_width, beam_length))
    log_probs_by_llm = numpy.random.uniform(
        low=-10, high=0, size=(batch_size, beam_width, beam_length, vocab_size)
    )
    log_probs_by_drafter = numpy.random.uniform(
        low=-10, high=0, size=(batch_size, beam_width, beam_length - 1, vocab_size)
    )

    rand_vals = numpy.random.rand(batch_size, beam_width, 1)
    mx_rand.return_value = mx.array(rand_vals)
    torch_rand.return_value = torch.tensor(rand_vals)

    ref_n_tokens_in_seq, ref_seq_in_beam = (
        recurrent_drafting.recurrent_drafting._choose_from_candidates(
            torch.tensor(beams), torch.tensor(log_probs_by_llm), torch.tensor(log_probs_by_drafter)
        )
    )
    mlx_n_tokens_in_seq, mlx_seq_in_beam = (
        mlx_recurrent_drafting.recurrent_drafting._choose_from_candidates(
            mx.array(beams), mx.array(log_probs_by_llm), mx.array(log_probs_by_drafter)
        )
    )

    assert mx.all(mlx_n_tokens_in_seq == mx.array(ref_n_tokens_in_seq))
    # TODO how to verify ref_seq_in_beam and mlx_seq_in_beam? The max selection can be different.
