# Copyright © 2024 Apple Inc.

import unittest.mock
from typing import Tuple

import mlx.core as mx
import numpy
import pytest
import recurrent_drafting
import torch

import mlx_recurrent_drafting
from mlx_recurrent_drafting.modeling_drafter import LOG_0

from . import modeling_llama_test, tree_attention_test


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
    # Check the selected sequence has the correct number of tokens accepted == mlx_n_tokens_in_seq
    # Comparing mlx_seq_in_beam == ref_seq_in_beam can be incorrect because mlx and pytorch can
    # choose different longest sequence.
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

    rand_vals = numpy.random.rand(batch_size, beam_width, beam_length - 1)
    mx_rand.return_value = mx.array(rand_vals)
    torch_rand.return_value = torch.tensor(rand_vals)

    ref_n_tokens_in_seq, _ = recurrent_drafting.recurrent_drafting._choose_from_candidates(
        torch.tensor(beams), torch.tensor(log_probs_by_llm), torch.tensor(log_probs_by_drafter)
    )
    mlx_n_tokens_in_seq, mlx_seq_in_beam = (
        mlx_recurrent_drafting.recurrent_drafting._choose_from_candidates(
            mx.array(beams), mx.array(log_probs_by_llm), mx.array(log_probs_by_drafter)
        )
    )

    assert mx.all(mlx_n_tokens_in_seq == mx.array(ref_n_tokens_in_seq))
    # Check the selected sequence has the correct number of tokens accepted == mlx_n_tokens_in_seq.
    # Comparing mlx_seq_in_beam == ref_seq_in_beam can be incorrect because mlx and pytorch can
    # choose different longest sequence.
    for batch_i in range(batch_size):
        seq_i = mlx_seq_in_beam[batch_i]
        cur_beam = mx.array(beams[batch_i][seq_i][1:])
        cur_log_probs_llm = mx.array(log_probs_by_llm[batch_i][seq_i][:-1])[
            mx.arange(beam_length - 1), cur_beam
        ]
        cur_log_probs_drafter = mx.array(log_probs_by_drafter[batch_i][seq_i])[
            mx.arange(beam_length - 1), cur_beam
        ]
        cm = mx.exp(cur_log_probs_llm - cur_log_probs_drafter) > mx.array(rand_vals[batch_i][seq_i])
        count_accepted = mx.sum(mx.cumprod(cm.astype(mx.int32), axis=-1), axis=-1)
        assert count_accepted.item() == mlx_n_tokens_in_seq[batch_i]


@pytest.mark.parametrize("shape", [(1, 4), (3, 17)])
def test_count_left_paddings(shape: Tuple[int]) -> None:
    numpy.random.seed(123)
    x = numpy.random.randint(low=0, high=2, size=shape)
    ref_o = recurrent_drafting.recurrent_drafting._count_left_paddings(torch.tensor(x), 0)
    mlx_o = mlx_recurrent_drafting.recurrent_drafting._count_left_paddings(mx.array(x), 0)
    assert mx.all(mlx_o == mx.array(ref_o))


@pytest.mark.parametrize(["beam_width", "beam_length"], [pytest.param(1, 3), pytest.param(3, 17)])
@pytest.mark.parametrize("past_kv_len", [0, 1, 7])
def test_present_kv_as_beam(beam_width: int, beam_length: int, past_kv_len: int) -> None:
    numpy.random.seed(123)
    batch_size, max_len, n_layers, n_heads, head_dim = 2, 100, 4, 4, 4
    ref_cache = recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, torch.float32, device=torch.device("cpu")
    )
    mlx_cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, mx.float32, device=mx.gpu
    )
    for layer_i in range(n_layers):
        for kv_i in range(2):
            kv_len = past_kv_len + beam_width * beam_length
            kvs = numpy.random.rand(batch_size, n_heads, kv_len, head_dim)
            ref_cache.sliced[layer_i][kv_i].cat(torch.tensor(kvs))
            mlx_cache.sliced[layer_i][kv_i].cat(mx.array(kvs))

    ref_kv = recurrent_drafting.recurrent_drafting._present_kv_as_beam(
        recurrent_drafting.modeling_drafter.BeamShape(beam_width, beam_length),
        past_kv_len,
        ref_cache,
    )
    mlx_kv = mlx_recurrent_drafting.recurrent_drafting._present_kv_as_beam(
        mlx_recurrent_drafting.modeling_drafter.BeamShape(beam_width, beam_length),
        past_kv_len,
        mlx_cache,
    )

    assert len(mlx_kv) == n_layers
    for layer_i in range(n_layers):
        for kv_j in range(2):
            assert mx.all(
                mx.allclose(mlx_kv[layer_i][kv_j], mx.array(ref_kv[layer_i][kv_j].numpy()))
            )


@pytest.mark.parametrize(["beam_width", "beam_length"], [pytest.param(1, 3), pytest.param(3, 17)])
@pytest.mark.parametrize("past_kv_len", [1, 7])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_update_kv_cache_and_input_ids(
    batch_size: int, beam_width: int, beam_length: int, past_kv_len: int
) -> None:
    numpy.random.seed(123)
    max_len, n_layers, n_heads, head_dim = 100, 4, 4, 4
    input_ids = numpy.random.randint(low=0, high=100, size=(batch_size, past_kv_len))
    seq_in_beam = numpy.random.randint(low=0, high=beam_width, size=(batch_size,))
    n_tokens_in_seq = numpy.random.randint(low=0, high=beam_length - 1, size=(batch_size,))
    beams = numpy.random.randint(low=1, high=100, size=(batch_size, beam_width, beam_length))
    beams[..., 0] = 1  # token sampled from llm

    ref_cache = recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, torch.float32, device=torch.device("cpu")
    )
    mlx_cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, mx.float32, device=mx.gpu
    )
    for layer_i in range(n_layers):
        for kv_i in range(2):
            kv_len = past_kv_len + beam_width * beam_length
            kvs = numpy.random.rand(batch_size, n_heads, kv_len, head_dim)
            ref_cache.sliced[layer_i][kv_i].cat(torch.tensor(kvs))
            mlx_cache.sliced[layer_i][kv_i].cat(mx.array(kvs))

    ref_appended_input_ids = recurrent_drafting.recurrent_drafting._update_kv_cache_and_input_ids(
        torch.tensor(input_ids),
        torch.tensor(n_tokens_in_seq),
        torch.tensor(seq_in_beam),
        torch.tensor(beams),
        ref_cache,
        pad_token_id=0,
    )

    mlx_appended_input_ids = (
        mlx_recurrent_drafting.recurrent_drafting._update_kv_cache_and_input_ids(
            mx.array(input_ids),
            mx.array(n_tokens_in_seq),
            mx.array(seq_in_beam),
            mx.array(beams),
            mlx_cache,
            pad_token_id=0,
        )
    )
    assert mx.all(mlx_appended_input_ids == mx.array(ref_appended_input_ids.numpy()))


@pytest.mark.parametrize("prompt_len", [1, 7])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("temperature", [0.1, 0.5])
@pytest.mark.parametrize("greedy", [True, False])
@unittest.mock.patch("mlx.core.random.categorical")
@unittest.mock.patch("torch.distributions.Categorical")
def test_comprehend_prompt(
    ref_categorical_mock: unittest.mock.MagicMock,
    mlx_categorical_mock: unittest.mock.MagicMock,
    temperature: float,
    greedy: bool,
    batch_size: int,
    prompt_len: int,
):
    numpy.random.seed(123)

    ref_llm, mlx_llm = modeling_llama_test.create_test_models()
    config = ref_llm.config
    n_layers, n_heads, head_dim = (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    max_len, vocab_size = prompt_len + 1, config.vocab_size
    pad_token_id = 0

    logits = numpy.random.rand(batch_size, prompt_len, vocab_size) * LOG_0
    samples = logits[:, -1, :].argmax(axis=-1)
    sample_mock = unittest.mock.Mock()
    sample_mock.sample.return_value = torch.tensor(samples)
    ref_categorical_mock.return_value = sample_mock
    mlx_categorical_mock.return_value = torch.tensor(samples)

    input_ids = numpy.random.randint(low=1, high=vocab_size, size=(batch_size, prompt_len))

    ref_cache = recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, torch.float32, device=torch.device("cpu")
    )
    mlx_cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, mx.float32, device=mx.gpu
    )

    ref_state, ref_token = recurrent_drafting.recurrent_drafting._comprehend_prompt(
        ref_llm,
        torch.tensor(input_ids),
        ref_cache,
        recurrent_drafting.recurrent_drafting.SamplingArgs(temperature, greedy),
        pad_token_id,
    )

    mlx_state, mlx_token = mlx_recurrent_drafting.recurrent_drafting._comprehend_prompt(
        mlx_llm,
        mx.array(input_ids),
        mlx_cache,
        mlx_recurrent_drafting.recurrent_drafting.SamplingArgs(temperature, greedy),
        pad_token_id,
    )

    assert mx.all(mx.allclose(mlx_state, mx.array(ref_state)))
    assert mx.all(mlx_token == mx.array(ref_token))

    # check the probs/logits for sampling
    if not greedy:
        mlx_categorical_mock.assert_called()
        assert mx.all(
            mx.allclose(
                mx.softmax(mlx_categorical_mock.call_args.kwargs["logits"], axis=-1),
                mx.array(ref_categorical_mock.call_args.kwargs["probs"]),
            )
        )


@torch.inference_mode()
@pytest.mark.parametrize("prompt_len", [1, 7])
def test_verify_candidates(prompt_len: int) -> None:
    numpy.random.seed(123)
    batch_size = 1
    beams = tree_attention_test.BEAMS_NO_COMMON_PREFIX
    beam_width, beam_length = beams.shape[1], beams.shape[2]

    ref_llm, mlx_llm = mlx_recurrent_drafting.modeling_llama_test.create_test_models()
    config = ref_llm.config
    n_layers, n_heads, head_dim = (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    max_len, vocab_size = prompt_len + beam_width * beam_length, config.vocab_size
    pad_token_id = 0

    input_ids = numpy.random.randint(low=1, high=vocab_size, size=(batch_size, prompt_len))

    ref_cache = recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, torch.float32, device=torch.device("cpu")
    )
    mlx_cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size, max_len, n_layers, n_heads, head_dim, mx.float32, device=mx.gpu
    )
    for layer_i in range(n_layers):
        for kv_i in range(2):
            kvs = numpy.random.rand(batch_size, n_heads, prompt_len, head_dim)
            ref_cache.sliced[layer_i][kv_i].cat(torch.tensor(kvs))
            mlx_cache.sliced[layer_i][kv_i].cat(mx.array(kvs))

    stats_mock = unittest.mock.Mock()
    stats_mock.time.return_value.__enter__ = lambda *_: None
    stats_mock.time.return_value.__exit__ = lambda *_: None
    ref_states, ref_logits = recurrent_drafting.recurrent_drafting._verify_candidates(
        stats_mock,
        ref_llm,
        torch.tensor(input_ids),
        torch.tensor(tree_attention_test.BEAMS_NO_COMMON_PREFIX),
        ref_cache,
        pad_token_id,
    )

    mlx_states, mlx_logits = mlx_recurrent_drafting.recurrent_drafting._verify_candidates(
        mlx_llm,
        mx.array(input_ids),
        mx.array(tree_attention_test.BEAMS_NO_COMMON_PREFIX),
        mlx_cache,
        pad_token_id,
    )

    assert mx.all(mx.allclose(mlx_states, mx.array(ref_states.numpy())))
    assert mx.all(mx.allclose(mlx_logits, mx.array(ref_logits.numpy())))
    assert mx.all(
        mx.allclose(mlx_cache._cache, mx.array(ref_cache._cache.numpy()), atol=1e-4, rtol=1e-4)
    )
