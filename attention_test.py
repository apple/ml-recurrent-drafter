# Copyright © 2024 Apple Inc.
import mlx.core as mx
import numpy
import pytest
import torch
from recurrent_drafting import tree_attention

from . import attention

NF = attention.LOG_0


@pytest.mark.parametrize(
    ["padding_mask", "query_len", "expected_causal_mask"],
    [
        pytest.param(
            mx.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]]),
            1,  # when query contains only one toke, the 2D causal mask is the padding mask
            mx.array([[[0, 0, 1]], [[0, 1, 1]], [[1, 1, 1]]]),
        ),
        pytest.param(
            mx.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]]),
            3,  # query_length == key_value_length or past_key_value_length == 0
            mx.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                    [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                ]
            ),
        ),
        pytest.param(
            mx.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]]),
            2,  # query_length == past_key_value_length ==2; key_value_length ==4
            mx.array(
                [
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 1, 0, 0], [0, 1, 0, 1]],
                    [[1, 1, 1, 0], [1, 1, 1, 1]],
                ]
            ),
        ),
    ],
)
def test_causal_mask(padding_mask: mx.array, query_len: int, expected_causal_mask: mx.array):
    causal_mask = attention.causal_mask(padding_mask, query_len)
    assert mx.all(mx.equal(causal_mask, expected_causal_mask))


@pytest.mark.parametrize(
    ["attn_mask", "expected_attn_bias"],
    [
        pytest.param(
            mx.array(
                [
                    [[0, 0, 0, 1]],
                    [[0, 1, 1, 1]],
                    [[1, 1, 1, 1]],
                ],
            ),
            mx.array(
                [
                    [
                        [[NF, NF, NF, 0]],
                        [[NF, 0, 0, 0]],
                        [[0, 0, 0, 0]],
                    ]
                ]
            ),
        ),
        pytest.param(
            mx.array(
                [
                    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]],
                    [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]],
                    [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]],
                ]
            ),
            mx.array(
                [
                    [[[NF, NF, NF, NF], [NF, NF, NF, NF], [NF, NF, NF, NF], [NF, NF, NF, 0]]],
                    [[[NF, NF, NF, NF], [NF, 0, NF, NF], [NF, 0, 0, NF], [NF, 0, 0, 0]]],
                    [[[0, NF, NF, NF], [0, 0, NF, NF], [0, 0, 0, NF], [0, 0, 0, 0]]],
                ]
            ),
        ),
        pytest.param(
            mx.array(
                [
                    [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]],
                    [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]]],
                ]
            ),
            mx.array(
                [
                    [[[0, NF, NF, NF], [0, 0, NF, NF], [0, NF, 0, NF], [0, NF, NF, 0]]],
                    [[[NF, NF, NF, NF], [NF, 0, NF, NF], [NF, 0, 0, NF], [NF, 0, NF, 0]]],
                ]
            ),
        ),
    ],
)
def test_attn_bias(attn_mask: mx.array, expected_attn_bias: mx.array) -> None:
    attn_bias = attention.bias(attn_mask, dtype=expected_attn_bias.dtype)
    assert mx.all(mx.equal(attn_bias, expected_attn_bias))


BEAMS_NO_COMMON_PREFIX = numpy.array(
    [  # Assuming a batch of two sequences, each has 3 candidates of 4 tokens.
        # Use no common prefix to avoid the compression by tree_attention.
        [[1, 2, 3, 3], [2, 2, 4, 4], [3, 2, 3, 3]],
    ]
)


@pytest.mark.parametrize(
    ["prompt_len", "expected_attn_mask"],
    [
        pytest.param(
            1,
            mx.array(
                [
                    [
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    ]
                ]
            ),
        ),
        pytest.param(
            7,
            mx.array(
                [
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    ]
                ]
            ),
        ),
    ],
)
def test_pack(prompt_len: int, expected_attn_mask: mx.array, batch_size: int = 1) -> None:
    padding_mask = numpy.ones(shape=(batch_size, prompt_len)).astype(numpy.bool_)
    ref_packed_beams, _, ref_causal_mask, ref_position_offsets = tree_attention.pack(
        torch.tensor(BEAMS_NO_COMMON_PREFIX), torch.tensor(padding_mask)
    )

    mlx_packed_beams, mlx_causal_mask, mlx_position_offsets = attention.pack(
        mx.array(BEAMS_NO_COMMON_PREFIX), mx.array(padding_mask)
    )

    assert mx.all(mlx_packed_beams == mx.array(ref_packed_beams.numpy())).item()
    assert mx.all(mlx_causal_mask == mx.array(ref_causal_mask.numpy())).item()
    assert mx.all(mlx_causal_mask == expected_attn_mask).item()
    assert mx.all(mlx_position_offsets == mx.array(ref_position_offsets.numpy())).item()
