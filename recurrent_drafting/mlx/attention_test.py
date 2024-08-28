# Copyright © 2024 Apple Inc.
import mlx.core as mx
import pytest

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
