#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import pytest
import torch

from . import attention

NF = torch.finfo(torch.float32).min


@pytest.mark.parametrize(
    ["padding_mask", "query_len", "expected_causal_mask"],
    [
        pytest.param(
            torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]]),
            1,  # when query contains only one toke, the 2D causal mask is the padding mask
            torch.tensor([[[0, 0, 1]], [[0, 1, 1]], [[1, 1, 1]]]),
        ),
        pytest.param(
            torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]]),
            3,  # query_length == key_value_length or past_key_value_length == 0
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                    [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                ]
            ),
        ),
        pytest.param(
            torch.tensor([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]]),
            2,  # query_length == past_key_value_length ==2; key_value_length ==4
            torch.tensor(
                [
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 1, 0, 0], [0, 1, 0, 1]],
                    [[1, 1, 1, 0], [1, 1, 1, 1]],
                ]
            ),
        ),
    ],
)
def test_causal_mask(
    padding_mask: torch.Tensor, query_len: int, expected_causal_mask: torch.Tensor
):
    causal_mask = attention.causal_mask(padding_mask, query_len, device=torch.device("cpu"))
    assert torch.all(torch.eq(causal_mask, expected_causal_mask))


@pytest.mark.parametrize(
    ["causal_mask", "expected_causal_mask"],
    [
        pytest.param(
            torch.tensor([[[0, 0, 1]], [[0, 1, 1]], [[1, 1, 1]]]),
            torch.tensor([[[0, 0, 1]], [[0, 1, 1]], [[1, 1, 1]]]),
        ),
        pytest.param(
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                    [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 1, 1], [1, 1, 1], [0, 0, 1]],
                    [[1, 1, 1], [0, 1, 0], [0, 1, 1]],
                    [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                ]
            ),
        ),
        pytest.param(
            torch.tensor(
                [
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 1, 0, 0], [0, 1, 0, 1]],
                    [[1, 1, 1, 0], [1, 1, 1, 1]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                    [[0, 1, 0, 0], [0, 1, 0, 1]],
                    [[1, 1, 1, 0], [1, 1, 1, 1]],
                ]
            ),
        ),
    ],
)
def test_set_sdpa_fully_masked_rows(
    causal_mask: torch.Tensor,
    expected_causal_mask: torch.Tensor,
) -> None:
    sdpa_causal_mask = attention.set_sdpa_fully_masked_rows(causal_mask)
    assert torch.all(torch.eq(sdpa_causal_mask, expected_causal_mask))


@pytest.mark.parametrize(
    ["attn_mask", "expected_attn_bias"],
    [
        pytest.param(
            torch.tensor(
                [
                    [[0, 0, 0, 1]],
                    [[0, 1, 1, 1]],
                    [[1, 1, 1, 1]],
                ],
            ),
            torch.tensor(
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
            torch.tensor(
                [
                    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]],
                    [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]],
                    [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]],
                ]
            ),
            torch.tensor(
                [
                    [[[NF, NF, NF, NF], [NF, NF, NF, NF], [NF, NF, NF, NF], [NF, NF, NF, 0]]],
                    [[[NF, NF, NF, NF], [NF, 0, NF, NF], [NF, 0, 0, NF], [NF, 0, 0, 0]]],
                    [[[0, NF, NF, NF], [0, 0, NF, NF], [0, 0, 0, NF], [0, 0, 0, 0]]],
                ]
            ),
        ),
        pytest.param(
            torch.tensor(
                [
                    [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]],
                    [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]]],
                ]
            ),
            torch.tensor(
                [
                    [[[0, NF, NF, NF], [0, 0, NF, NF], [0, NF, 0, NF], [0, NF, NF, 0]]],
                    [[[NF, NF, NF, NF], [NF, 0, NF, NF], [NF, 0, 0, NF], [NF, 0, NF, 0]]],
                ]
            ),
        ),
    ],
)
def test_attn_bias(attn_mask: torch.Tensor, expected_attn_bias: torch.Tensor) -> None:
    attn_bias = attention.bias(attn_mask, dtype=expected_attn_bias.dtype)
    assert torch.all(torch.eq(attn_bias, expected_attn_bias))
