# Copyright © 2024 Apple Inc.
import mlx.core as mx
import numpy
import pytest
import recurrent_drafting
import torch

from . import tree_attention

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
    ref_packed_beams, _, ref_causal_mask, ref_position_offsets = (
        recurrent_drafting.tree_attention.pack(
            torch.tensor(BEAMS_NO_COMMON_PREFIX), torch.tensor(padding_mask)
        )
    )

    mlx_packed_beams, mlx_causal_mask, mlx_position_offsets = tree_attention.pack(
        mx.array(BEAMS_NO_COMMON_PREFIX), mx.array(padding_mask)
    )

    assert mx.all(mlx_packed_beams == mx.array(ref_packed_beams.numpy())).item()
    assert mx.all(mlx_causal_mask == mx.array(ref_causal_mask.numpy())).item()
    assert mx.all(mlx_causal_mask == expected_attn_mask).item()
    assert mx.all(mlx_position_offsets == mx.array(ref_position_offsets.numpy())).item()
