#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import pytest
import torch

from . import tree_attention

T, F = True, False

BEAMS = torch.tensor(
    [  # Assuming a batch of two sequences, each has 3 candidates of 4 tokens.
        [[91, 92, 93, 95], [91, 92, 94, 96], [91, 92, 93, 97]],
        [[93, 94, 95, 92], [93, 95, 96, 93], [93, 94, 97, 96]],
    ]
)

PREFIX_TREE = torch.tensor(
    [
        [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]],
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 2, 2]],
    ]
)

PACKED_BEAMS = torch.tensor(
    [
        [91, 92, 93, 95, 94, 96, 97, 93, 94],  # 7 effective tokens and 2 ignored.
        [93, 94, 95, 92, 95, 96, 93, 97, 96],  # 9 effective tokens
    ]
)

PACKED_TOKEN_INDICES = torch.tensor(
    [
        [0, 1, 2, 3, 6, 7, 11, 0, 1],
        [0, 1, 2, 3, 5, 6, 7, 10, 11],
    ]
)

UNPACKER = torch.tensor(
    [
        [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 6]],
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 1, 7, 8]],
    ]
)


def test_pack_beams():
    packed_beams, packed_token_indices = tree_attention._pack_beams(BEAMS, PREFIX_TREE)
    assert torch.all(packed_beams == PACKED_BEAMS)
    assert torch.all(packed_token_indices == PACKED_TOKEN_INDICES)


def test_get_unpacker():
    assert torch.all(tree_attention._get_unpacker(BEAMS, PREFIX_TREE) == UNPACKER)


@pytest.mark.parametrize(
    ["padding_mask", "expected_packed_causal_mask", "expected_position_offsets"],
    [
        pytest.param(
            torch.tensor([[1, 1, 1], [1, 1, 1]]),  # previous_seq_len=3
            torch.tensor(
                [
                    [
                        [T, T, T, T, F, F, F, F, F, F, F, F],
                        [T, T, T, T, T, F, F, F, F, F, F, F],
                        [T, T, T, T, T, T, F, F, F, F, F, F],
                        [T, T, T, T, T, T, T, F, F, F, F, F],
                        [T, T, T, T, T, F, F, T, F, F, F, F],
                        [T, T, T, T, T, F, F, T, T, F, F, F],
                        [T, T, T, T, T, T, F, F, F, T, F, F],
                        [T, T, T, T, F, F, F, F, F, F, F, F],
                        [T, T, T, T, T, F, F, F, F, F, F, F],
                    ],
                    [
                        [T, T, T, T, F, F, F, F, F, F, F, F],
                        [T, T, T, T, T, F, F, F, F, F, F, F],
                        [T, T, T, T, T, T, F, F, F, F, F, F],
                        [T, T, T, T, T, T, T, F, F, F, F, F],
                        [T, T, T, T, F, F, F, T, F, F, F, F],
                        [T, T, T, T, F, F, F, T, T, F, F, F],
                        [T, T, T, T, F, F, F, T, T, T, F, F],
                        [T, T, T, T, T, F, F, F, F, F, T, F],
                        [T, T, T, T, T, F, F, F, F, F, T, T],
                    ],
                ],
                dtype=torch.bool,
            ),
            torch.tensor([[0, 1, 2, 3, 2, 3, 3, 0, 1], [0, 1, 2, 3, 1, 2, 3, 2, 3]]),
        ),
        pytest.param(
            torch.tensor([[0, 0, 1], [1, 1, 1]]),  # previous_seq_len=[1, 3]
            torch.tensor(
                [
                    [
                        [0, 0, T, T, F, F, F, F, F, F, F, F],
                        [0, 0, T, T, T, F, F, F, F, F, F, F],
                        [0, 0, T, T, T, T, F, F, F, F, F, F],
                        [0, 0, T, T, T, T, T, F, F, F, F, F],
                        [0, 0, T, T, T, F, F, T, F, F, F, F],
                        [0, 0, T, T, T, F, F, T, T, F, F, F],
                        [0, 0, T, T, T, T, F, F, F, T, F, F],
                        [0, 0, T, T, F, F, F, F, F, F, F, F],
                        [0, 0, T, T, T, F, F, F, F, F, F, F],
                    ],
                    [
                        [T, T, T, T, F, F, F, F, F, F, F, F],
                        [T, T, T, T, T, F, F, F, F, F, F, F],
                        [T, T, T, T, T, T, F, F, F, F, F, F],
                        [T, T, T, T, T, T, T, F, F, F, F, F],
                        [T, T, T, T, F, F, F, T, F, F, F, F],
                        [T, T, T, T, F, F, F, T, T, F, F, F],
                        [T, T, T, T, F, F, F, T, T, T, F, F],
                        [T, T, T, T, T, F, F, F, F, F, T, F],
                        [T, T, T, T, T, F, F, F, F, F, T, T],
                    ],
                ],
                dtype=torch.bool,
            ),
            torch.tensor([[0, 1, 2, 3, 2, 3, 3, 0, 1], [0, 1, 2, 3, 1, 2, 3, 2, 3]]),
        ),
    ],
)
def test_pack(
    padding_mask: torch.Tensor,
    expected_packed_causal_mask: torch.Tensor,
    expected_position_offsets: torch.Tensor,
) -> None:
    (
        packed_beams,
        unpacker,
        causal_mask,
        position_offsets,
    ) = tree_attention.pack(BEAMS, padding_mask)

    assert torch.all(packed_beams == PACKED_BEAMS)
    assert torch.all(unpacker == UNPACKER)
    assert torch.all(causal_mask == expected_packed_causal_mask)
    assert torch.all(position_offsets == expected_position_offsets)


@pytest.mark.parametrize(
    ["padding_mask", "expected_packed_causal_mask"],
    [
        pytest.param(
            torch.tensor([[1, 1, 1], [1, 1, 1]]),  # previous_seq_len=3
            torch.tensor(
                [
                    [
                        [T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, T, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, T, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, T, T, 0, 0, 0],
                        [T, T, T, T, T, T, 0, 0, 0, T, 0, 0],
                        [T, T, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, T, 0, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, T, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, T, T, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, T, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, T, T],
                    ],
                ],
                dtype=torch.bool,
            ),
        ),
        pytest.param(
            torch.tensor([[0, 0, 1], [1, 1, 1]]),  # previous_seq_len=[1, 3]
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, T, T, T, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, T, T, T, T, 0, 0, 0, 0, 0],
                        [0, 0, T, T, T, 0, 0, T, 0, 0, 0, 0],
                        [0, 0, T, T, T, 0, 0, T, T, 0, 0, 0],
                        [0, 0, T, T, T, T, 0, 0, 0, T, 0, 0],
                        [0, 0, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, 0, 0, 0, 0, 0, 0],
                        [T, T, T, T, T, T, T, 0, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, 0, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, T, 0, 0, 0],
                        [T, T, T, T, 0, 0, 0, T, T, T, 0, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, T, 0],
                        [T, T, T, T, T, 0, 0, 0, 0, 0, T, T],
                    ],
                ],
                dtype=torch.bool,
            ),
        ),
    ],
)
def test_get_causal_mask(
    padding_mask: torch.Tensor,
    expected_packed_causal_mask: torch.Tensor,
) -> None:
    causal_mask = tree_attention._get_causal_mask(padding_mask, UNPACKER, PACKED_TOKEN_INDICES)
    assert torch.all(causal_mask == expected_packed_causal_mask)


@pytest.mark.parametrize(
    ["packed_candidate_len", "unpacker"],
    [
        pytest.param(3, torch.tensor([[[0, 1, 2]]])),
        pytest.param(
            5, torch.tensor([[[0, 1, 2], [0, 1, 3], [0, 1, 4]], [[0, 1, 2], [0, 2, 3], [0, 2, 4]]])
        ),
    ],
)
def test_unpack(
    packed_candidate_len: int,
    unpacker: torch.Tensor,
    last_dim_size: int = 8,
) -> None:
    batch_size = unpacker.shape[0]
    packed_tensor = torch.rand((batch_size, packed_candidate_len, last_dim_size))
    unpacked_tensor = tree_attention.unpack(packed_tensor, unpacker)
    assert unpacked_tensor.shape[:3] == unpacker.shape
    for batch_i, beam_token_indices in enumerate(unpacker):
        for candidate_i, candidate in enumerate(beam_token_indices):
            for token_i, token_index in enumerate(candidate):
                torch.allclose(
                    unpacked_tensor[batch_i][candidate_i][token_i],
                    packed_tensor[batch_i][token_index],
                )
