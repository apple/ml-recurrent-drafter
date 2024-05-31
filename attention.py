# Copyright © 2024 Apple Inc.
"""This file contains functions to create different types
of attention mask and attention bias.
"""

from typing import Tuple

import mlx.core as mx

LOG_0 = -50000.0


def causal_mask(padding_mask: mx.array, query_len: int) -> mx.array:
    """Create a causal mask from a padding mask.

    For each sequence, the padding mask is a 1D boolean vector. A zero or false element corresponds
    to a padding token in the input, whereas a non-zero or true element denotes a non-padding or
    normal token.

    ┌─────────────────────────────────────┬───────────────────┐
    │    past_key_value_length            │   query_length    │
    └─────────────────────────────────────┴───────────────────┘
      key_value_length = past_key_value_length + query_length

    The output causal mask is a 2D boolean tensor, where the element in row i and column j tells if
    the i-th token in the input should attend to the j-th.

    The range of i and j may differ as rows cover the range of query tokens, which could be 1 token
    to be generated per auto-regression step, whereas columns cover a longer range started with the
    first token in the prompt. As a result, the causal mask may not be square but a wide and short
    matrix with fewer rows than columns.

    Consider the invocation of LLM to verify a generated token "red" given the prompt "Mars is", the
    query length is 1, which is the number of tokens in "red", and the key/value length is 3 that
    covers "Mars is red".

    In practice, the padding mask is in the shape (B, L), where B is the batch size and L is the
    key/value length. The causal mask is in the shape (B, L', L), where L' is the query length.

    Args:
        padding_mask (batch_size, key_value_len)

    Returns:
        causal_mask (batch_size, query_len, key_value_len)

    """
    assert padding_mask.dtype in [mx.int32, mx.int64, mx.bool_]
    batch_size, key_value_len = padding_mask.shape
    causal_mask = mx.tril(mx.full((key_value_len, key_value_len), vals=True))
    batch_causal_mask = mx.repeat(causal_mask[None, :, :], repeats=batch_size, axis=0)
    batch_causal_mask = mx.logical_and(batch_causal_mask, padding_mask[:, None, :])
    assert query_len > 0
    batch_causal_mask = batch_causal_mask[:, -query_len:, :]
    return batch_causal_mask


def bias(causal_mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Create attention bias from the attention mask. Because attention weight and
    bias are in log scale, so this function indeed converts 0 to log(0)=-inf and 1
    to log(1)=0.

    Args:
        causal_mask (batch_size, .., query_len, key_value_len)
        dtype: Used to determine the value of -inf.

    Returns:
        attn_bias (batch_size, .., query_len, key_value_len)
    """
    return mx.logical_not(causal_mask) * LOG_0


def pack(
    beams: mx.array,
    padding_mask: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Flatten each 2D beam in a batch into a vector of candidate tokens without any compression.
    This is to enable the reuse of past key and value in LLM verification step by using the
    corresponding attention mask.

    Consider a beam of two candidate token sequences [[91,92,93], [91,92,94]].  The corresponding
    packed beam is [91,92,93,91,92,94], and the attention mask for the beam is
    [[1,0,0,0,0,0],
     [1,1,0,0,0,0],
     [1,1,1,0,0,0],
     [0,0,0,1,0,0],
     [0,0,0,1,1,0],
     [0,0,0,1,1,1]]
    Prompt attention mask are considered and appended to the left of the attention mask.

      beams: (batch_size, beam_width, beam_length)

      padding_mask: (batch_size, previous_seq_len) The padding mask of the previously generated
      tokens and the prompt. A zero or false element corresponds to a a padding token in the
      sequence.

    Returns:

      packed_beams: (batch_size, beam_width*beam_length)

      causal_mask: (batch_size, beam_width*beam_length, previous_seq_len + beam_width*beam_length)
      When an LLM verifies packed_beams, it expects this causal mask.

      position_offsets: the distance from each packed token to the end of the processed
      sequence. Base LLMs using RoPE need it to derive the position_id by adding it with the length
      of the processed sequence.

    """
    batch_size, beam_width, beam_length = beams.shape
    previous_seq_len = padding_mask.shape[1]
    packed_beams = beams.reshape(batch_size, beam_width * beam_length)
    causal_mask = mx.zeros(
        shape=(
            batch_size,
            beam_width * beam_length,
            previous_seq_len + beam_width * beam_length,
        ),
        dtype=mx.bool_,
    )
    causal_mask[:, :, :previous_seq_len] = padding_mask[:, None]
    # Modified from pytorch tree_attention._get_causal_mask_of_packed_beams
    arange_all_candidate_tokens = mx.arange(beam_width * beam_length)
    beam_blocks = arange_all_candidate_tokens // beam_length
    block_diagonal_mask = beam_blocks[..., None] - beam_blocks[None, ...] == 0
    lower_triangular_mask = (
        arange_all_candidate_tokens[..., None] - arange_all_candidate_tokens[None, ...] >= 0
    )
    causal_mask[:, :, previous_seq_len:] = mx.repeat(
        mx.logical_and(lower_triangular_mask, block_diagonal_mask)[None, :, :],
        repeats=batch_size,
        axis=0,
    )
    position_offsets = mx.repeat(
        (arange_all_candidate_tokens % beam_length)[None, ...], repeats=batch_size, axis=0
    )
    return packed_beams, causal_mask, position_offsets


def unpack(
    packed_array: mx.array,
    beam_width: int,
    beam_length: int,
) -> mx.array:
    """Sending packed beams as input to the base LLM, we get hidden states and vocabulary logits of
    packed tokens. This function unpack them as if they were computed from the unpacked beams. Note
    this unpack assumes no compression is used to pack the beams.

    Consider a beam of two candidate token sequences [[91,92,93], [91,92,94]].  The corresponding
    packed beam would be [91,92,93,91,92,94]. The verification produces the following hidden states:

      [ s(91), s(92), s(93), s(91), s(92), s(94) ]

    and logits

      [ g(91), g(92), g(93), g(91), g(92), g(94) ]

    After unpacking, we get

      [[ s(91), s(92), s(93) ],
       [ s(91), s(92), s(94) ]]

    and logits

      [[ g(91), g(92), g(93) ],
       [ g(91), g(92), g(94) ]]

      packed_array: (batch_size, beam_width * beam_length, last_dim_size)

    Returns:

      unpacked_array: (batch_size, beam_width, beam_length, last_dim_size)

    """
    assert len(packed_array.shape) == 3
    batch_size, width_times_length, last_dim_size = packed_array.shape
    assert beam_width * beam_length == width_times_length
    return packed_array.reshape(batch_size, beam_width, beam_length, last_dim_size)
