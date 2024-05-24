#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Please refer to docs/tree_attention.md for general ideas of tree attention."""

from typing import Tuple

import torch

from . import attention


def _dedup_prefix(beams: torch.Tensor) -> torch.Tensor:
    """For each prefix in each candidate, find the smallest candidate index that shares the same
    prefix. For context and examples, refer to docs/tree_attention.md#From_Beam_to_Prefix_Tree.
    For algorithmic details, refer to docs/beam_to_prefix_tree.py

      beams: (batch_size, beam_width, beam_length)

    Returns:

      prefix_tree: (batch_size, beam_width, beam_length) prefix_tree[b][i][j]==k, k<=i
      indicates that candidate sequences i and k in batch b share the same prefix, or, in other
      words, beams[b][i][:j+1]== beams[b][k][:j+1]

    """
    device, dtype = beams.device, beams.dtype
    beam_length = beams.shape[2]
    prefix_target = torch.arange(1, beam_length + 1, dtype=dtype, device=device)
    # For each sequence b in the batch, build a square boolean matrix matches[b]. If
    # matches[b][i][j][k]==True, then in beams[b], the k-th token of the i-th beams is the same as
    # the k-th token in the j-th beams. So, i and j are in range [0,beam_width), k in [0,beam_len).
    matches = (beams[:, :, None] == beams[:, None, :]).to(dtype)
    seq_matches = (torch.cumsum(matches, dim=3) == prefix_target[None, None, None, :]).to(dtype)
    # The previous candidate with smallest index that shares the same prefix.
    prefix_tree = torch.argmax(seq_matches, dim=2)
    return prefix_tree


def _pack_beams(
    beams: torch.Tensor, prefix_tree: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack each 2D beams into a 1D token sequence given prefix_tree. Please refer to
    test_pack_beams for examples.

      beams: (batch_size, beam_width, beam_length)

      prefix_tree: (batch_size, beam_width, beam_length) Refer to _dedup_prefix for explanation.

    Returns:

      packed_beams: (batch_size, n_selected) Flattened and compact draft token
      representations. n_selected is the number of selected candidate tokens per sequence from the
      beam.

      packed_token_indices: (batch_size, n_selected) indexes into the flattened beam and select
      candidates in the prefix tree. packed_token_indices//beam_length tells the candidate token
      sequence from which the token is selected. packed_token_indices%beam_length tells the location
      in the sequence from which the token is selected.

    """
    device, dtype = beams.device, beams.dtype
    batch_size, beam_width, _ = beams.shape

    # Step 1. Collect all tokens to be selected from beams in a vector.

    # (batch_size, beam_width*beam_length) where 1 in pack_mask indicates the corresponding token in
    # beams should be selected and packed, 0 indicates the token has appeared in a previous
    # candidate sequence and has been packed. If you don't understand the comparison, you may want
    # to make sure that you understand docs/pairwise_comparison.py.
    ith_row_has_value_i = torch.arange(beam_width, device=device)[None, :, None]
    pack_mask = (prefix_tree == ith_row_has_value_i).reshape(batch_size, -1)

    # (n_selected_tokens, 2) locations of selected tokens from all sequences in the
    # batch. n_selected_tokens is the number of non-zeros in pack_mask. Each row of location is a
    # pair [seq-in-batch, token-in-beam_width*beam_length].
    locations = torch.nonzero(pack_mask)

    # (batch_size, beam_width*beam_length) Flatten the beam_width and beam_length dims of beams.
    flattened_beams = beams.view((batch_size, -1))

    # (n_selected_tokens,) Tokens selected from all beams for the batch. Please be aware that it is
    # likely that variable number of tokens are selected from various sequences, so we put tokens
    # selected from all beams in a single vector.
    selected_tokens = flattened_beams[locations[:, 0], locations[:, 1]]

    # Step 2. Reorganize the vector into 2D packed_beams.
    #
    # selected_tokens is the most compact form that does not include any unused
    # tokens. However, it flattens sequences in a batch that the LLM cannot verify them using the KV
    # cache organized in a sequence-by-sequence per batch fashion. So we need to convert the 1D
    # selected_tokens back to 2D in the shape (batch_size, max_n_selected).

    # (batch_size,) the number of tokens selected in each sequence.
    n_selected = pack_mask.sum(dim=1)

    # Among sequences in a batch, the max number of selected tokens.
    max_n_selected = int(n_selected.max())

    # (batch_size,) The starting location in selected_tokens of each sequence.
    starts = torch.cumsum(n_selected, dim=0) - n_selected

    # (batch_size, max_n_selected) Indexing selected_tokens to form the 2D packed_beams.
    select_indices = starts[:, None] + torch.arange(max_n_selected, dtype=dtype, device=device)

    # Clamp select_indices in [0, selected_tokens.numel()).
    select_indices = torch.minimum(
        select_indices, torch.full_like(select_indices, selected_tokens.numel() - 1)
    ).view(-1)

    # Recover the 1D selected_tokens to the 2D packed_beams.
    packed_beams = torch.gather(selected_tokens, dim=0, index=select_indices).reshape(
        batch_size, -1
    )

    # Similarly, recover the per-sequence indices
    packed_token_indices = torch.gather(locations[:, 1], dim=0, index=select_indices).reshape(
        batch_size, -1
    )

    return packed_beams, packed_token_indices


def _get_unpacker(beams: torch.Tensor, prefix_tree: torch.Tensor) -> torch.Tensor:
    """For each draft token, calculate its index in a flattened and compact representation.  This
    function implements an algorithm that does not use loops.  For a high-level description of this
    algorithm, please refer to docs/tree_attention.py, section The Unpacker.

      beams: (batch_size, beam_width, beam_length) Batched input candidates.

      prefix_tree: (batch_size, beam_width, beam_length) Refer to _dedup_prefix for explanation.

    Returns:

      unpacker: (batch_size, beam_width, beam_length) Each element indices into packed_beams and
      packed_beams[unpacker]==beams.
    """
    device, dtype = beams.device, beams.dtype
    batch_size, beam_width, beam_length = beams.shape

    # Step 1. Calculate segment_index. If you don't understand the comparison, you may want to make
    # sure that you understand docs/pairwise_comparison.py.
    ith_row_has_value_i = torch.arange(beam_width, device=device)[None, :, None]
    pack_mask = prefix_tree == ith_row_has_value_i

    # The length of the second segment of each candidate sequence in each beam. Note that the first
    # segment of the first candidate sequence is empty, so the length of the second segment is the
    # lenght of the first candidate sequence.
    n_selected_per_seq = pack_mask.sum(dim=-1)

    # The location in packed beam where the second segment of each candidate token sequence starts.
    # The first segment of each candidate always starts at position 0 of the packed beam.
    start_per_seq = n_selected_per_seq.cumsum(dim=-1) - n_selected_per_seq

    # Elements in prefix_tree indices candidate sequences, in which, the corresponding token has
    # appeared.  Using prefix_tree to gather from start_per_seq assigns each token the starting
    # position in packed beam of the segment the token belongs to.
    segment_index = torch.gather(start_per_seq, dim=1, index=prefix_tree.view(batch_size, -1)).view(
        batch_size, beam_width, beam_length
    )

    # Step 2. Calculate offsets.  To do so, we need a pairwise matching and a cumsum along rows in
    # the lower diagonal matrix. E.g. for [0, 0, 1, 1], the pariwise matching is
    #
    #         [1, 1, 0, 0],
    #         [1, 1, 0, 0],
    #         [0, 0, 1, 1],
    #         [0, 0, 1, 1],
    #
    # The lower diagonal is
    #
    #         [0, 0, 0, 0],
    #         [1, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 1, 0],
    #
    # Sum over dim=-1, we get the offset
    #
    #         [0, 1, 0, 1]
    #
    match = segment_index[:, :, :, None] == segment_index[:, :, None, :]
    # Only consider lower triangles.
    seq_index = torch.arange(beam_length, dtype=dtype, device=device)
    lower_triangle = seq_index[:, None] > seq_index[None, :]
    offset = (match * lower_triangle[None, None, :, :]).sum(dim=-1)

    return segment_index + offset


def _get_causal_mask_of_packed_beams(
    unpacker: torch.Tensor,
    packed_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Return the causal mask of packed beams (not including the prompt).

      unpacker: (batch_size, beam_width, beam_length) A Mapping of draft candidates index from a
      stacked representation to a flattened and compact representation.

      packed_token_indices: (batch_size, n_selected) A Mapping of draft candidates index from a
      flattened and compact representation to a stacked representation.

    Returns:

      causal_mask_of_packed_beams: (batch_size, n_selected, n_selected) Output a causal mask tensor
      for packed beams with a flattened and compact indexing.

    """

    batch_size, beam_width, beam_length = unpacker.shape
    device = unpacker.device
    n_selected = packed_token_indices.shape[1]

    arange_batch = torch.arange(batch_size, device=device).unsqueeze(dim=1)
    arange_all_candidate_tokens = torch.arange(beam_length * beam_width, device=device)

    packed_token_beams = packed_token_indices // beam_length

    causal_mask_of_packed_beams = torch.full(
        size=(batch_size, n_selected, n_selected),
        fill_value=False,
        device=device,
        dtype=torch.bool,
    )

    # `causal_mask_of_beams` is the flattened causal mask of beams
    beam_blocks = arange_all_candidate_tokens // beam_length
    lower_triangular_mask = (
        arange_all_candidate_tokens.unsqueeze(-1) - arange_all_candidate_tokens.unsqueeze(0) >= 0
    )
    block_diagonal_mask = beam_blocks.unsqueeze(-1) - beam_blocks.unsqueeze(0) == 0
    causal_mask_of_beams = torch.logical_and(lower_triangular_mask, block_diagonal_mask)[
        None, :, :
    ].expand(batch_size, -1, -1)

    # `causal_mask_of_packed_beams` is the compact and flattened causal mask of packed beams
    selected_token_mask = causal_mask_of_beams[arange_batch, packed_token_indices]
    src_idx = (
        packed_token_beams.unsqueeze(dim=-1) * beam_length
        + torch.arange(beam_length, device=device)[None, None, :]
    )
    src_mask = torch.gather(selected_token_mask, dim=2, index=src_idx)
    tgt_idx = torch.gather(
        unpacker,
        dim=1,
        index=packed_token_beams[:, :, None].expand(batch_size, n_selected, beam_length),
    )
    # scatter_ requires index has dtype=int64
    causal_mask_of_packed_beams.scatter_(dim=2, index=tgt_idx, src=src_mask)
    return causal_mask_of_packed_beams


def _get_causal_mask(
    padding_mask: torch.Tensor,
    unpacker: torch.Tensor,
    packed_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Return the full causal mask according to the flattened and compact index.
    Implement without using any for-loop.

      padding_mask: (batch_size, previous_seq_len) The padding mask from the sequence tokens
      containing the prompt tokens and the previouly generated and verified tokens. A zero or false
      element corresponds to a a padding token in the sequence.

      unpacker: (batch_size, beam_width, beam_length) A Mapping of draft candidates index from a
      stacked representation to a flattened and compact representation.

      packed_token_indices: (batch_size, n_selected) A Mapping of draft candidates index from a
      flattened and compact representation to a stacked representation.

    Returns:

      causal_mask: (batch_size, previous_seq_len + n_selected, previous_seq_len + n_selected) The
      causal mask tensor for the sequence and candidates with a flattened and compact indexing.

    Examples:
        Checkout `test_get_causal_mask` in the unit test
        pytest recurrent_drafting/tree_attention_test.py -k test_get_causal_mask

    """
    batch_size, n_selected = packed_token_indices.shape
    previous_seq_len = padding_mask.shape[1]
    # Initialize with a causal mask.
    extended_padding_mask = torch.cat(
        (padding_mask, torch.full((batch_size, n_selected), True, device=padding_mask.device)),
        dim=-1,
    )
    causal_mask = attention.causal_mask(
        extended_padding_mask,
        previous_seq_len + n_selected,
        unpacker.device,
    )
    causal_mask_of_packed_beams = _get_causal_mask_of_packed_beams(
        unpacker,
        packed_token_indices,
    )
    causal_mask[:, previous_seq_len:, previous_seq_len:] = causal_mask_of_packed_beams
    return causal_mask


def pack(
    beams: torch.Tensor, padding_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten each 2D beam in a batch into a vector of candidate tokens that do not share
    prefixes. This reduces the total number of tokens to be verified by the LLM and saves FLOPs.

    Consider a beam of two candidate token sequences [[91,92,93], [91,92,94]].  The corresponding
    packed beam is [91,92,93,94], which has two less tokens than the unpacked beam.

      beams: (batch_size, beam_width, beam_length)

      padding_mask: (batch_size, previous_seq_len) The padding mask of the previously generated
      tokens and the prompt. A zero or false element corresponds to a a padding token in the
      sequence.

    Returns:

      packed_beams: (batch_size, n_tokens_in_prefix_tree)

      unpacker: (batch_size, beam_width, beam_length) Indices into packed_beams used to recover
      packed_beams back to beams.

      causal_mask: (batch_size, n_tokens_in_prefix_tree, previous_seq_len + n_tokens_in_prefix_tree)
      When an LLM verifies packed_beams, it expects this causal mask.

      position_offsets: the distance from each packed token to the end of the processed
      sequence. Base LLMs using RoPE need it to derive the position_id by adding it with the length
      of the processed sequence.

    """
    prefix_tree = _dedup_prefix(beams)
    unpacker = _get_unpacker(beams, prefix_tree)
    packed_beams, packed_token_indices = _pack_beams(beams, prefix_tree)
    beam_length = beams.shape[2]
    position_offsets = packed_token_indices % beam_length
    causal_mask = _get_causal_mask(padding_mask, unpacker, packed_token_indices)
    causal_mask = causal_mask[:, -packed_beams.shape[1] :, :]
    return packed_beams, unpacker, causal_mask, position_offsets


def unpack(
    packed_tensor: torch.Tensor,
    unpacker: torch.Tensor,
) -> torch.Tensor:
    """Sending packed beams as input to the base LLM, we get hidden states and vocabulary logits of
    packed tokens. This function unpack them as if they were computed from the unpacked beams.

    To unpack KV cache of packed beams, please call kv_cache.unpack.

    Consider a beam of two candidate token sequences [[91,92,93], [91,92,94]].  The corresponding
    packed beam would be [91,92,93,94]. The verification produces the following hidden states:

      [ s(91), s(92), s(93), s(94) ]

    and logits

      [ g(91), g(92), g(93), g(94) ]

    After unpacking, we get

      [[ s(91), s(92), s(93) ],
       [ s(91), s(92), s(94) ]]

    and logits

      [[ g(91), g(92), g(93) ],
       [ g(91), g(92), g(94) ]]

      packed_tensor: (batch_size, max_length, last_dim_size)

      unpacker: (batch_size, beam_width, beam_length) Each element indices into packed_tensor.

    Returns:

      unpacked_tensor: (batch_size, beam_width, beam_length, last_dim_size)

    """
    assert len(packed_tensor.shape) == 3
    last_dim_size = packed_tensor.shape[2]
    batch_size, beam_width, beam_length = unpacker.shape
    unpacked_data_indices = unpacker.view(batch_size, beam_width * beam_length, 1).expand(
        -1, -1, last_dim_size
    )
    unpacked_tensor = torch.gather(packed_tensor, 1, unpacked_data_indices).reshape(
        batch_size, beam_width, beam_length, -1
    )
    return unpacked_tensor
