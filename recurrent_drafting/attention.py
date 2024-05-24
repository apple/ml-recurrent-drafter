#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""This file contains functions to create different types
of attention mask and attention bias.
"""

import torch


def causal_mask(padding_mask: torch.Tensor, query_len: int, device: torch.device) -> torch.Tensor:
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
    assert padding_mask.dtype in [torch.int32, torch.int64, torch.bool]
    batch_size, key_value_len = padding_mask.shape
    causal_mask = torch.tril(
        torch.full((key_value_len, key_value_len), fill_value=True, device=device),
    )
    batch_causal_mask = causal_mask[None, :, :].repeat(batch_size, 1, 1)
    batch_causal_mask = batch_causal_mask.masked_fill(padding_mask[:, None, :].eq(0), False)
    assert query_len > 0
    batch_causal_mask = batch_causal_mask[:, -query_len:, :]
    return batch_causal_mask


def set_sdpa_fully_masked_rows(causal_mask: torch.Tensor):
    """Set the row of causal mask to all True if the row is all False, i.e. fully masked out.
    This is to avoid nan from sdpa attention on GPU due to padding tokens
    https://github.com/pytorch/pytorch/issues/110213

    Args:
        causal_mask (batch_size, query_len, key_value_len)

    Returns:
        causal_mask (batch_size, query_len, key_value_len)
    """
    assert causal_mask.dtype in [torch.int32, torch.int64, torch.bool]
    key_value_len = causal_mask.shape[-1]
    row_all_false = causal_mask.eq(0).sum(dim=-1).eq(key_value_len)
    return causal_mask.masked_fill(row_all_false[:, :, None], True)


def bias(causal_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Create attention bias from the attention mask. Because attention weight and
    bias are in log scale, so this function indeed converts 0 to log(0)=-inf and 1
    to log(1)=0.

    Args:
        causal_mask (batch_size, .., query_len, key_value_len)
        dtype: Used to determine the value of -inf.

    Returns:
        attn_bias (batch_size, .., query_len, key_value_len)
    """
    attn_bias = torch.zeros(causal_mask.shape, dtype=dtype, device=causal_mask.device)
    return attn_bias.masked_fill_(causal_mask.logical_not(), torch.finfo(dtype).min)
