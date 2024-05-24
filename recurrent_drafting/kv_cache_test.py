#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch

from . import kv_cache


def test_cache() -> None:
    batch_size, max_length, n_layers, n_heads, head_dim, kv = 7, 11, 3, 5, 2, 2
    _dtype = torch.float32
    _device = torch.device("cpu")

    cache = kv_cache.Cache(batch_size, max_length, n_layers, n_heads, head_dim, _dtype, _device)
    assert cache._cache.shape == (n_layers, 2, batch_size, n_heads, max_length, head_dim)
    assert cache._cache.dtype == _dtype
    assert len(cache.sliced) == n_layers
    assert all(len(cache.sliced[i]) == kv for i in range(n_layers))
    assert all(
        cache.sliced[i][j].shape == (batch_size, n_heads, 0, head_dim)
        for i in range(n_layers)
        for j in range(kv)
    )

    new_length = 2
    new_kv = torch.ones([batch_size, n_heads, new_length, head_dim])
    assert torch.all(cache.sliced[0][0].cat(new_kv) == new_kv)
    assert cache.sliced[0][0].length == new_length

    concated = torch.cat([new_kv, new_kv], axis=2)  # type: ignore
    assert torch.all(cache.sliced[0][0].cat(new_kv) == concated)
    assert cache.sliced[0][0].length == new_length * 2

    assert torch.all(cache.sliced[0][0].slice(2) == new_kv)


def test_unpack():
    """This test case follows the example in the docstring of Cache.unpack."""
    # beam_width = 2
    # beam_len = 3
    # total_candidate_tokens = beam_width * beam_len
    # unique_candidate_tokens = 4
    prompt_length = 4
    batch_size = 2
    max_length = 10  # prompt_len + (beam_width * beam_len)
    n_layers = 3
    n_heads = 2  # the kv value for a token is in shape (n_heads, head_dim)
    head_dim = 8
    _dtype = torch.float32
    _device = torch.device("cpu")
    kvcache = kv_cache.Cache(batch_size, max_length, n_layers, n_heads, head_dim, _dtype, _device)

    # Initialize the KV cache as in the docstring of Cache.unpack.
    def kv_value(kv: int, token_id: int) -> float:
        """Referring to the example of Cache.unpack, k{j}=1.{j}, v=2.{j}"""
        return (kv + 1) + (token_id / 10.0)

    kv_prompt = 0.1

    for layer in range(n_layers):
        for kv in range(2):
            # [prompt, k1, k2, k3, k4]
            # [prompt, v1, v2, v3, v4]
            candidate_kv = torch.tensor(
                [kv_prompt] * prompt_length + [kv_value(kv, j) for j in range(4)]
            )
            kvcache.sliced[layer][kv].cat(
                candidate_kv[None, None, :, None].expand(batch_size, n_heads, -1, head_dim)
            )

    tree_attn_idx = torch.tensor([[[0, 1, 2], [0, 1, 3]]])  # (batch_size, beam_width, beam_length)
    kvcache.unpack(prompt_length, tree_attn_idx)

    for layer in range(n_layers):
        for kv in range(2):
            # [prompt, k(11), k(22), k(33), k(11), k(22), k(44)]
            # [prompt, v(11), v(22), v(33), v(11), v(22), v(44)]
            candidate_kv = torch.tensor(
                [kv_prompt] * prompt_length
                + [
                    kv_value(kv, 0),
                    kv_value(kv, 1),
                    kv_value(kv, 2),
                    kv_value(kv, 0),
                    kv_value(kv, 1),
                    kv_value(kv, 3),
                ]
            )
            assert torch.all(
                kvcache.sliced[layer][kv].slice(0)
                == candidate_kv[None, None, :, None].expand(batch_size, n_heads, -1, head_dim)
            )
