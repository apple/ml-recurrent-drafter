# Copyright © 2024 Apple Inc.
import mlx.core as mx

from . import kv_cache


def test_cache() -> None:
    batch_size, max_length, n_layers, n_heads, head_dim, kv = 7, 11, 3, 5, 2, 2
    _dtype = mx.float32

    cache = kv_cache.Cache(batch_size, max_length, n_layers, n_heads, head_dim, _dtype)
    assert len(cache.sliced) == n_layers
    for k_cache, v_cache in cache.sliced:
        assert k_cache._cache.shape == (batch_size, n_heads, max_length, head_dim)
        assert v_cache._cache.shape == (batch_size, n_heads, max_length, head_dim)
        assert k_cache._cache.dtype == _dtype
        assert v_cache._cache.dtype == _dtype
    assert all(
        cache.sliced[i][j].shape == (batch_size, n_heads, 0, head_dim)
        for i in range(n_layers)
        for j in range(kv)
    )

    new_length = 2
    new_kv = mx.ones([batch_size, n_heads, new_length, head_dim])
    assert mx.all(cache.sliced[0][0].cat(new_kv) == new_kv)
    assert cache.sliced[0][0].length == new_length

    concated = mx.concatenate([new_kv, new_kv], axis=2)  # type: ignore
    assert mx.all(cache.sliced[0][0].cat(new_kv) == concated)
    assert cache.sliced[0][0].length == new_length * 2

    assert mx.all(cache.sliced[0][0].slice(2) == new_kv)
