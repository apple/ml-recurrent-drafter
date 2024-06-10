# Copyright © 2024 Apple Inc.
from typing import Tuple

import mlx.core as mx


class View:
    """A slice of the KV cache tensor for key or value cache for a transformer decoding layer."""

    def __init__(self, cache: mx.array, layer: int, kv: int):
        self._cache = cache
        self._layer = layer
        self._kv = kv  # 0 for key, 1 for value
        self.length = 0

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        batch_size, n_heads, _, head_dim = self._cache.shape
        return (batch_size, n_heads, self.length, head_dim)

    def cat(self, new_kv: mx.array) -> mx.array:
        batch_size, n_heads, max_length, head_dim = self._cache.shape
        new_length = new_kv.shape[2]
        assert new_kv.shape == (batch_size, n_heads, new_length, head_dim)
        assert self.length + new_length <= max_length
        self._cache[:, :, self.length : self.length + new_length, :] = new_kv
        self.length += new_length
        return self._cache[:, :, 0 : self.length, :]

    def slice(self, prev: int) -> mx.array:
        """Returns the current view, but skipping over the first prev tokens."""
        return self._cache[:, :, prev : self.length, :]


class Cache:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,  # config.hidden_size // config.num_attention_heads
        dtype: mx.Dtype,
    ) -> None:
        _sliced = []
        for layer in range(n_layers):
            key = mx.zeros(
                [batch_size, n_heads, max_length, head_dim],
                dtype=dtype,
            )
            val = mx.zeros(
                [batch_size, n_heads, max_length, head_dim],
                dtype=dtype,
            )
            _sliced.append((View(key, layer, 0), View(val, layer, 1)))
        self.sliced = tuple(_sliced)
