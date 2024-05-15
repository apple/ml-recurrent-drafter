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
        _, _, batch_size, n_heads, _, head_dim = self._cache.shape
        return (batch_size, n_heads, self.length, head_dim)

    def cat(self, new_kv: mx.array) -> mx.array:
        _, _, batch_size, n_heads, max_length, head_dim = self._cache.shape
        new_length = new_kv.shape[2]
        assert new_kv.shape == (batch_size, n_heads, new_length, head_dim)
        assert self.length + new_length <= max_length
        self._cache[self._layer, self._kv, :, :, self.length : self.length + new_length, :] = new_kv
        self.length += new_length
        return self._cache[self._layer, self._kv, :, :, 0 : self.length, :]

    def slice(self, prev: int) -> mx.array:
        """Returns the current view, but skipping over the first prev tokens."""
        return self._cache[self._layer, self._kv, :, :, prev : self.length, :]


class Cache:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,  # config.hidden_size // config.num_attention_heads
        dtype: mx.Dtype,
        device: mx.Device,
    ) -> None:
        self._cache = mx.zeros(
            [n_layers, 2, batch_size, n_heads, max_length, head_dim],  # 2 for key and value
            dtype=dtype,
            stream=device,
        )
        self.sliced = tuple(
            (View(self._cache, layer, 0), View(self._cache, layer, 1)) for layer in range(n_layers)
        )
