#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Tuple

import torch


class View:
    """A slice of the KV cache tensor for key or value cache for a transformer decoding layer."""

    def __init__(self, cache: torch.Tensor, layer: int, kv: int):
        self._cache = cache
        self._layer = layer
        self._kv = kv  # 0 for key, 1 for value
        self.length = 0

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        _, _, batch_size, n_heads, _, head_dim = self._cache.shape
        return (batch_size, n_heads, self.length, head_dim)

    def cat(self, new_kv: torch.Tensor) -> torch.Tensor:
        _, _, batch_size, n_heads, max_length, head_dim = self._cache.shape
        new_length = new_kv.shape[2]
        assert new_kv.shape == (batch_size, n_heads, new_length, head_dim)
        assert self.length + new_length <= max_length
        self._cache[self._layer, self._kv, :, :, self.length : self.length + new_length, :] = new_kv
        self.length += new_length
        return self._cache[self._layer, self._kv, :, :, 0 : self.length, :]

    def slice(self, prev: int) -> torch.Tensor:
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
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._cache = torch.zeros(
            [n_layers, 2, batch_size, n_heads, max_length, head_dim],  # 2 for key and value
            dtype=dtype,
            device=device,
        )
        self.sliced = tuple(
            (View(self._cache, layer, 0), View(self._cache, layer, 1)) for layer in range(n_layers)
        )

    def unpack(
        self,
        prompt_length: int,
        tree_attn_index: torch.Tensor,
    ) -> None:
        """Unpack KV Cache given tree attention before verification draft tokens.

        Assume the the draft model outputs a beam of two candidate token sequences: [[11,22,33],
        [11,22,44]], the tree attention will pack them into: [11,22,33,44] with tree_attn_index:
        [[0,1,2], [0,1,3]].

        After verifying [11,22,33,44] using the base model, the kv_cache of each layer would look
        like the following:

        [prompt, k(11), k(22), k(33), k(44)]
        [prompt, v(11), v(22), v(33), v(44)]

        After this function call, kv_cache will be

        [prompt, k(11), k(22), k(33), k(11), k(22), k(44)]
        [prompt, v(11), v(22), v(33), v(11), v(22), v(44)]

        Args:
        tree_attn_index: (batch_size, beam_width, beam_length+1)
            A Mapping of draft candidates index from a stacked representation to a
            flattened and compact representation.

        """
        batch_size = tree_attn_index.shape[0]
        beam_width, beam_length = tree_attn_index.shape[1], tree_attn_index.shape[2] - 1
        n_candidate_tokens = beam_width * (beam_length + 1)
        n_layers, _, _, n_heads, _, hidden_size = self._cache.shape
        key_values_data_indices = (
            prompt_length + tree_attn_index.view(1, 1, batch_size, 1, n_candidate_tokens, 1)
        ).expand(n_layers, 2, -1, n_heads, -1, hidden_size)
        self._cache[:, :, :, :, prompt_length : prompt_length + n_candidate_tokens, :] = (
            torch.clone(torch.gather(self._cache, 4, key_values_data_indices))
        )
        for layer in range(len(self.sliced)):
            for kv in range(2):
                self.sliced[layer][kv].length = prompt_length + n_candidate_tokens
