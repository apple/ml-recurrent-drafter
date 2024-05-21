# Copyright © 2024 Apple Inc.
"""A variant of
https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py
with pre-allocated KV cache."""
import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from . import attention, kv_cache


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # type: ignore
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # type: ignore
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]  # type: ignore
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,  # required by rotary embedding
        mask: mx.array,
        cache: Tuple[kv_cache.View, kv_cache.View],
    ) -> mx.array:
        # Only support batch_size = 1
        assert position_ids.shape[0] == x.shape[0] == 1
        assert position_ids.shape[1] == x.shape[1]
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Assume no tree attention or any other input compression algorithm is applied
        beam_length = (position_ids[0][-1] - position_ids[0][0] + 1).item()
        beam_width = L // beam_length
        assert beam_width * beam_length == L

        # Prepare the queries, keys and values for RoPE computation
        queries = queries.reshape(B * beam_width, beam_length, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B * beam_width, beam_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B * beam_width, beam_length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        # Compute RoPE
        key_cache, value_cache = cache
        queries = self.rope(queries, offset=key_cache.shape[2])
        keys = self.rope(keys, offset=value_cache.shape[2])

        # Prepare the queries, keys and values for the attention computation
        queries = (
            queries.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        )
        keys = keys.transpose(0, 2, 1, 3).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = (
            values.transpose(0, 2, 1, 3).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        )

        keys = key_cache.cat(keys)  # in-place
        values = value_cache.cat(values)  # in-place

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,  # required by rotary embedding
        mask: mx.array,
        cache: Tuple[kv_cache.View, kv_cache.View],
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), position_ids, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        position_ids: mx.array,  # required by rotary embedding
        mask: mx.array,
        cache: Tuple[Tuple[kv_cache.View, kv_cache.View], ...],
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        bias = attention.bias(mask, h.dtype)

        for layer, c in zip(self.layers, cache):
            h = layer(h, position_ids, bias, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        position_ids: mx.array,  # required by rotary embedding
        mask: mx.array,
        cache: Tuple[Tuple[kv_cache.View, kv_cache.View], ...],
    ) -> mx.array:
        assert inputs.shape[0] == 1  # Only supports batch_size=1 for now.
        out = self.model(inputs, position_ids, mask, cache)
        return self.lm_head(out)

    @property
    def layers(self) -> List[TransformerBlock]:
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads  # type: ignore


def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    # Remove unused precomputed rotary freqs
    return {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}


def load_model(model_path) -> Model:
    model_path = Path(model_path)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())

    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(sanitize(weights).items()))
    return model
