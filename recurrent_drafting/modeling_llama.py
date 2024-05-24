#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""A variant of
https://github.com/huggingface/transformers/blob/v4.35-release/src/transformers/models/llama/
with pre-allocated KV cache."""

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig

from . import attention, kv_cache


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same
        # calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)

    def forward(self, device: torch.device, dtype: torch.dtype, seq_len: int = 0):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype).to(device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype).to(device),
        )


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def apply_rotary_pos_emb(
    q: torch.Tensor,  # [batch_size, heads, seq_len, head_dim]
    k: torch.Tensor,
    cos: torch.Tensor,  # [batch_size, seq_len, head_dim]
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """GPT NeoX style: rotates [repeat] half the hidden dims of the input.
        To match sin: [θ0,θ1,θ2,...,θd/2-1......θ0,θ1,θ2,...,θd/2-1]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_kv_groups = self.num_heads // self.n_kv_heads
        assert self.head_dim * self.num_heads == self.hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,  # required by rotary embedding
        past_key_value: Tuple[kv_cache.View, kv_cache.View],
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(
            value_states.device,
            value_states.dtype,
            key_states.shape[-2] + past_key_value[0].shape[-2],
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        past_key, past_value = past_key_value
        key_states = past_key.cat(key_states)  # in-place
        value_states = past_value.cat(value_states)  # in-place

        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.n_kv_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.n_kv_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_bias,
        )
        # attn_output shape before reshaping: [bsz, num_heads, q_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Tuple[kv_cache.View, kv_cache.View],
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_bias=attention_bias,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states += attn_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states += mlp_output
        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Tuple[Tuple[kv_cache.View, kv_cache.View], ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        attention_bias = attention.bias(attention_mask.unsqueeze(dim=1), dtype=hidden_states.dtype)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_bias=attention_bias,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
            )
        return self.norm(hidden_states)


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,  # (batch_size, q_len, kv_len)
        position_ids: torch.LongTensor,  # input_ids.shape
        past_key_values: Tuple[Tuple[kv_cache.View, kv_cache.View], ...],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Avoid nan from sdpa attention on GPU due to padding tokens
        # See https://github.com/pytorch/pytorch/issues/110213
        if attention_mask.device.type == "cuda":
            attention_mask = attention.set_sdpa_fully_masked_rows(attention_mask)
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states).float()
        return CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
        )
