#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Generator

import torch

from . import attention, kv_cache, recurrent_drafting


@torch.inference_mode()
def generate(
    model: recurrent_drafting.RecurrentDrafting,
    input_ids: torch.Tensor,
    max_length: int,
    special_tokens: recurrent_drafting.SpecialTokens,
) -> Generator[torch.Tensor, None, None]:
    cache = kv_cache.Cache(
        batch_size=input_ids.shape[0],
        max_length=max_length,
        n_layers=model.llm.config.num_hidden_layers,
        n_heads=model.llm_n_kv_heads(),
        head_dim=model.llm.config.hidden_size // model.llm.config.num_attention_heads,
        dtype=model.llm.dtype,
        device=input_ids.device,
    )
    context_len = input_ids.shape[1]
    while input_ids.shape[1] < max_length:
        padding_mask = input_ids != special_tokens.pad
        cur_input_ids = input_ids[:, -1:] if input_ids.shape[1] > context_len else input_ids
        attention_mask = attention.causal_mask(
            padding_mask, cur_input_ids.shape[1], input_ids.device
        )
        past_kv_len = cache.sliced[0][0].length
        cur_kv_len = past_kv_len + cur_input_ids.shape[1]
        position_ids = torch.arange(
            past_kv_len, cur_kv_len, dtype=torch.long, device=input_ids.device
        ).expand(cur_input_ids.shape)
        output = model.llm(
            input_ids=cur_input_ids,
            past_key_values=cache.sliced,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = output.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        yield input_ids
        if (
            (input_ids[:, context_len:] == special_tokens.eos).sum(dim=-1) > 0
        ).sum() == input_ids.shape[0]:
            break
