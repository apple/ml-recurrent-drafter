#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM

from recurrent_drafting.modeling_drafter import Drafter


class ReDrafter(nn.Module):
    def __init__(
        self,
        llm: transformers.PreTrainedModel,
        drafter: Drafter,
    ):
        super().__init__()
        self.llm = llm
        self.drafter = drafter

    @classmethod
    def from_pretrained(
        cls,
        llm_name_or_path,
        drafter_name_or_path,
        **kwargs,
    ):
        llm = LlamaForCausalLM.from_pretrained(llm_name_or_path, **kwargs)
        drafter = Drafter.from_pretrained(drafter_name_or_path, **kwargs)
        model = cls(llm, drafter)
        return model

    def base_forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    ):
        """Forward pass of the base model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional):
                Tuple containing past key and value states for attention.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            Predictions from the base model's LM head.
        """
        with torch.inference_mode():
            outputs = self.llm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            base_forward = self.llm.lm_head(outputs[0])
        return base_forward

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        position_ids=None,
        next_n=1,
    ):
        """Forward pass of the ReDrafter.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional):
                Tuple containing past key and value states for attention.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from ReDrafter.
        """
        with torch.inference_mode():
            outputs = self.llm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        # Clone the output hidden states
        hidden_states = outputs[0].clone()  # [batch_size, seq_len, hidden_dim]
        drafter_logits = []
        input_embs = self.llm.model.embed_tokens(input_ids)
        h, cumsum_input_embs = hidden_states, torch.zeros_like(
            input_embs, dtype=input_embs.dtype, device=input_embs.device
        )
        # Drafter forward function
        # ht: hidden_states at position t
        # et: input_embs at position t
        # [lt]_i: drafter_logits at iteration i predicting token t

        # Iteration 1:
        # [h0,e1] [h1,e2] ... [h_{n-2},e_{n-1}] [h_{n-1},e0]
        #                 V
        #            drafter head
        #                 V
        # [l2]_1  [l3]_1  ... [ln]_1            [l_{n+1}]_1  --> drafter_logits[0]
        #
        # Iteration 2:
        # [h0,e1+e2] [h1,e2+e3] ... [h_{n-2},e_{n-1}+e0] [h_{n-1},e0+e1]
        #                 V
        #            drafter head
        #                 V
        # [l3]_2     [l4]_2     ... [l_{n+1}]_2          [l_{n+3}]_2  --> drafter_logits[1]
        # ...

        # Stack drafter_logits as the return value.
        # E.g. when next_n = 5,
        # [[[l2]_1, [l3]_1, ..., [l_{n+1}]_1],
        #  [[l3]_2, [l4]_2, ..., [l_{n+2}]_2],
        #  ...
        #  [[l6]_5, [l7]_5, ..., [l_{n+5}]_5]],
        for _ in range(next_n):
            input_embs = torch.roll(input_embs, -1, dims=1)
            if self.drafter.config.rnn:
                # s = f(U * s + W * w + b).
                o = self.drafter.rnn_u(cumsum_input_embs)
                cumsum_input_embs = nn.SiLU()(o + self.drafter.rnn_w(input_embs))
            else:
                cumsum_input_embs += input_embs
            h = torch.cat((hidden_states, cumsum_input_embs), -1)
            drafter_logits.append(self.drafter.lm_head(h))
        return torch.stack(drafter_logits, dim=0)
        # [next_n, batch_size, seq_len, vocab_size]
