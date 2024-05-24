#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Tuple

import torch
import transformers

from . import configuration_drafter

LOG_0 = -50000.0
LOG_1 = 0.0


@dataclass
class BeamShape:
    width: int
    length: int


def register_auto_models() -> None:
    transformers.AutoConfig.register(
        model_type=configuration_drafter.DrafterConfig.model_type,
        config=configuration_drafter.DrafterConfig,
    )
    transformers.AutoModel.register(
        config_class=configuration_drafter.DrafterConfig, model_class=Drafter
    )


class ResBlock(torch.nn.Module):
    def __init__(self, cfg: configuration_drafter.DrafterConfig):
        super().__init__()
        self.linear = torch.nn.Linear(cfg.exit_dim, cfg.exit_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.silu(self.linear(x))


def _gather_beams(x: torch.Tensor, selected_beams: torch.Tensor) -> torch.Tensor:
    """Given the input tensor x of size (batch_size, beam_width, ...), returns
    x[:,selected_beams,...]. As long as the index vector selecte_beams has beam_width elements, the
    returned value has the same size as the input.

    """
    batch_size, beam_width = x.shape[0], x.shape[1]
    batch_indices = torch.reshape(
        torch.arange(batch_size * beam_width, device=x.device) // beam_width,
        (batch_size, beam_width),
    )
    return x[batch_indices, selected_beams]


def _add_decoding_dim(x: torch.Tensor, beam_width: int) -> torch.Tensor:
    """Creates beam_width as second dimension in non-scalar array x and tiles into it."""
    return torch.unsqueeze(x, dim=1).repeat(
        [beam_width if i == 1 else 1 for i in range(x.ndim + 1)]
    )


def maintain_logits(logits: torch.Tensor) -> torch.Tensor:
    """The maintain_logits applies a normalization to the logits tensor to avoid
    all values become -inf and lead to numerical instability.

    The normalization steps are:
    1. Firstly get the max value for each beam;
    2. Subtract the `logits` values by the max value;
    Args:
        logits: (batch_size, beam_width, vocab_size). The output from drafter's lm_head.
    Returns:
        logits: (batch_size, beam_width, vocab_size), the logits after the normalization
    """
    bs, beam_width, vocab_size = logits.shape
    max_logits = torch.max(logits, dim=-1)[0].unsqueeze(-1).expand(bs, beam_width, vocab_size)
    return logits - max_logits


def warp_logits(logits: torch.Tensor, top_k: int = 50, mask_value: float = LOG_0) -> torch.Tensor:
    """warp_logits performs top-k, i.e. restricting to the k highest probability elements.
    Reference in huggingface:
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L470

    Args:
        logits (`torch.FloatTensor`):
            The input logits.
        top_k (`int`, *optional*, default to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        mask_value (`float`, *optional*, defaults to LOG_0):
            All filtered values will be set to this float value.
    Returns:
        logits: (same shape as input logits) The logits after top-k warp transformation
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, mask_value)
    return logits


class Drafter(transformers.PreTrainedModel):
    config_class = configuration_drafter.DrafterConfig

    def __init__(
        self,
        cfg: configuration_drafter.DrafterConfig,
    ):
        super().__init__(cfg)
        self.config = cfg

        # Layer 1 is optional. drafter's hidden_size = llm's hidden_size
        input_dim = 2 * cfg.hidden_size
        if input_dim != cfg.exit_dim:
            self.input_proj = torch.nn.Linear(input_dim, cfg.exit_dim, bias=True)

        # Layer 2 is mandatory.
        self.lm_head = torch.nn.Sequential(
            *([ResBlock(cfg) for _ in range(cfg.num_draft_layers)]),  # residual blocks
            torch.nn.Linear(in_features=cfg.exit_dim, out_features=cfg.vocab_size, bias=False),
        )

        if cfg.rnn:
            self.rnn_u = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
            self.rnn_w = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = maintain_logits(
            self.lm_head(self.input_proj(x) if hasattr(self, "input_proj") else x)
        )
        return warp_logits(logits)

    def beam_search_candidates(
        self,
        prompt_state: torch.Tensor,
        init_token: torch.Tensor,
        embeddings: torch.nn.Embedding,
        beam_shape: BeamShape = BeamShape(10, 4),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample draft tokens from Drafter.

        Args:
            prompt_state: (batch_size, hidden_size). The hidden state of the last token in prompt.
            init_token: (batch_size), int. The token sampled from the vanilla LM head of the LLM.
            embeddings: (vocab_size, hidden_size). The embedding layer of the LLM.

        Returns:
            beams: (batch, beam_width, beam_length), int.  init_token and draft tokens per beam.
            log_probs: (batch, beam_width, beam_length-1, vocab_size), float.  Log probs of draft
            tokens.
        """
        assert init_token.dtype == torch.long
        assert prompt_state.ndim == 2
        assert init_token.ndim == 1
        assert beam_shape.length > 1

        batch_size = prompt_state.shape[0]
        hidden_size = prompt_state.shape[1]
        vocab_size = embeddings.num_embeddings
        device = prompt_state.device

        # (bs,beam_width) The log_P(beam). Because all beams start with init_token sampled from the
        # LLM's vanilla LM head, log_p_beam should be all log(1)=0.0. However, this would make the
        # following loop growing all beams into the same look. So, we initialize only the first beam
        # with log(1) and all rest with log(0). This will make the first loop step grows tokens on
        # and only on top of the first beam. At the end of the first loop step, log_p_beam will be
        # replaced with the result from topk that distributes the first beam_width of selected
        # tokens to all beams. So the rest loop steps can take log_p_beam as log_P(beam) and keep
        # grow beams.
        log_p_beam = torch.tile(
            torch.tensor([LOG_1] + [LOG_0] * (beam_shape.width - 1), device=device), (batch_size, 1)
        )
        # (bs,beam_width,hidden_size) prompt_state replicated for all beams.
        # Constant in this function.
        context = _add_decoding_dim(prompt_state, beam_width=beam_shape.width)
        # (bs,beam_width,1) All beams start with init_token, sampled from the vanilla LM head.
        beams = _add_decoding_dim(init_token.unsqueeze(dim=-1), beam_width=beam_shape.width)
        # (bs,beam_width,hidden_size) The RNN state of each beam.
        state = torch.zeros(
            batch_size,
            beam_shape.width,
            hidden_size,
            device=device,
            dtype=prompt_state.dtype,
        )
        # (bs,beam_width,[0,beam_length-1),vocab_size) P(token | beam, candidate_token).
        log_p_token_in_beam = (
            torch.Tensor().reshape(batch_size, beam_shape.width, 0, vocab_size).to(device)
        )

        for _ in range(beam_shape.length - 1):
            # Updates the RNN state of each beam given the input of the previous draft token.
            state = (
                torch.nn.functional.silu(self.rnn_w(embeddings(beams[..., -1])) + self.rnn_u(state))
                if self.config.rnn
                else embeddings(beams[..., -1]) + state
            )

            # (bs,beam_width,vocab_size). For each beam, predicts the next token by computing
            # log_P(new_token) using the drafter LM head, which takes context and state as input.
            log_p_new_token = self.compute_logits(torch.cat([context, state], dim=-1)).log_softmax(
                dim=-1
            )

            # (bs,beam_width,vocab_size). log_P(new_token, beam) = log_P(new_token) + log_P(beam)
            log_p_beam_new_token = log_p_new_token + torch.unsqueeze(log_p_beam, dim=2)

            # reshape so topk searches for top beam_width tokens among the vocab_size candidate
            # tokens grown out from all beam_width beams.
            tokens_times_beams = log_p_beam_new_token.reshape(
                (batch_size, beam_shape.width * vocab_size)
            )
            log_p_beam, topk_indices = torch.topk(tokens_times_beams, k=beam_shape.width, dim=-1)
            top_token_ids = topk_indices % vocab_size
            top_beam_indices = topk_indices // vocab_size

            # Select the top beams and grow them with the top tokens.
            beams = torch.cat(
                [_gather_beams(beams, top_beam_indices), top_token_ids.unsqueeze(dim=-1)], dim=-1
            )

            state = _gather_beams(state, top_beam_indices)

            log_p_token_in_beam = torch.cat(
                [
                    _gather_beams(log_p_token_in_beam, top_beam_indices),
                    _gather_beams(log_p_new_token, top_beam_indices).unsqueeze(dim=2),
                ],
                dim=2,
            )

        return beams, log_p_token_in_beam

    def assert_valid(self) -> None:
        assert isinstance(self.input_proj, torch.nn.Linear)
        assert (
            self.input_proj is not None
            if self.config.hidden_size != self.config.exit_dim
            else self.input_proj is None
        )
        assert isinstance(self.lm_head, torch.nn.Sequential)
        assert self.lm_head is not None
        assert len(self.lm_head) == self.config.num_draft_layers + 1
