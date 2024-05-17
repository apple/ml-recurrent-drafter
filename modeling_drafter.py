# Copyright © 2024 Apple Inc.
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
from mlx import nn

LOG_0 = -50000.0
LOG_1 = 0.0


@dataclass
class ModelArgs:
    hidden_size: int
    vocab_size: int
    exit_dim: int
    num_draft_layers: int
    model_type: str = "recurrent_drafting_drafter"
    rnn: bool = False


@dataclass
class BeamShape:
    width: int
    length: int


class ResBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.linear = nn.Linear(args.exit_dim, args.exit_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return x + nn.silu(self.linear(x))


def _gather_beams(x: mx.array, selected_beams: mx.array) -> mx.array:
    """Given the input tensor x of size (batch_size, beam_width, ...), returns
    x[:,selected_beams,...]. As long as the index vector selecte_beams has beam_width elements, the
    returned value has the same size as the input.

    """
    batch_size, beam_width = x.shape[0], x.shape[1]
    batch_indices = (mx.arange(batch_size * beam_width) // beam_width).reshape(
        (batch_size, beam_width)
    )
    return x[batch_indices, selected_beams]


def _add_decoding_dim(x: mx.array, beam_width: int) -> mx.array:
    """Creates beam_width as second dimension in non-scalar array x and tiles into it."""
    return mx.repeat(x[:, None], axis=1, repeats=beam_width)


def maintain_logits(logits: mx.array) -> mx.array:
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
    bs, _, vocab_size = logits.shape
    max_logits = mx.repeat(mx.max(logits, axis=-1)[..., None], axis=-1, repeats=vocab_size)
    return logits - max_logits


def warp_logits(logits: mx.array, top_k: int = 50, mask_value: float = LOG_0) -> mx.array:
    """warp_logits performs top-k, i.e. restricting to the k highest probability elements.
    Reference in huggingface:
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L470

    Args:
        logits (`mx.array`):
            The input logits.
        top_k (`int`, *optional*, default to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        mask_value (`float`, *optional*, defaults to LOG_0):
            All filtered values will be set to this float value.
    Returns:
        logits: (same shape as input logits) The logits after top-k warp transformation
    """
    top_k = min(top_k, logits.shape[-1])  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < mx.topk(logits, top_k).min(axis=-1)[..., None]
    logits = mx.logical_not(indices_to_remove) * logits + indices_to_remove * mask_value
    return logits


class Drafter(nn.Module):

    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()
        self.args = args

        # Layer 1 is optional. drafter's hidden_size = llm's hidden_size
        input_dim = 2 * args.hidden_size
        if input_dim != args.exit_dim:
            self.input_proj = nn.Linear(input_dim, args.exit_dim, bias=True)

        # Layer 2 is mandatory.
        self.lm_head = [
            *([ResBlock(args) for _ in range(args.num_draft_layers)]),  # residual blocks
            nn.Linear(input_dims=args.exit_dim, output_dims=args.vocab_size, bias=False),
        ]

        if args.rnn:
            self.rnn_u = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
            self.rnn_w = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def compute_logits(self, x: mx.array) -> mx.array:
        x = self.input_proj(x) if hasattr(self, "input_proj") else x
        for layer in self.lm_head:
            x = layer(x)
        x = maintain_logits(x)
        return warp_logits(x)

    def beam_search_candidates(
        self,
        prompt_state: mx.array,
        init_token: mx.array,
        embeddings: nn.Embedding,
        beam_shape: BeamShape = BeamShape(10, 4),
    ) -> Tuple[mx.array, mx.array]:
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
        assert prompt_state.ndim == 2
        assert init_token.ndim == 1
        assert beam_shape.length > 1

        batch_size = prompt_state.shape[0]
        hidden_size = prompt_state.shape[1]
        vocab_size = embeddings.weight.shape[0]

        # (bs,beam_width) The log_P(beam). Because all beams start with init_token sampled from the
        # LLM's vanilla LM head, log_p_beam should be all log(1)=0.0. However, this would make the
        # following loop growing all beams into the same look. So, we initialize only the first beam
        # with log(1) and all rest with log(0). This will make the first loop step grows tokens on
        # and only on top of the first beam. At the end of the first loop step, log_p_beam will be
        # replaced with the result from topk that distributes the first beam_width of selected
        # tokens to all beams. So the rest loop steps can take log_p_beam as log_P(beam) and keep
        # grow beams.
        log_p_beam = mx.tile(mx.array([LOG_1] + [LOG_0] * (beam_shape.width - 1)), (batch_size, 1))
        # (bs,beam_width,hidden_size) prompt_state replicated for all beams.
        # Constant in this function.
        context = _add_decoding_dim(prompt_state, beam_width=beam_shape.width)
        # (bs,beam_width,1) All beams start with init_token, sampled from the vanilla LM head.
        beams = _add_decoding_dim(mx.expand_dims(init_token, axis=-1), beam_width=beam_shape.width)
        # (bs,beam_width,hidden_size) The RNN state of each beam.
        state = mx.zeros(
            shape=(batch_size, beam_shape.width, hidden_size),
            dtype=prompt_state.dtype,
        )
        # (bs,beam_width,[0,beam_length-1),vocab_size) P(token | beam, candidate_token).
        log_p_token_in_beam = mx.zeros(0).reshape(batch_size, beam_shape.width, 0, vocab_size)

        for _ in range(beam_shape.length - 1):
            # Updates the RNN state of each beam given the input of the previous draft token.
            state = (
                nn.silu(self.rnn_w(embeddings(beams[..., -1])) + self.rnn_u(state))
                if self.args.rnn
                else embeddings(beams[..., -1]) + state
            )

            # (bs,beam_width,vocab_size). For each beam, predicts the next token by computing
            # log_P(new_token) using the drafter LM head, which takes context and state as input.
            log_p_new_token = nn.log_softmax(
                self.compute_logits(mx.concatenate([context, state], axis=-1)), axis=-1
            )

            # (bs,beam_width,vocab_size). log_P(new_token, beam) = log_P(new_token) + log_P(beam)
            log_p_beam_new_token = log_p_new_token + mx.expand_dims(log_p_beam, axis=2)

            # reshape so topk searches for top beam_width tokens among the vocab_size candidate
            # tokens grown out from all beam_width beams.
            tokens_times_beams = log_p_beam_new_token.reshape(
                (batch_size, beam_shape.width * vocab_size)
            )
            kth = tokens_times_beams.shape[1] - beam_shape.width
            topk_indices = mx.argpartition(tokens_times_beams, kth=kth, axis=-1)[:, kth:]
            log_p_beam = mx.take_along_axis(tokens_times_beams, topk_indices, axis=-1)
            top_token_ids = topk_indices % vocab_size
            top_beam_indices = topk_indices // vocab_size

            # Select the top beams and grow them with the top tokens.
            beams = mx.concatenate(
                [_gather_beams(beams, top_beam_indices), mx.expand_dims(top_token_ids, axis=-1)],
                axis=-1,
            )

            state = _gather_beams(state, top_beam_indices)

            log_p_token_in_beam = mx.concatenate(
                [
                    _gather_beams(log_p_token_in_beam, top_beam_indices),
                    mx.expand_dims(_gather_beams(log_p_new_token, top_beam_indices), axis=2),
                ],
                axis=2,
            )

        return beams, log_p_token_in_beam

    def assert_valid(self) -> None:
        assert isinstance(self.input_proj, nn.Linear)
        assert (
            self.input_proj is not None
            if self.args.hidden_size != self.args.exit_dim
            else self.input_proj is None
        )
        assert self.lm_head is not None
        assert len(self.lm_head) == self.args.num_draft_layers + 1
