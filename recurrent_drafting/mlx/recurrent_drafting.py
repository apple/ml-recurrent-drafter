# Copyright © 2024 Apple Inc.
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import mlx.core as mx
import mlx.nn

from . import attention, kv_cache, modeling_drafter, modeling_llama, tree_attention


@dataclass
class SamplingArgs:
    temperature: float
    greedy: bool  # greedy_search=True eliminates randomness


@dataclass
class SpecialTokens:
    pad: int
    eos: Optional[int] = 1


def _select_one_per_row(x: mx.array, batch_index: mx.array) -> mx.array:
    """x: (batch_size, seq_len, ....). batch_index: (batch_size,),int.
    Return (batch_size, 1, ...)."""
    return x[mx.arange(x.shape[0]), batch_index]


def _greedy_choose_from_candidates(
    beams_by_drafter: mx.array,
    beams_by_llm: mx.array,
) -> Tuple[mx.array, mx.array]:  # n_tokens_in_seq  # seq_in_beam
    """Choose a candidate token sequence from each of the batch of beams by drafter, and choose
    the first n tokens from that sequence. With greedy_search enabled, the drafter's longest
    candidate token sequence exact-matching the candidate token sequence by llm would be chosen.

      beams_by_drafter: (batch_size, beam_width, beam_length) Contains the init token generated
      by llm.

      beams_by_llm: (batch_size, beam_width, beam_length) Doesn't include the init token but
      contains an additional token sampled after beams_by_drafter.

      For example, for a candidate token sequence (beam_length=5)
      In beams_by_drafter:  xxxxx    # The first token is the init token generated by llm.
      In beams_by_llm:       xxxxx   # The last token is generated after beams_by_drafter by llm.

    Returns:

      n_tokens_in_seq: (batch_size) The number of candidate tokens chosen in the chosen seq_in_beam.

      seq_in_beam: (batch_size) The index of the chosen sequence in each of the batch of beams.
    """
    beams_by_drafter_without_init = beams_by_drafter[:, :, 1:]
    beams_by_llm_without_last = beams_by_llm[:, :, :-1]
    compare = (beams_by_drafter_without_init == beams_by_llm_without_last).astype(mx.int32)
    n_tokens_in_seqs = mx.sum(mx.cumprod(compare, axis=-1), axis=-1)
    kth = n_tokens_in_seqs.shape[1] - 1
    seq_in_beam = mx.argpartition(n_tokens_in_seqs, kth=kth, axis=-1)[:, kth:]
    n_tokens_in_seq = mx.take_along_axis(n_tokens_in_seqs, seq_in_beam, axis=-1)
    return n_tokens_in_seq.squeeze(axis=-1), seq_in_beam.squeeze(axis=-1)


def _choose_from_candidates(
    beams: mx.array,
    log_probs_by_llm: mx.array,
    log_probs_by_drafter: mx.array,
) -> Tuple[mx.array, mx.array]:  # n_tokens_in_seq  # seq_in_beam
    """Choose a candidate token sequence from each of the batch of beams, and choose the first n
    tokens from that sequence. For details, please refer to docs/speculative_sampling.md.

      beams: (batch_size, beam_width, beam_length)

      log_probs_by_llm: (batch_size, beam_width, beam_length, vocab_size) Log probability
      distribution over the vocabulary for the initial draft token sampled from the vanilla LM head
      of the LLM and each candidate token from the drafter. For each input token, the LLM outputs
      the probability distribution of the next token, so we have
      log_probs_by_llm[:,:,k,:]=P(beams[k+1]).

      log_probs_by_drafter: (batch_size, beam_width, beam_length-1, vocab_size) Log probability
      distribution over the vocabulary for each drafted candidate token in the beams.  The beam
      search algorithm outputs the token distribution for each drafted token, so we have
      log_probs_by_llm[:,:,k,:]=Q(beams[k]).  Please be aware of the shifting of the index k
      compared with log_probs_by_llm.

    Returns:

      n_tokens_in_seq: (batch_size) The number of candidate tokens chosen in the chosen seq_in_beam.

      seq_in_beam: (batch_size) The index of the chosen sequence in each of the batch of beams.

    """
    assert beams.dtype in [mx.int32, mx.int64]

    # Due to the index shift between log_probs_by_llm and _by_drafter, removing the last P(v) from
    # log_probs_by_llm makes it comparable with log_probs_by_drafter.
    log_probs_by_llm = log_probs_by_llm[:, :, :-1, :]

    # Remove the token sampled from the vanilla LM head of the LLM from beams and leaves only those
    # from the drafter model.
    drafted_tokens = beams[..., 1:]

    # Look up Q(drafted_tokens) and P(drafted_tokens) in log_probs_by_drafter and log_probs_by_llm.
    drafted_tokens = drafted_tokens[..., None]
    by_drafter = mx.take_along_axis(log_probs_by_drafter, drafted_tokens, axis=-1).squeeze(axis=-1)
    by_llm = mx.take_along_axis(log_probs_by_llm, drafted_tokens, axis=-1).squeeze(axis=-1)

    # If a drafted token v has P(v) > Q(v), compare(v)==True for sure; otherwise, compare(v)==True
    # is subject to the probability P(v)/Q(v).
    compare = (
        mx.random.uniform(low=0, high=1, shape=by_drafter.shape) < mx.exp(by_llm - by_drafter)
    ).astype(mx.int32)
    # Among candidate token sequences in each of the batch of beams, choose the sequence that has
    # the longest prefix Trues in compare.
    n_tokens_in_seqs = mx.sum(
        mx.cumsum(compare, axis=-1) == mx.arange(1, compare.shape[-1] + 1), axis=-1
    )

    kth = n_tokens_in_seqs.shape[1] - 1
    seq_in_beam = mx.argpartition(n_tokens_in_seqs, kth=kth, axis=-1)[:, kth:]
    n_tokens_in_seq = mx.take_along_axis(n_tokens_in_seqs, seq_in_beam, axis=-1)
    return n_tokens_in_seq.squeeze(axis=-1), seq_in_beam.squeeze(axis=-1)


def _greedy_prepare_next_input(
    beams_by_llm: mx.array,
    last_hidden_state: mx.array,
    seq_in_beam: mx.array,
    n_tokens_in_seq: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Prepare the last_hidden_state and input_ids for the next generation of draft
    candidate tokens when greedy_search is enabled.

      beams_by_llm: (batch_size, beam_width, beam_length)

      last_hidden_state: (batch_size, beam_width, beam_length, hidden_size)

      seq_in_beam: (batch_size) Sequence index with the maximal number of accepted tokens.

      n_tokens_in_seq: (batch_size) The maximal number of accepted tokens per prompt per
      step.

    Returns:

      accepted_last_token_state: (batch_size, hidden_size) Last hidden state of the last
      accepted token in the accepted beam for the next iteration.

      next_tokens: (batch_size) The next token sampled from the current iteration for the
      next iteration.
    """
    # select next tokens
    selected_seqs = _select_one_per_row(beams_by_llm, seq_in_beam)
    next_tokens = _select_one_per_row(selected_seqs, n_tokens_in_seq)

    # select last token state
    selected_last_hidden_state = _select_one_per_row(last_hidden_state, seq_in_beam)
    accepted_last_token_state = _select_one_per_row(selected_last_hidden_state, n_tokens_in_seq)
    return accepted_last_token_state, next_tokens


def _prepare_next_input(
    log_probs_by_drafter: mx.array,
    log_probs_by_llm: mx.array,
    last_hidden_state: mx.array,
    seq_in_beam: mx.array,
    n_tokens_in_seq: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Prepare the last_hidden_state and input_ids for the next generation of draft
    candidate tokens.

      log_probs_by_drafter: (batch_size, beam_width, beam_length-1, vocab_size)
      Draft head log probs for beams.

      log_probs_by_llm: (batch_size, beam_width, beam_length, vocab_size)
      LLM log probs for beams and the predicted next token beyond beam candidates.

      last_hidden_state: (batch_size, beam_width, beam_length, hidden_size)

      seq_in_beam: (batch_size) Sequence index with the maximal number of accepted tokens.

      n_tokens_in_seq: (batch_size) The maximal number of accepted tokens per prompt per
      step.

    Returns:

      accepted_last_token_state: (batch_size, hidden_size) Last hidden state of the last
      accepted token in the accepted beam for the next iteration.

      next_tokens: (batch_size) The next token sampled from the current iteration for the
      next iteration.

    """
    # Select according to the chosen beam index.
    candidate_length = log_probs_by_drafter.shape[-2]
    last_log_probs_by_llm = log_probs_by_llm[:, :, -1, :]
    log_probs_by_llm = log_probs_by_llm[:, :, :-1, :]

    selected_log_probs_by_drafter = _select_one_per_row(log_probs_by_drafter, seq_in_beam)
    selected_log_probs_by_llm = _select_one_per_row(log_probs_by_llm, seq_in_beam)
    selected_last_log_probs_by_llm = _select_one_per_row(last_log_probs_by_llm, seq_in_beam)
    selected_last_hidden_state = _select_one_per_row(last_hidden_state, seq_in_beam)

    # Check if the entire beam is accepted or not.
    entire_beam_accepted = mx.expand_dims(n_tokens_in_seq == candidate_length, axis=-1)
    # If the entire beam is accepted, we use maybe_last_probs to sample next token.
    beam_last_probs = mx.exp(selected_last_log_probs_by_llm)

    # Note the shape of selected_log_probs_by_drafter and selected_log_probs_by_llm is the same
    # as [batch_size, candidate_length, vocab_size].
    # Thus, we clamp resampe_index to be up to candidate_length - 1.
    # Since when n_tokens_in_seq == candidate_length, we use maybe_last_probs above.
    # next_token_index = torch.clamp(n_tokens_in_seq, max=candidate_length - 1)
    next_token_index = n_tokens_in_seq - (n_tokens_in_seq == candidate_length).astype(
        n_tokens_in_seq.dtype
    )
    next_token_log_probs_by_drafter = _select_one_per_row(
        selected_log_probs_by_drafter, next_token_index
    )
    next_token_log_probs_by_llm = _select_one_per_row(selected_log_probs_by_llm, next_token_index)
    # Rejection sampling probs.
    #
    # probs = torch.clamp(
    #    torch.exp(next_token_log_probs_by_llm) - torch.exp(next_token_log_probs_by_drafter),
    #    min=0.0
    # )
    probs = mlx.nn.relu(
        mx.exp(next_token_log_probs_by_llm) - mx.exp(next_token_log_probs_by_drafter)
    )
    probs = mx.where(entire_beam_accepted, beam_last_probs, probs)
    # Modified for mlx: mlx.random.categorical only supports logits while the pytorch version
    # uses probs.
    log_probs = mx.log(probs)
    next_tokens = mx.random.categorical(logits=log_probs)
    # Collect the draft input for next
    accepted_last_token_state = _select_one_per_row(selected_last_hidden_state, n_tokens_in_seq)
    return accepted_last_token_state, next_tokens


def _count_left_paddings(tokens: mx.array, pad_token_id: int):
    return mx.sum(mx.cumprod((tokens == pad_token_id).astype(mx.int32), axis=-1), axis=-1)


def _present_kv_as_beam(
    beam_shape: modeling_drafter.BeamShape,
    past_kv_length: int,
    cache: kv_cache.Cache,
) -> Tuple[Tuple[mx.array, mx.array], ...]:
    """Split past keys, values and update the kv_cache. Return present keys and values.

    Returns:
      present_key_values: A [num_layers, 2] tuple.
      present_key_values[k][0] is the k-th layer key Tensor with shape
      (
        batch_size * num_kv_heads,
        beam_width,
        beam_length,
        head_dim,
      )
      present_key_values[k][1] is the k-th layer value Tensor with shape
      (
        batch_size * num_kv_heads,
        beam_width,
        beam_length,
        head_dim,
      )
    """
    batch_size, n_heads, _, head_dim = cache.sliced[0][0].shape
    present_key_values: Tuple[Tuple[mx.array, mx.array], ...] = ()
    for key_value in cache.sliced:
        key, value = key_value
        present_key = key.slice(past_kv_length).reshape(
            (batch_size * n_heads, beam_shape.width, beam_shape.length, head_dim)
        )
        present_value = value.slice(past_kv_length).reshape(
            (batch_size * n_heads, beam_shape.width, beam_shape.length, head_dim)
        )
        present_key_values = present_key_values + ((present_key, present_value),)

        key.length = value.length = past_kv_length

    return present_key_values


def _update_kv_cache_and_input_ids(
    input_ids: mx.array,
    n_tokens_in_seq: mx.array,
    seq_in_beam: mx.array,
    beams: mx.array,
    cache: kv_cache.Cache,
    pad_token_id: int,
) -> mx.array:
    """This function appends accepted tokens to input_ids and the associated keys and values to the
    KV cache. Input ids and the KV cache are right-aligned and left-padded to prepare for the next
    text decoding loop step.
    Note this MLX version focuses on batch_size = 1, thus there's no need to redo the left padding
    in Pytorch version.

      input_ids: (batch_size, previous_seq_len)

      n_tokens_in_seq: (batch_size) The maximal number of accepted tokens.

      seq_in_beam: (batch_size) The index of the sequence candidate which has the maximal
      number of accepted tokens in a beam.

      beams: (batch_size, beam_width, beam_length) Generated from the drafter model.

      pad_token_id: Pad token id.

    Returns:

      appended_input_ids: (batch_size, previous_seq_len+1+max(n_tokens_in_seq)) Updated input
      ids with the accepted tokens for the next iteration.

    An Example:

          input_ids |    selected_beams

        - hello what     |    is    your  name  <pad> <pad>

        n_tokens_in_seq = [2,] (Note this doesn't include the token sampled from llm.)
        seq_len = [5,]

        After the token appendment and re-alignment

          appended_input_ids

        - hello what  is    your  name

    """
    assert input_ids.dtype in [mx.int32, mx.int64]
    batch_size, beam_width, beam_length = beams.shape
    # Gather along selected beam index dimension.
    selected_seqs = _select_one_per_row(beams, seq_in_beam)

    # Collect the present keys and values
    present_key_values = _present_kv_as_beam(
        beam_shape=modeling_drafter.BeamShape(beam_width, beam_length),
        past_kv_length=input_ids.shape[1],
        cache=cache,
    )
    # Not counting the draft_init_token since it is used in drafting.
    _, n_heads, _, head_dim = cache.sliced[0][0].shape
    key_seq_in_beam = mx.repeat(seq_in_beam[:, None], n_heads, axis=1).reshape(-1)
    value_seq_in_beam = mx.repeat(seq_in_beam[:, None], n_heads, axis=1).reshape(-1)
    for key_value, present_key_value in zip(cache.sliced, present_key_values):
        past_key, past_value = key_value
        present_key, present_value = present_key_value
        selected_present_key = _select_one_per_row(present_key, key_seq_in_beam)
        selected_present_key = selected_present_key.reshape(
            (batch_size, n_heads, beam_length, head_dim)
        )
        selected_present_value = _select_one_per_row(present_value, value_seq_in_beam)
        selected_present_value = selected_present_value.reshape(
            (batch_size, n_heads, beam_length, head_dim)
        )
        past_key.cat(selected_present_key)
        past_value.cat(selected_present_value)

    selected_seqs = selected_seqs[:, : 1 + n_tokens_in_seq[0].item()]
    appended_input_ids = mx.concatenate([input_ids, selected_seqs], axis=-1)
    # Reset key and value length.
    for key, value in cache.sliced:
        key.length = value.length = appended_input_ids.shape[1]

    return appended_input_ids


def _comprehend_prompt(
    llm: mlx.nn.Module,
    input_ids: mx.array,
    cache: kv_cache.Cache,
    sampling_args: SamplingArgs,
    pad_token_id: int,
) -> Tuple[mx.array, mx.array]:
    """Calls llm to convert input_ids into cached keys and values. Also reuse the hidden states from
    the call to predict the next token and append it to input_ids.

      input_ids: (batch_size, prompt_length)

    Returns:

      draft_input: (batch_size, hidden_size). The hidden state of the last token.

      next_token: (batch_size). The next token generated from input_ids.
    """
    assert input_ids.dtype in [mx.int32, mx.int64]
    causal_mask = attention.causal_mask(input_ids != pad_token_id, input_ids.shape[1])
    # TODO add back position_ids is supported.
    # position_ids = (
    #     mx.repeat(mx.arange(input_ids.shape[1], dtype=mx.int32), \
    #           repeats=input_ids.shape[0], axis=0)
    #     - _count_left_paddings(input_ids, pad_token_id)[..., None]
    # )
    hidden_states, logits = llm(
        input_ids,
        beam_len=input_ids.shape[1],
        mask=causal_mask,
        cache=cache.sliced,
    )
    last_token_state = hidden_states[:, -1, :]
    next_token_logits = modeling_drafter.warp_logits(logits[:, -1, :])
    next_token = (
        mx.argmax(next_token_logits, axis=-1)
        if sampling_args.greedy
        else mx.random.categorical(logits=next_token_logits / sampling_args.temperature)
    )
    return (
        last_token_state,
        next_token,
    )


def _verify_candidates(
    llm: mlx.nn.Module,
    input_ids: mx.array,
    beams: mx.array,
    cache: kv_cache.Cache,
    pad_token_id: int,
) -> Tuple[mx.array, mx.array]:
    packed_beams, mask, position_offsets = tree_attention.pack(
        beams, padding_mask=(input_ids != pad_token_id).astype(input_ids.dtype)
    )
    # TODO Enable the compression from tree_attention and position_ids.
    hidden_states, logits = llm(
        packed_beams,
        mask=mask,
        cache=cache.sliced,
        beam_len=beams.shape[2],
    )

    logits = modeling_drafter.warp_logits(logits)
    hidden_states = tree_attention.unpack(hidden_states, beams.shape[1], beams.shape[2])
    logits = tree_attention.unpack(logits, beams.shape[1], beams.shape[2])
    # No need to unpack the kv cache since without compression, keys and values are
    # at the correct place.
    return hidden_states, logits


def _greedy_accept_candidate_tokens(
    input_ids: mx.array,
    beams_by_drafter: mx.array,
    logits_by_llm: mx.array,
    last_hidden_state: mx.array,
    cache: kv_cache.Cache,
    pad_token_id: int,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    beams_by_llm = mx.argmax(logits_by_llm, axis=-1)
    n_tokens_in_seq, seq_in_beam = _greedy_choose_from_candidates(
        beams_by_drafter,
        beams_by_llm,
    )
    input_ids = _update_kv_cache_and_input_ids(
        input_ids=input_ids,
        n_tokens_in_seq=n_tokens_in_seq,
        seq_in_beam=seq_in_beam,
        beams=beams_by_drafter,
        cache=cache,
        pad_token_id=pad_token_id,
    )
    last_token_state, next_tokens = _greedy_prepare_next_input(
        beams_by_llm, last_hidden_state, seq_in_beam, n_tokens_in_seq
    )

    return last_token_state, input_ids, next_tokens, n_tokens_in_seq


def _accept_candidate_tokens(
    input_ids: mx.array,
    beams: mx.array,
    logits_by_drafter: mx.array,
    logits_by_llm: mx.array,
    last_hidden_state: mx.array,
    cache: kv_cache.Cache,
    pad_token_id: int,
    temperature: float,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    log_probs_by_llm = mlx.nn.log_softmax(logits_by_llm / temperature, axis=-1)
    log_probs_by_drafter = mlx.nn.log_softmax(logits_by_drafter, axis=-1)
    n_tokens_in_seq, seq_in_beam = _choose_from_candidates(
        beams,
        log_probs_by_llm,
        logits_by_drafter,
    )
    input_ids = _update_kv_cache_and_input_ids(
        input_ids=input_ids,
        n_tokens_in_seq=n_tokens_in_seq,
        seq_in_beam=seq_in_beam,
        beams=beams,
        cache=cache,
        pad_token_id=pad_token_id,
    )
    last_token_state, next_tokens = _prepare_next_input(
        log_probs_by_drafter=log_probs_by_drafter,
        log_probs_by_llm=log_probs_by_llm,
        last_hidden_state=last_hidden_state,
        seq_in_beam=seq_in_beam,
        n_tokens_in_seq=n_tokens_in_seq,
    )

    return last_token_state, input_ids, next_tokens, n_tokens_in_seq


def n_kv_heads(model: mlx.nn.Module) -> int:
    if isinstance(model, modeling_llama.Model):
        return model.args.num_key_value_heads  # type: ignore
    else:
        raise TypeError(f"Unsupported base model type {type(model)}")


class ReDrafterModel(mlx.nn.Module):
    def __init__(
        self,
        llm: mlx.nn.Module,
        drafter: modeling_drafter.Drafter,
    ):
        super().__init__()
        self.llm = llm
        self.drafter = drafter

    def generate(
        self,
        input_ids: mx.array,
        max_length: int,
        beam_shape: modeling_drafter.BeamShape = modeling_drafter.BeamShape(10, -1),
        sampling_args: SamplingArgs = SamplingArgs(1.0, False),
        special_tokens: SpecialTokens = SpecialTokens(0, 1),
    ) -> Generator[mx.array, None, None]:
        for output_tokens in self._generate(
            input_ids=input_ids,
            max_length=max_length,
            beam_shape=beam_shape,
            sampling_args=sampling_args,
            special_tokens=special_tokens,
        ):
            yield output_tokens

    def _generate(
        self,
        input_ids: mx.array,
        max_length: int,
        beam_shape: modeling_drafter.BeamShape,
        sampling_args: SamplingArgs,
        special_tokens: SpecialTokens,
    ) -> Generator[mx.array, None, None]:
        assert input_ids.dtype in [mx.int32, mx.int64]
        batch_size, seq_len = input_ids.shape
        init_seq_len = seq_len

        cache = kv_cache.Cache(
            batch_size=batch_size,
            max_length=max_length + beam_shape.length * beam_shape.width,
            n_layers=self.llm.args.num_hidden_layers,
            n_heads=n_kv_heads(self.llm),
            head_dim=self.llm.args.hidden_size // self.llm.args.num_attention_heads,
            dtype=self.llm.lm_head.weight.dtype,
        )

        drafting_context, drafting_init_tokens = _comprehend_prompt(
            self.llm,
            input_ids,
            cache,
            sampling_args,
            special_tokens.pad,
        )
        while seq_len < max_length:
            # Call the draft head to generate candidates.
            # Generate draft candidates conditioning on the draft_input and next_token.
            beams, logits_by_drafter = self.drafter.beam_search_candidates(
                drafting_context,
                drafting_init_tokens,
                self.llm.model.input_embeddings,
                beam_shape,
            )
            hidden_states, logits_by_llm = _verify_candidates(
                self.llm,
                input_ids,
                beams,
                cache,
                special_tokens.pad,
            )
            if sampling_args.greedy:
                (
                    drafting_context,
                    input_ids,
                    drafting_init_tokens,
                    n_tokens_in_seq,
                ) = _greedy_accept_candidate_tokens(
                    input_ids,
                    beams,
                    logits_by_llm,
                    hidden_states,
                    cache,
                    special_tokens.pad,
                )
            else:
                (
                    drafting_context,
                    input_ids,
                    drafting_init_tokens,
                    n_tokens_in_seq,
                ) = _accept_candidate_tokens(
                    input_ids,
                    beams,
                    logits_by_drafter,
                    logits_by_llm,
                    last_hidden_state=hidden_states,
                    cache=cache,
                    pad_token_id=special_tokens.pad,
                    temperature=sampling_args.temperature,
                )

            output_tokens = mx.concatenate((input_ids, drafting_init_tokens[..., None]), axis=-1)
            seq_len = output_tokens.shape[1]
            yield (
                output_tokens[:, :max_length]
                if output_tokens.shape[1] > max_length
                else output_tokens
            )
            # EOS check for the whole batch.
            if (
                special_tokens.eos is not None
                and ((output_tokens[:, init_seq_len:] == special_tokens.eos).sum(axis=-1) > 0).sum()
                == batch_size
            ):
                break
