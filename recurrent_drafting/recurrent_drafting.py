#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import torch
import torch.distributions as dist
import transformers

from . import (
    attention,
    kv_cache,
    modeling_drafter,
    modeling_llama,
    stats,
    tree_attention,
)

# https://rb.gy/ypfmuu
dist.Distribution.set_default_validate_args(False)


@dataclass
class SamplingArgs:
    temperature: float
    greedy: bool  # greedy_search=True eliminates randomness


@dataclass
class SpecialTokens:
    pad: int
    eos: Optional[int] = 1


def _select_one_per_row(x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """x: (batch_size, seq_len, ....). batch_index: (batch_size,),int.
    Return (batch_size, 1, ...)."""
    return x[torch.arange(x.shape[0], device=x.device), batch_index]


def _greedy_accept_candidate_tokens(
    input_ids: torch.Tensor,
    beams_by_drafter: torch.Tensor,
    logits_by_llm: torch.Tensor,
    last_hidden_state: torch.Tensor,
    cache: kv_cache.Cache,
    pad_token_id: int,
    step_record: stats.TextGeneration.Step,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beams_by_llm = torch.argmax(logits_by_llm, dim=-1)
    with step_record.time("choose_from_candidates"):
        n_tokens_in_seq, seq_in_beam = _greedy_choose_from_candidates(
            beams_by_drafter,
            beams_by_llm,
        )
    with step_record.time("update_kv_cache_and_input_ids"):
        input_ids = _update_kv_cache_and_input_ids(
            input_ids=input_ids,
            n_tokens_in_seq=n_tokens_in_seq,
            seq_in_beam=seq_in_beam,
            beams=beams_by_drafter,
            cache=cache,
            pad_token_id=pad_token_id,
        )
    with step_record.time("greedy_prepare_next_input"):
        last_token_state, next_tokens = _greedy_prepare_next_input(
            beams_by_llm, last_hidden_state, seq_in_beam, n_tokens_in_seq
        )

    return last_token_state, input_ids, next_tokens, n_tokens_in_seq


def _accept_candidate_tokens(
    input_ids: torch.Tensor,
    beams: torch.Tensor,
    log_probs_by_drafter: torch.Tensor,
    logits_by_llm: torch.Tensor,
    last_hidden_state: torch.Tensor,
    cache: kv_cache.Cache,
    pad_token_id: int,
    temperature: float,
    step_record: stats.TextGeneration.Step,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs_by_llm = (logits_by_llm / temperature).log_softmax(dim=-1)
    with step_record.time("choose_from_candidates"):
        n_tokens_in_seq, seq_in_beam = _choose_from_candidates(
            beams,
            log_probs_by_llm,
            log_probs_by_drafter,
        )
    with step_record.time("update_kv_cache_and_input_ids"):
        input_ids = _update_kv_cache_and_input_ids(
            input_ids=input_ids,
            n_tokens_in_seq=n_tokens_in_seq,
            seq_in_beam=seq_in_beam,
            beams=beams,
            cache=cache,
            pad_token_id=pad_token_id,
        )
    with step_record.time("prepare_next_input"):
        last_token_state, next_tokens = _prepare_next_input(
            log_probs_by_drafter=log_probs_by_drafter,
            log_probs_by_llm=log_probs_by_llm,
            last_hidden_state=last_hidden_state,
            seq_in_beam=seq_in_beam,
            n_tokens_in_seq=n_tokens_in_seq,
        )

    return last_token_state, input_ids, next_tokens, n_tokens_in_seq


def _update_kv_cache_and_input_ids(
    input_ids: torch.Tensor,
    n_tokens_in_seq: torch.Tensor,
    seq_in_beam: torch.Tensor,
    beams: torch.Tensor,
    cache: kv_cache.Cache,
    pad_token_id: int,
):
    """This function appends accepted tokens to input_ids and the associated keys and values to the
    KV cache. Input ids and the KV cache are right-aligned and left-padded to prepare for the next
    text decoding loop step.

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
        - <pad> I        |    am    bob   how   are   you

        num_prev_left_pads = [0, 1]
        n_tokens_in_seq = [2, 4] (Note this doesn't include the token sampled from llm.)
        seq_len = [5, 6]
        max_seq_len = 6
        num_realigned_left_pads = [1, 0]

        After the token appendment and re-alignment

          appended_input_ids

        - <pad> hello what  is    your  name
        - I     am    bob   how   are   you

    """
    assert input_ids.dtype == torch.long, f"prev_input_ids.dtype {input_ids.dtype} is not long"
    batch_size, beam_width, beam_length = beams.shape
    # Caluate the max new length.
    num_prev_left_pads = _count_left_paddings(input_ids, pad_token_id)
    # concat_length: (batch_size,)
    n_tokens_in_seq_with_init = n_tokens_in_seq + 1
    seq_len = input_ids.shape[1] - num_prev_left_pads + n_tokens_in_seq_with_init
    max_seq_len = torch.max(seq_len)
    num_realigned_left_pads = max_seq_len - seq_len

    # Gather along selected beam index dimension.
    selected_seqs = _select_one_per_row(beams, seq_in_beam)

    # Collect the present keys and values
    present_key_values = _present_kv_as_beam(
        beam_shape=modeling_drafter.BeamShape(beam_width, beam_length),
        past_kv_length=input_ids.shape[1],
        cache=cache,
    )
    # Not counting the draft_init_token since it is used in drafting.
    _, _, _, n_heads, _, head_dim = cache._cache.shape
    key_seq_in_beam = seq_in_beam[:, None].expand(batch_size, n_heads).reshape(-1)
    value_seq_in_beam = seq_in_beam[:, None].expand(batch_size, n_heads).reshape(-1)
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

    # Updating kv_cache.View requires shifting examples with different offsets.
    combined_key_value = cache._cache[:, :, :, : cache.sliced[0][0].length, :].clone()
    for i in range(batch_size):
        start = num_prev_left_pads[i]
        end = input_ids.shape[1] + n_tokens_in_seq_with_init[i]
        assert max_seq_len - num_realigned_left_pads[i] == end - start
        # Right aligned.
        cache._cache[:, :, i, :, num_realigned_left_pads[i] : max_seq_len, :] = combined_key_value[
            :, :, i, :, start:end, :
        ]

    # Collect new input_ids.
    appended_input_ids = torch.full(
        size=(batch_size, int(max_seq_len.item())),
        fill_value=pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    for i in range(batch_size):
        # Copy previous ids, excluding the pad ids on the left.
        start = num_realigned_left_pads[i]
        end = start + input_ids.shape[1] - num_prev_left_pads[i]
        appended_input_ids[i, start:end] = input_ids[i, num_prev_left_pads[i] :]
        # Copy accepted proposed ids.
        start = end
        end = start + n_tokens_in_seq_with_init[i]
        appended_input_ids[i, start:end] = selected_seqs[i, : n_tokens_in_seq_with_init[i]]

    # Reset key and value length.
    for key, value in cache.sliced:
        key.length = value.length = appended_input_ids.shape[1]

    return appended_input_ids


def _greedy_prepare_next_input(
    beams_by_llm: torch.Tensor,
    last_hidden_state: torch.Tensor,
    seq_in_beam: torch.Tensor,
    n_tokens_in_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    log_probs_by_drafter: torch.Tensor,
    log_probs_by_llm: torch.Tensor,
    last_hidden_state: torch.Tensor,
    seq_in_beam: torch.Tensor,
    n_tokens_in_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    entire_beam_accepted = torch.eq(n_tokens_in_seq, candidate_length).unsqueeze(dim=-1)
    # If the entire beam is accepted, we use maybe_last_probs to sample next token.
    beam_last_probs = torch.exp(selected_last_log_probs_by_llm)

    # Note the shape of selected_log_probs_by_drafter and selected_log_probs_by_llm is the same
    # as [batch_size, candidate_length, vocab_size].
    # Thus, we clamp resampe_index to be up to candidate_length - 1.
    # Since when n_tokens_in_seq == candidate_length, we use maybe_last_probs above.
    # next_token_index = torch.clamp(n_tokens_in_seq, max=candidate_length - 1)
    next_token_index = n_tokens_in_seq - (n_tokens_in_seq == candidate_length).to(
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
    probs = torch.relu(
        torch.exp(next_token_log_probs_by_llm) - torch.exp(next_token_log_probs_by_drafter)
    )
    probs = torch.where(entire_beam_accepted, beam_last_probs, probs)
    next_tokens = dist.Categorical(probs=probs).sample()
    # Collect the draft input for next
    accepted_last_token_state = _select_one_per_row(selected_last_hidden_state, n_tokens_in_seq)
    return accepted_last_token_state, next_tokens


def _present_kv_as_beam(
    beam_shape: modeling_drafter.BeamShape,
    past_kv_length: int,
    cache: kv_cache.Cache,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
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
    present_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = ()
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


def _count_left_paddings(tokens: torch.Tensor, pad_token_id: int):
    return torch.sum(torch.cumprod(torch.eq(tokens, pad_token_id), dim=-1), dim=-1)


def _comprehend_prompt(
    llm: transformers.PreTrainedModel,
    input_ids: torch.Tensor,
    cache: kv_cache.Cache,
    sampling_args: SamplingArgs,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calls llm to convert input_ids into cached keys and values. Also reuse the hidden states from
    the call to predict the next token and append it to input_ids.

      input_ids: (batch_size, prompt_length)

    Returns:

      draft_input: (batch_size, hidden_size). The hidden state of the last token.

      next_token: (batch_size). The next token generated from input_ids.
    """
    assert input_ids.dtype in [torch.int32, torch.int64]
    causal_mask = attention.causal_mask(
        input_ids != pad_token_id, input_ids.shape[1], input_ids.device
    )
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).expand(input_ids.shape) - _count_left_paddings(input_ids, pad_token_id).unsqueeze(dim=-1)
    llm_output = llm(
        input_ids,
        past_key_values=cache.sliced,
        attention_mask=causal_mask,
        position_ids=position_ids,
    )
    last_token_state = llm_output.hidden_states[:, -1, :]  # (batch_size, hidden_size)
    next_token_logits = modeling_drafter.warp_logits(llm_output.logits[:, -1, :])
    next_token = (
        torch.argmax(next_token_logits, dim=-1)
        if sampling_args.greedy
        else dist.Categorical(
            probs=torch.softmax(next_token_logits / sampling_args.temperature, dim=-1)
        ).sample()
    )
    return (
        last_token_state,
        next_token,
    )


def _verify_candidates(
    step_record: stats.TextGeneration.Step,
    llm: transformers.PreTrainedModel,
    input_ids: torch.Tensor,
    beams: torch.Tensor,
    cache: kv_cache.Cache,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    packed_beams, unpacker, attention_mask, position_offsets = tree_attention.pack(
        beams,
        padding_mask=(input_ids != pad_token_id).to(input_ids.dtype),
    )
    past_kv_len = cache.sliced[0][0].length - _count_left_paddings(
        input_ids, pad_token_id
    ).unsqueeze(dim=-1)
    with step_record.time("base_forward"):
        llm_output = llm(
            packed_beams,
            attention_mask=attention_mask,
            past_key_values=cache.sliced,
            position_ids=past_kv_len + position_offsets,
        )

    logits = modeling_drafter.warp_logits(llm_output.logits)

    hidden_states = tree_attention.unpack(llm_output.hidden_states, unpacker)
    logits = tree_attention.unpack(logits, unpacker)
    cache.unpack(input_ids.shape[-1], unpacker)
    return hidden_states, logits


def _greedy_choose_from_candidates(
    beams_by_drafter: torch.Tensor,
    beams_by_llm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:  # n_tokens_in_seq  # seq_in_beam
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
    compare = beams_by_drafter_without_init == beams_by_llm_without_last
    n_tokens_in_seqs = torch.sum(torch.cumprod(compare, dim=-1), dim=-1)
    return torch.max(n_tokens_in_seqs, dim=-1)


def _choose_from_candidates(
    beams: torch.Tensor,
    log_probs_by_llm: torch.Tensor,
    log_probs_by_drafter: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:  # n_tokens_in_seq  # seq_in_beam
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
    assert beams.dtype in [torch.int32, torch.int64]

    # Due to the index shift between log_probs_by_llm and _by_drafter, removing the last P(v) from
    # log_probs_by_llm makes it comparable with log_probs_by_drafter.
    log_probs_by_llm = log_probs_by_llm[:, :, :-1, :]

    # Remove the token sampled from the vanilla LM head of the LLM from beams and leaves only those
    # from the drafter model.
    drafted_tokens = beams[..., 1:]

    # Look up Q(drafted_tokens) and P(drafted_tokens) in log_probs_by_drafter and log_probs_by_llm.
    drafted_tokens = drafted_tokens.unsqueeze(dim=-1)
    by_drafter = torch.gather(log_probs_by_drafter, dim=-1, index=drafted_tokens).squeeze(dim=-1)
    by_llm = torch.gather(log_probs_by_llm, dim=-1, index=drafted_tokens).squeeze(dim=-1)

    # If a drafted token v has P(v) > Q(v), compare(v)==True for sure; otherwise, compare(v)==True
    # is subject to the probability P(v)/Q(v).
    compare = torch.lt(
        torch.rand(by_drafter.shape, device=by_drafter.device), torch.exp(by_llm - by_drafter)
    )

    # Among candidate token sequences in each of the batch of beams, choose the sequence that has
    # the longest prefix Trues in compare.
    n_tokens_in_seqs = torch.sum(
        torch.cumsum(compare, dim=-1)
        == torch.arange(1, compare.shape[-1] + 1, device=compare.device),
        dim=-1,
    )
    return torch.max(n_tokens_in_seqs, dim=-1)


class RecurrentDrafting(torch.nn.Module):
    def __init__(
        self,
        llm: transformers.PreTrainedModel,
        drafter: modeling_drafter.Drafter,
    ):
        super().__init__()
        self.llm = llm
        self.drafter = drafter

    def llm_n_kv_heads(self) -> int:
        if isinstance(self.llm, modeling_llama.LlamaForCausalLM):
            return self.llm.config.num_key_value_heads
        else:
            raise TypeError(f"Unsupported base model type {type(self.llm)}")

    def generate(
        self,
        ledger: stats.Ledger,
        input_ids: torch.Tensor,
        max_length: int,
        beam_shape: modeling_drafter.BeamShape = modeling_drafter.BeamShape(10, -1),
        sampling_args: SamplingArgs = SamplingArgs(1.0, False),
        special_tokens: SpecialTokens = SpecialTokens(0, 1),
    ) -> Generator[torch.Tensor, None, None]:
        with torch.inference_mode():
            with stats.text_generation(ledger) as r:
                for output_tokens in self._generate(
                    text_generation=r,
                    input_ids=input_ids,
                    max_length=max_length,
                    beam_shape=beam_shape,
                    sampling_args=sampling_args,
                    special_tokens=special_tokens,
                ):
                    yield output_tokens

    def _generate(
        self,
        text_generation: stats.TextGeneration,
        input_ids: torch.Tensor,
        max_length: int,
        beam_shape: modeling_drafter.BeamShape,
        sampling_args: SamplingArgs,
        special_tokens: SpecialTokens,
    ) -> Generator[torch.Tensor, None, None]:
        assert input_ids.dtype == torch.long, f"input_ids.dtype {input_ids.dtype} is not long"
        batch_size, seq_len = input_ids.shape
        init_seq_len = seq_len

        cache = kv_cache.Cache(
            batch_size=batch_size,
            max_length=max_length + beam_shape.length * beam_shape.width,
            n_layers=self.llm.config.num_hidden_layers,
            n_heads=self.llm_n_kv_heads(),
            head_dim=self.llm.config.hidden_size // self.llm.config.num_attention_heads,
            dtype=self.llm.dtype,
            device=input_ids.device,
        )

        drafting_context, drafting_init_tokens = _comprehend_prompt(
            self.llm,
            input_ids,
            cache,
            sampling_args,
            special_tokens.pad,
        )
        while seq_len < max_length:
            with text_generation.step() as step_record:
                # Call the draft head to generate candidates.
                with step_record.time("generate_candidates"):
                    # Generate draft candidates conditioning on the draft_input and next_token.
                    beams, log_probs_by_drafter = self.drafter.beam_search_candidates(
                        drafting_context,
                        drafting_init_tokens,
                        self.llm.base_model.get_input_embeddings(),
                        beam_shape,
                    )
                with step_record.time("verify_candidates"):
                    hidden_states, logits_by_llm = _verify_candidates(
                        step_record,
                        self.llm,
                        input_ids,
                        beams,
                        cache,
                        special_tokens.pad,
                    )
                if sampling_args.greedy:
                    with step_record.time("accept_candidate_tokens"):
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
                            step_record,
                        )
                else:
                    with step_record.time("accept_candidate_tokens"):
                        (
                            drafting_context,
                            input_ids,
                            drafting_init_tokens,
                            n_tokens_in_seq,
                        ) = _accept_candidate_tokens(
                            input_ids,
                            beams,
                            log_probs_by_drafter,
                            logits_by_llm,
                            last_hidden_state=hidden_states,
                            cache=cache,
                            pad_token_id=special_tokens.pad,
                            temperature=sampling_args.temperature,
                            step_record=step_record,
                        )

                output_tokens = torch.cat((input_ids, drafting_init_tokens.unsqueeze(-1)), dim=-1)

                seq_len = output_tokens.shape[1]
                step_record.set(
                    {
                        "input_id_length": output_tokens.shape[1],
                        "avg_num_accepted": torch.sum(n_tokens_in_seq).item() / batch_size,
                    }
                )

                yield (
                    output_tokens[:, :max_length]
                    if output_tokens.shape[1] > max_length
                    else output_tokens
                )

                # EOS check for the whole batch.
                if (
                    special_tokens.eos is not None
                    and (
                        (output_tokens[:, init_seq_len:] == special_tokens.eos).sum(dim=-1) > 0
                    ).sum()
                    == batch_size
                ):
                    break
