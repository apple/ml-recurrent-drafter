from typing import Generator, Tuple

import mlx
import mlx.core as mx

from . import attention, kv_cache, modeling_llama, recurrent_drafting


def supports_cache(model: mlx.nn.Module) -> bool:
    if isinstance(model, modeling_llama.Model):
        return True
    return False


def streaming_generate(
    model: mlx.nn.Module,
    input_ids: mx.array,
    max_length: int,
    special_tokens: recurrent_drafting.SpecialTokens,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    cache = (
        kv_cache.Cache(
            batch_size=input_ids.shape[0],
            max_length=max_length,
            n_layers=model.args.num_hidden_layers,
            n_heads=recurrent_drafting.n_kv_heads(model),
            head_dim=model.args.hidden_size // model.args.num_attention_heads,
            dtype=model.lm_head.weight.dtype,
        )
        if supports_cache(model)
        else None
    )
    context_len = input_ids.shape[1]
    output_logits = mx.zeros((0, model.args.vocab_size))
    while input_ids.shape[1] < max_length:
        padding_mask = input_ids != special_tokens.pad
        if cache is not None:
            cur_input_ids = input_ids[:, -1:] if input_ids.shape[1] > context_len else input_ids
            attention_mask = attention.causal_mask(padding_mask, cur_input_ids.shape[1])
            output = model(
                inputs=cur_input_ids,
                cache=cache.sliced,
                mask=attention_mask,
                beam_len=1,
            )
        else:
            output = model(
                inputs=input_ids,
                mask=padding_mask,
            )
        logits = output[1][:, -1, :]
        mx.eval(logits)
        next_token = mx.argmax(logits, axis=-1)
        input_ids = mx.concatenate([input_ids, next_token[..., None]], axis=-1)
        output_logits = mx.concatenate([output_logits, logits])
        yield input_ids, output_logits
        if (
            special_tokens
            and ((input_ids[:, context_len:] == special_tokens.eos).sum(axis=-1) > 0).sum()
            == input_ids.shape[0]
        ):
            break


def generate(
    model: mlx.nn.Module,
    input_ids: mx.array,
    max_new_tokens: int,
    special_tokens: recurrent_drafting.SpecialTokens = recurrent_drafting.SpecialTokens(0, 1),
) -> Tuple[mx.array, mx.array]:
    streaming_generator = streaming_generate(
        model, input_ids, max_new_tokens + input_ids.shape[1], special_tokens=special_tokens
    )
    tokens, logits = next(streaming_generator)
    for tokens, logits in streaming_generator:
        continue
    return tokens, logits
