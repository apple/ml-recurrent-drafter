#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Usage:
1. Download the LLM and the tokenizer to $HOME/m

2. Run this script

   python recurrent_drafting/mlx/experiments/benchmark_autoregression.py \
    > /tmp/autoregression.csv

   Or, if you want to watch the output in the terminal, you need unbuffer, which comes with expect.

   brew install expect

   unbuffer python recurrent_drafting/mlx/experiments/benchmark_autoregression.py \
    | tee /tmp/autoregression.csv
"""
import itertools
import os

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.models.base
import mlx_lm.models.llama
import transformers

import recurrent_drafting.mlx

MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
new_ids = tokenizer(
    "A chat between a curious user and an artificial intelligence assistant. "
    + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    + "USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting "
    + "cultural experiences and must-see attractions. ASSISTANT:"
).input_ids
PROMPT = mx.array([new_ids])  # batch size = 1


def benchmark_our_model(dtype: mx.Dtype, num_new_tokens: int) -> None:
    @recurrent_drafting.mlx.time_mlx.function("test process prompt takes")
    def process_prompt(prompt, our_model, sliced_cache):
        mask = recurrent_drafting.mlx.attention.causal_mask(
            mx.ones(shape=(1, prompt.shape[1]), dtype=mx.bool_), prompt.shape[1]
        )
        logits = our_model(prompt, prompt.shape[1], mask, sliced_cache)
        return logits[1][:, -1, :]

    @recurrent_drafting.mlx.time_mlx.function(f"test generate {num_new_tokens} tokens takes")
    def generate_tokens(logits, model, cache, prompt_len):
        for i in range(num_new_tokens - 1):
            y = mx.argmax(mx.softmax(logits), axis=-1)
            mask = recurrent_drafting.mlx.attention.causal_mask(
                mx.ones(shape=(1, prompt_len + i + 1), dtype=mx.bool_), 1
            )
            logits = model(y[None], 1, mask, cache)[1][:, -1, :]
        return mx.argmax(logits, axis=-1)

    our_model = recurrent_drafting.mlx.modeling_llama.load_model(MODEL_PATH)
    our_model.set_dtype(dtype)
    cache = recurrent_drafting.mlx.kv_cache.Cache(
        batch_size=1,
        max_length=PROMPT.shape[1] + num_new_tokens,
        n_layers=our_model.args.num_hidden_layers,
        n_heads=our_model.args.num_key_value_heads or our_model.args.num_attention_heads,
        head_dim=our_model.args.hidden_size // our_model.args.num_attention_heads,
        dtype=our_model.model.embed_tokens.weight.dtype,
    )
    logits = process_prompt(PROMPT, our_model, cache.sliced)
    generate_tokens(logits, our_model, cache.sliced, PROMPT.shape[1])


def benchmark_mlxlm_model(dtype: mx.Dtype, num_new_tokens: int) -> None:
    @recurrent_drafting.mlx.time_mlx.function("ref prompt processing takes")
    def process_prompt(prompt, model, cache):
        return model(prompt, cache=cache)[:, -1, :]

    @recurrent_drafting.mlx.time_mlx.function(f"ref generate {num_new_tokens} tokens takes")
    def generate_tokens(logits, model, cache):
        for _ in range(num_new_tokens - 1):
            y = mx.argmax(mx.softmax(logits), axis=-1)
            logits = model(y[None], cache)[:, -1, :]
        return mx.argmax(logits, axis=-1)

    mlxlm_model, _ = mlx_lm.utils.load(MODEL_PATH)
    mlxlm_model.set_dtype(dtype)
    ref_cache = [
        mlx_lm.models.base.KVCache(
            mlxlm_model.args.hidden_size // mlxlm_model.args.num_attention_heads,
            mlxlm_model.args.num_key_value_heads,
        )
        for _ in range(mlxlm_model.args.num_hidden_layers)
    ]
    logits = process_prompt(PROMPT, mlxlm_model, ref_cache)
    generate_tokens(logits, mlxlm_model, ref_cache)


if __name__ == "__main__":
    print(
        "run,dtype,comprehension_mlx,generation_mlx,comprehension_ours,generation_ours"
    )  # table header
    for run, dtype, num_new_tokens in itertools.product(range(2), [mx.float16, mx.bfloat16], [200]):
        mlx.core.metal.clear_cache()
        recurrent_drafting.mlx.time_mlx.ledger.reset()
        benchmark_mlxlm_model(dtype, num_new_tokens)
        assert len(recurrent_drafting.mlx.time_mlx.ledger.records) == 2
        comprehension_mlx = recurrent_drafting.mlx.time_mlx.ledger.records[0].timing[0]
        generation_mlx = recurrent_drafting.mlx.time_mlx.ledger.records[1].timing[0]

        mlx.core.metal.clear_cache()
        recurrent_drafting.mlx.time_mlx.ledger.reset()
        benchmark_our_model(dtype, num_new_tokens)
        assert len(recurrent_drafting.mlx.time_mlx.ledger.records) == 2
        comprehension_ours = recurrent_drafting.mlx.time_mlx.ledger.records[0].timing[0]
        generation_ours = recurrent_drafting.mlx.time_mlx.ledger.records[1].timing[0]

        print(
            f"{run},{dtype},{comprehension_mlx},{generation_mlx},"
            + f"{comprehension_ours},{generation_ours}"
        )
