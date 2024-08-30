"""Usage:
python recurrent_drafting/mlx/experiments/benchmark_llama.py
"""

import itertools
import os

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.models.base
import mlx_lm.models.llama

import recurrent_drafting.mlx

MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
"""<s> A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user\'s questions.
USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting
cultural experiences and must-see attractions. ASSISTANT:
"""
# Disable the multi-line reformatting by black.
# fmt: off
PROMPT = mx.array(
    [
        [
            529, 29879, 29958, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082
            , 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568
            , 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 3831
            , 852, 385, 3033, 6751, 9850, 12618, 1400, 1048, 263, 7786, 17487, 304, 26901
            , 29875, 29892, 12141, 292, 16375, 27482, 322, 1818, 29899, 4149, 19650, 1953
            , 29889, 319, 1799, 9047, 13566, 29901
        ]
    ]
)
# fmt: on


def benchmark_base_model(dtype: mx.Dtype, num_new_tokens: int) -> None:
    @recurrent_drafting.mlx.time_mlx.function("test process prompt takes")
    def process_prompt(prompt, base_model, sliced_cache):
        mask = recurrent_drafting.mlx.attention.causal_mask(
            mx.ones(shape=(1, prompt.shape[1]), dtype=mx.bool_), prompt.shape[1]
        )
        logits = base_model(prompt, 1, mask, sliced_cache)
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

    base_model = recurrent_drafting.mlx.modeling_llama.load_model(MODEL_PATH)
    base_model.set_dtype(dtype)
    cache = recurrent_drafting.mlx.kv_cache.Cache(
        batch_size=1,
        max_length=PROMPT.shape[1] + num_new_tokens,
        n_layers=base_model.args.num_hidden_layers,
        n_heads=base_model.args.num_key_value_heads or base_model.args.num_attention_heads,
        head_dim=base_model.args.hidden_size // base_model.args.num_attention_heads,
        dtype=base_model.model.embed_tokens.weight.dtype,
    )
    logits = process_prompt(PROMPT, base_model, cache.sliced)
    generate_tokens(logits, base_model, cache.sliced, PROMPT.shape[1])


def benchmark_ref_model(dtype: mx.Dtype, num_new_tokens: int) -> None:
    @recurrent_drafting.mlx.time_mlx.function("ref prompt processing takes")
    def process_prompt(prompt, model, cache):
        return model(prompt, cache=cache)[:, -1, :]

    @recurrent_drafting.mlx.time_mlx.function(f"ref generate {num_new_tokens} tokens takes")
    def generate_tokens(logits, model, cache):
        for _ in range(num_new_tokens - 1):
            y = mx.argmax(mx.softmax(logits), axis=-1)
            logits = model(y[None], cache)[:, -1, :]
        return mx.argmax(logits, axis=-1)

    ref_model, _ = mlx_lm.utils.load(MODEL_PATH)
    ref_model.set_dtype(dtype)
    ref_cache = [
        mlx_lm.models.base.KVCache(
            ref_model.args.hidden_size // ref_model.args.num_attention_heads,
            ref_model.args.num_key_value_heads,
        )
        for _ in range(ref_model.args.num_hidden_layers)
    ]
    logits = process_prompt(PROMPT, ref_model, ref_cache)
    generate_tokens(logits, ref_model, ref_cache)


if __name__ == "__main__":
    print("dtype,implementation,comprehension,generation")  # table header
    for run, dtype, num_new_tokens in itertools.product(range(2), [mx.float16, mx.bfloat16], [100]):
        mlx.core.metal.clear_cache()
        recurrent_drafting.mlx.time_mlx.ledger.reset()
        benchmark_ref_model(dtype, num_new_tokens)
        assert len(recurrent_drafting.mlx.time_mlx.ledger.records) == 2
        print(
            f"{dtype},MLX's LLaMA,{recurrent_drafting.mlx.time_mlx.ledger.records[0].timing[0]},"
            + f"{recurrent_drafting.mlx.time_mlx.ledger.records[1].timing[0]}"
        )

        mlx.core.metal.clear_cache()
        recurrent_drafting.mlx.time_mlx.ledger.reset()
        benchmark_base_model(dtype, num_new_tokens)
        assert len(recurrent_drafting.mlx.time_mlx.ledger.records) == 2
        print(
            f"{dtype},MLX's LLaMA,{recurrent_drafting.mlx.time_mlx.ledger.records[0].timing[0]},"
            + f"{recurrent_drafting.mlx.time_mlx.ledger.records[1].timing[0]}"
        )
