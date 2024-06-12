import os

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.models.base
import mlx_lm.models.llama

import mlx_recurrent_drafting

MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
NUM_NEW_TOKENS = 100
PROMPT = mx.array([[1, 2, 3, 4, 5, 6]])


def benchmark_base_model() -> None:
    @mlx_recurrent_drafting.time_mlx.function("test process prompt takes")
    def process_prompt(prompt, base_model, sliced_cache):
        mask = mlx_recurrent_drafting.attention.causal_mask(
            mx.ones(shape=(1, prompt.shape[1]), dtype=mx.bool_), prompt.shape[1]
        )
        logits = base_model(prompt, mx.arange(prompt.shape[1])[None], mask, sliced_cache)
        return logits[1][:, -1, :]

    @mlx_recurrent_drafting.time_mlx.function(f"test generate {NUM_NEW_TOKENS} tokens takes")
    def generate_tokens(logits, model, cache, prompt_len):
        for i in range(NUM_NEW_TOKENS - 1):
            y = mx.argmax(mx.softmax(logits), axis=-1)
            mask = mlx_recurrent_drafting.attention.causal_mask(
                mx.ones(shape=(1, prompt_len + i + 1), dtype=mx.bool_), 1
            )
            logits = model(y[None], mx.array([[prompt_len + i + 1]]), mask, cache)[1][:, -1, :]
        return mx.argmax(logits, axis=-1)

    base_model = mlx_recurrent_drafting.modeling_llama.load_model(MODEL_PATH)
    cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size=1,
        max_length=PROMPT.shape[1] + NUM_NEW_TOKENS,
        n_layers=base_model.args.num_hidden_layers,
        n_heads=base_model.args.num_key_value_heads or base_model.args.num_attention_heads,
        head_dim=base_model.args.hidden_size // base_model.args.num_attention_heads,
        dtype=base_model.model.embed_tokens.weight.dtype,
    )
    logits = process_prompt(PROMPT, base_model, cache.sliced)
    generate_tokens(logits, base_model, cache.sliced, PROMPT.shape[1])


def benchmark_ref_model() -> None:
    @mlx_recurrent_drafting.time_mlx.function("ref prompt processing takes")
    def process_prompt(prompt, model, cache):
        return model(prompt, cache=cache)[:, -1, :]

    @mlx_recurrent_drafting.time_mlx.function(f"ref generate {NUM_NEW_TOKENS} tokens takes")
    def generate_tokens(logits, model, cache):
        for _ in range(NUM_NEW_TOKENS - 1):
            y = mx.argmax(mx.softmax(logits), axis=-1)
            logits = ref_model(y[None], ref_cache)[:, -1, :]
        return mx.argmax(logits, axis=-1)

    ref_model, _ = mlx_lm.utils.load(MODEL_PATH)
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
    for i in range(5):
        print(f"run {i}")
        print("** Benchmark Ref Model **")
        mlx.core.metal.clear_cache()
        benchmark_ref_model()
        print("** Benchmark Base Model **")
        mlx.core.metal.clear_cache()
        benchmark_base_model()
