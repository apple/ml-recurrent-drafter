import os
import time

import mlx
import mlx.core as mx
import mlx_lm
import mlx_lm.models.base
import mlx_lm.models.llama

import mlx_recurrent_drafting

MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
NUM_NEW_TOKENS = 100
INPUT_IDS = mx.array([[1, 2, 3, 4, 5, 6]])


def load_ref_model() -> mlx.nn.Module:
    ref_model, _ = mlx_lm.utils.load(MODEL_PATH)
    return ref_model


def load_base_model() -> mlx_recurrent_drafting.modeling_llama.Model:
    base_model = mlx_recurrent_drafting.modeling_llama.load_model(MODEL_PATH)
    return base_model


def benchmark_base_model() -> None:
    base_model = load_base_model()
    # prepare cache
    cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size=1,
        max_length=INPUT_IDS.shape[1] + NUM_NEW_TOKENS,
        n_layers=base_model.args.num_hidden_layers,
        n_heads=base_model.args.num_key_value_heads,  # type:ignore
        head_dim=base_model.args.hidden_size // base_model.args.num_attention_heads,
        dtype=base_model.model.embed_tokens.weight.dtype,
    )

    # prompt
    toi = time.perf_counter()
    mask = mlx_recurrent_drafting.attention.causal_mask(
        mx.ones(shape=(1, INPUT_IDS.shape[1]), dtype=mx.bool_), INPUT_IDS.shape[1]
    )
    logits = base_model(INPUT_IDS, mx.arange(INPUT_IDS.shape[1])[None], mask, cache.sliced)[1][
        :, -1, :
    ]
    mx.eval(logits)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi)
    print(f"test prompt processing takes {tpi:.3f} (ms)")

    # generation
    toi = time.perf_counter()
    for i in range(NUM_NEW_TOKENS - 1):
        y = mx.argmax(mx.softmax(logits), axis=-1)
        mask = mlx_recurrent_drafting.attention.causal_mask(
            mx.ones(shape=(1, INPUT_IDS.shape[1] + i + 1), dtype=mx.bool_), 1
        )
        logits = base_model(y[None], mx.array([[INPUT_IDS.shape[1] + i + 1]]), mask, cache.sliced)[
            1
        ][:, -1, :]
    y = mx.argmax(logits, axis=-1)
    mx.eval(y)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / NUM_NEW_TOKENS
    print(f"test each new token takes {tpi:.3f} (ms)")


def benchmark_ref_model() -> None:
    ref_model = load_ref_model()
    # prepare cache
    head_dim = ref_model.args.hidden_size // ref_model.args.num_attention_heads
    ref_cache = [
        mlx_lm.models.base.KVCache(head_dim, ref_model.args.num_key_value_heads)
        for _ in range(ref_model.args.num_hidden_layers)
    ]

    # prompt
    toi = time.perf_counter()
    logits = ref_model(INPUT_IDS, cache=ref_cache)[:, -1, :]
    mx.eval(logits)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi)
    print(f"ref prompt processing takes {tpi:.3f} (ms)")

    # generation
    toi = time.perf_counter()
    for _ in range(NUM_NEW_TOKENS - 1):
        y = mx.argmax(mx.softmax(logits), axis=-1)
        logits = ref_model(y[None], ref_cache)[:, -1, :]
    y = mx.argmax(logits, axis=-1)
    mx.eval(y)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / NUM_NEW_TOKENS
    print(f"ref each new token takes {tpi:.3f} (ms)")


if __name__ == "__main__":
    print("** Benchmark Ref Model **")
    mlx.core.metal.clear_cache()
    benchmark_ref_model()
    print("** Benchmark Base Model **")
    mlx.core.metal.clear_cache()
    benchmark_base_model()
