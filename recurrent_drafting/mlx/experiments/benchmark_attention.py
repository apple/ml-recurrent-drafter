import time

import mlx.core as mx
import mlx_lm.models.base
import mlx_recurrent_drafting


def benchmark_mlx_attention_mask(N: int, offset: int = 0):
    def func() -> mx.array:
        return mlx_lm.models.base.create_additive_causal_mask(N, offset)[None]

    for _ in range(10):  # warm up
        mx.eval(func())

    toi = time.perf_counter()
    for _ in range(100):
        mx.eval(func())
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / 100
    print(f"mlx_lm.models.base.create_additive_causal_mask takes {tpi:.3f} (ms)")


def benchmark_customized_attention_mask(N: int, query_len: int = 0):
    def func() -> mx.array:
        padding_mask = mx.ones((1, N)).astype(mx.int32)
        mask = mlx_recurrent_drafting.attention.causal_mask(padding_mask, query_len)
        bias = mlx_recurrent_drafting.attention.bias(mask, dtype=mx.int32)
        return bias

    for _ in range(10):  # warm up
        mx.eval(func())

    toi = time.perf_counter()
    for _ in range(100):
        mx.eval(func())
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / 100
    print(f"mlx_recurrent_drafting.attention.causal_mask takes {tpi:.3f} (ms)")


if __name__ == "__main__":
    print("prompt phase attn mask (100, 100)")
    benchmark_mlx_attention_mask(100, offset=0)
    benchmark_customized_attention_mask(100, 100)
    print("prompt phase attn mask (1000, 1000)")
    benchmark_mlx_attention_mask(1000, offset=0)
    benchmark_customized_attention_mask(1000, 1000)
    print("generation phase attn mask (1, 100)")
    benchmark_mlx_attention_mask(1, offset=99)
    benchmark_customized_attention_mask(100, 1)
    print("generation phase attn mask (1, 1000)")
    benchmark_mlx_attention_mask(1, offset=999)
    benchmark_customized_attention_mask(1000, 1)
