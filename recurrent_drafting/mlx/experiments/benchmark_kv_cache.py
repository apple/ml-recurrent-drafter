import time

import mlx
import mlx.core as mx
import mlx_lm

from recurrent_drafting.mlx import kv_cache


def benchmark_kv_cache_cat(max_len: int):
    n_layer, n_kv_heads = 1, 32
    head_dim = 4096 // n_kv_heads
    batch_size = 1
    v = mx.ones([batch_size, n_kv_heads, 1, head_dim]).astype(mx.bfloat16)
    mx.eval(v)
    print(f"num iterations = {max_len}")

    # ref kv
    ref_kv_cache = mlx_lm.models.base.KVCache(head_dim, n_kv_heads)
    mx.eval(ref_kv_cache.keys, ref_kv_cache.values)
    tic = time.perf_counter()
    for _ in range(max_len):
        ref_kv_cache.update_and_fetch(keys=v, values=v)
    mx.eval(ref_kv_cache.keys, ref_kv_cache.values)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - tic) / max_len
    print(f"ref_kv Time per iteration {tpi:.3f} (ms)")

    # test kv
    test_kv_cache = kv_cache.Cache(
        batch_size, max_len, n_layer, n_kv_heads, head_dim, dtype=mx.bfloat16
    )
    for kc, vc in test_kv_cache.sliced:
        mx.eval(kc._cache)
        mx.eval(vc._cache)
    tic = time.perf_counter()
    for _ in range(max_len):
        test_kv_cache.sliced[0][0].cat(v)  # key
        test_kv_cache.sliced[0][1].cat(v)  # value
    for kc, vc in test_kv_cache.sliced:
        mx.eval(kc._cache)
        mx.eval(vc._cache)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - tic) / max_len
    print(f"test_kv Time per iteration {tpi:.3f} (ms)")


def benchmark_array_assignment():
    n_kv_heads = 32
    head_dim = 4096 // n_kv_heads

    # Low dims
    a = mx.zeros([1, n_kv_heads, 100, head_dim])
    b = mx.ones([1, n_kv_heads, 1, head_dim])
    mx.eval(a, b)
    tic = time.perf_counter()
    for i in range(100):
        a[..., i : i + 1, :] = b
        mx.eval(a)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - tic) / 100
    print(f"Low dims Time per iteration {tpi:.3f} (ms)")

    # High dims
    a = mx.zeros([1, 2, 1, n_kv_heads, 100, head_dim])
    b = mx.ones([1, n_kv_heads, 1, head_dim])
    mx.eval(a, b)
    tic = time.perf_counter()
    for i in range(100):
        a[0, 0, :, :, i : i + 1, :] = b
        mx.eval(a)
    toc = time.perf_counter()
    tpi = 1e3 * (toc - tic) / 100
    print(f"High dims Time per iteration {tpi:.3f} (ms)")


if __name__ == "__main__":
    print("** Comparing KV cache cat **")
    benchmark_kv_cache_cat(128)
    benchmark_kv_cache_cat(1024)
    benchmark_kv_cache_cat(4096)
    benchmark_kv_cache_cat(65536)
    mlx.core.metal.clear_cache()
    print("** Comparing array assignment **")
    benchmark_array_assignment()
