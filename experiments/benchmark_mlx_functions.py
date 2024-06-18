import mlx.core as mx
import mlx.nn

from mlx_recurrent_drafting import attention, time_mlx


def benchmark_sdpa(
    batch_size: int, num_heads: int, q_len: int, kv_len: int, head_dim: int, dtype: mx.Dtype
) -> None:
    q = mx.random.uniform(shape=(batch_size, num_heads, q_len, head_dim)).astype(dtype)
    k = mx.random.uniform(shape=(batch_size, num_heads, kv_len, head_dim)).astype(dtype)
    v = mx.random.uniform(shape=(batch_size, num_heads, kv_len, head_dim)).astype(dtype)
    m = attention.bias(
        attention.causal_mask(
            padding_mask=mx.ones(shape=(1, kv_len)).astype(mx.int32), query_len=q_len
        ),
        dtype=dtype,
    )
    mx.eval(q, k, v, m)
    # warm up
    for _ in range(10):
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=m))

    iteration = 32

    @time_mlx.function(f"run {iteration} mlx sdpa takes")
    def time_sdpa(q, k, v):
        for _ in range(iteration):
            mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=m))

    time_sdpa(q, k, v)


def benchmark_linear_projection(
    input_dims: int, output_dims: int, q_len: int, dtype: mx.Dtype
) -> None:
    proj = mlx.nn.Linear(input_dims, output_dims, bias=False)
    proj.set_dtype(dtype)
    x = mx.random.uniform(shape=(1, q_len, input_dims)).astype(dtype)
    mx.eval(proj, x)
    # warm up
    for _ in range(10):
        mx.eval(proj(x))

    iteration = 32

    @time_mlx.function(f"run {iteration} mlx linear projection takes")
    def time_linear(proj, x):
        for _ in range(iteration):
            mx.eval(proj(x))

    return time_linear(proj, x)


if __name__ == "__main__":
    mx.random.seed(123)
    for bw in range(10, 510, 10):
        for bl in range(10, 110, 10):
            time_mlx.ledger.reset()
            print(f"benchmark beam_width {bw} beam_length {bl}")
            q_len = bw * bl
            # benchmark_sdpa(1, 32, q_len, q_len + 100, 128, mx.bfloat16)
            benchmark_linear_projection(4096, 4096, q_len, mx.bfloat16)
            time_mlx.ledger.print_summary()

    # bw 350 bl 50 for sdpa OOM
    # safe:
    #   - bw 350 bl 40 for sdpa
    #   - bw 500 bl 100 for linear projection
