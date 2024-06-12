import time

import torch
from recurrent_drafting import attention, rng


def benchmark_sdpa(
    batch_size: int, num_heads: int, q_len: int, kv_len: int, head_dim: int, dtype: torch.dtype
) -> None:
    device = torch.device("cuda")
    q = torch.randn(size=(batch_size, num_heads, q_len, head_dim)).to(dtype).to(device)
    k = torch.randn(size=(batch_size, num_heads, kv_len, head_dim)).to(dtype).to(device)
    v = torch.randn(size=(batch_size, num_heads, kv_len, head_dim)).to(dtype).to(device)
    m = attention.bias(
        attention.causal_mask(
            padding_mask=torch.ones(size=(1, kv_len)).to(torch.int32).to(device),
            query_len=q_len,
            device=device,
        ),
        dtype=dtype,
    )
    # warm up
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=m)
    tic = time.perf_counter()
    iteration = 32
    for _ in range(iteration):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=m)
    timing = 1e3 * (time.perf_counter() - tic)
    print(f"{iteration} x torch sdpa take {timing: .3f} (ms)")


def benchmark_linear_projection(
    input_dims: int, output_dims: int, q_len: int, dtype: torch.dtype
) -> None:
    device = torch.device("cuda")
    q_proj = torch.nn.Linear(input_dims, output_dims, bias=False).to(dtype).to(device)
    k_proj = torch.nn.Linear(input_dims, output_dims, bias=False).to(dtype).to(device)
    v_proj = torch.nn.Linear(input_dims, output_dims, bias=False).to(dtype).to(device)
    x = torch.randn(size=(1, q_len, input_dims)).to(dtype).to(device)
    # warm up
    for _ in range(5):
        q_proj(x)
        k_proj(x)
        v_proj(x)
    tic = time.perf_counter()
    iteration = 32
    for _ in range(iteration):
        q_proj(x)
        k_proj(x)
        v_proj(x)
    timing = 1e3 * (time.perf_counter() - tic)
    print(f"{iteration} x torch proj take {timing: .3f} (ms)")


if __name__ == "__main__":
    rng.seed_pytorch(123)
    for bw in range(1, 50):
        for bl in range(1, 10):
            print(f"benchmark beam_width {bw} beam_length {bl}")
            q_len = bw * bl
            benchmark_sdpa(1, 32, q_len, q_len + 100, 128, torch.bfloat16)
            benchmark_linear_projection(4096, 4096, q_len, torch.bfloat16)
