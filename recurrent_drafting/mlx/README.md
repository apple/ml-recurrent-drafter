# Recurrent Drafting in MLX

This project implements our paper https://arxiv.org/abs/2403.09919 in [MLX](https://github.com/ml-explore/mlx).

## Develop

It is recommended to develop MLX projects using macOS and Apple Silicon.  In a Conda environment with Python >3.10.1, running the following command to install this project.

```shell
pip install -e ".[dev,mlx]"
```

Run the following command to enable pre-commit:

```shell
pre-commit install
```

Many CI/CD run Linux on x86_64. MLX for Linux is only available on Conda.

```shell
conda install conda-forge::mlx
```

Run the following command to test.

```shell
pytest recurrent_drafting/mlx/
```

## Code Walkthrough

- `modeling_llama.py` implements the LLaMA model. This file is based on the LLaMA model implementation in MLX. We need this version for two reasons. (1) We use pre-allocated KV cache, similar to the PyTorch version in the parent directory; (2) the MLX version validates one token per call, whereas this version may verify several tokens.

- `modeling_drafter.py` implements the draft model, which includes the beam search-based drafting algorithm.

- `tree_attention.py` implements the dynamic tree attention algorithm, which removes duplicate common prefixes from the beam search result and reduces the number of tokens that must be verified by the LLM.

- `recurrent_drafting.py` implements the fast text decoding method, which is dependent on the previous three files.

- `autoregressive.py` implements the auto-regressive text decoding method for performance comparison. This files depends only on `modeling_llama.py` not `modeling_drafter.py` or `tree_attention.py`.

- `kv_cache.py` implements a pre-allocated KV cache.

- `attention.py` contains commonly-used routines for attention mask and bias.

- `time_mlx.py` contains Python decorators that measures the time used to run MLX graphs.

## Benchmark

Benchmark autoregression.

```shell
python recurrent_drafting/mlx/experiments/benchmark_autoregression.py \
 > /tmp/autoregression.csv
```

Benchmark recurrent drafting.

```shell
python recurrent_drafting/mlx/experiments/benchmark_recurrent_drafting.py \
  > /tmp/recurrent_drafting.csv
```

Run a script to load the two result files `/tmp/autoregression.csv` and `/tmp/recurrent_drafting.csv`, and draw the plot `/tmp/p.pdf`.

```shell
python recurrent_drafting/mlx/experiments/analyze_perf_data.py
```
