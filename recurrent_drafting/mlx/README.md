# Recurrent Drafting in MLX

## Install

Git-clone the source code and run the following command:

```shell
cd ml-recurent-drafter
pip install -e ".[dev,mlx]"
```

## Test

Run the following command:

```shell
pytest recurrent_drafting/mlx/
```

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
