# Performance w.r.t. Batch Size

We need a GPU host to run this benchmark to run the experiment:

```shell
rm nohup.out
nohup bash -c ./recurrent_drafting/benchmark/perf_wrt_batch_size/run.bash &
tail -f nohup.out
```

The following command summarize the benchmark result in a CSV file.

```shell
./recurrent_drafting/benchmark/perf_wrt_batch_size/make_csv.bash > /tmp/a.csv
```

## Troubleshooting

If you use a host with too early version of CUDA, you may need to downgrade PyTorch.

```
pip list | grep torch # Check if it is 2.1.1
pip uninstall torch torchvision torchaudio # Uninstall the too new version
pip install torch==2.0.1 torchvision==0.15.2
```
