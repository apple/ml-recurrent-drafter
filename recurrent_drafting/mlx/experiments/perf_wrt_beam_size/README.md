# Performance of ReDrafter w.r.t. the Number of Beam Width and Beam Length

1. Apply the changes in [PR#88](https://github.pie.apple.com/ywang999828/mlx_recurrent_drafting/pull/88) and run the benchmark script to create the log.

```
python mlx_recurrent_drafting/experiments/benchmark_recurrent_drafting.py > bw_48_bl_10_output_100.log
```

1. Install dependencies

```
brew install gawk
brew install ffmpeg
```

1. Convert the log into a CSV table with four columns (`beam_width`, `beam_length`, `average_num_accepted_tokens`, `tokens_per_sec`) using:

```
cat mlx_recurrent_drafting/experiments/perf_wrt_beam_size/bw_48_bl_10_output_100.log \
| gawk -f mlx_recurrent_drafting/experiments/perf_wrt_beam_size/make_csv.awk \
| sort -t, -k1,1n -k2,2n > mlx_recurrent_drafting/experiments/perf_wrt_beam_size/ \
bw_48_bl_10_output_100.csv
```

1. Visualize the CSV file as an MP4 video:

```
python3 mlx_recurrent_drafting/experiments/perf_wrt_beam_size/plot_cvs.py \
> --csv_file=mlx_recurrent_drafting/experiments/perf_wrt_beam_size/bw_48_bl_10_output_100.csv \
> --animation_file=mlx_recurrent_drafting/experiments/perf_wrt_beam_size/bw_48_bl_10_output_100.mov
```
