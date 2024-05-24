# Performance of ReDrafter w.r.t. the Number of Beam Width and Beam Length

## Experiment Procedure

We need a modern CUDA GPU to run this benchmark.

1. Train the drafter model by following recurrent_drafting/train/README.md and save the model at, say, `~/m/artifacts/torch_drafter/`.

1. Run the script using `nohup` to explore various combinations of `beam_width` and `beam_length`:

   ```shell
   rm nohup.out
   nohup bash -c \
     recurrent_drafting/benchmark/perf_wrt_candidates/run.bash &
   ```

1. Run the following command to see the title of the job:

   ```shell
   TITLE=$(head -n 1 nohup.out)
   echo $TITLE
   ```

   It should be something like `bs-8-np-8-beam-4-len-4`.  The log files should be named `/tmp/$TITLE-gpu-[0-7].log`.

1. Monitor the job using the `jobs` command until completion. Run the following command to watch the progress:

   ```shell
   tail -f /tmp/$TITLE-gpu-[0-7].log
   ```

1. Convert the log into a CSV table with four columns (`beam_width`, `beam_length`, `average_num_accepted_tokens`, `tokens_per_sec`) using:

   ```shell
   cat /tmp/$TITLE-gpu-[0-7].log \
   | gawk -f recurrent_drafting/benchmark/perf_wrt_candidates/make_csv.awk \
   | sort -t, -k1,1n -k2,2n \
   > /tmp/$TITLE.csv
   ```

   Please install gawk if you haven't. It is critical to sort the lines of the CSV file by the first two columns; otherwise, the following step would not be able to create the x-y mesh.

1. Visualize the CSV file as an MP4 video:

   ```shell
   python3 recurrent_drafting/benchmark/perf_wrt_candidates/plot_csv.py \
   --csv_file=/tmp/$TITLE.csv \
   --animation_file=/tmp/$TITLE.mov
   ```

   Please install ffmpeg if you haven't.

## Running Benchmark in Parallel Across GPUs

The script `run.bash` executes eight simultaneous Python processes at any time, each using one of the eight GPUs on a host running `generate.py`, thus maximizing GPU resource utilization.

This script achieves this by:

- Creating eight FIFO queues, named `/tmp/*-gpu-$gpu_id`, where `gpu_id ∈ [0, 8)`.
- Generating a 2D grid for `beam_length ∈ [1,16]` and `beam_width ∈ [1,16]`.
- For each grid combination, it writes a command `run_on_gpu $gpu_id $beam_length $beam_width` to one of the FIFOs. The allocation of commands to `gpu_id` is via MD5 hashing.

Implementation notes:

- We use `md5sum | cut -c 1-8` for a safe conversion to an integer without exceeding the signed integer range.
- By wrapping up the invocation of `generate.py` in the bash function `run_on_gpu`, we simplify the commands written to the FIFOs.  Another benefit is that `run_on_gpu` merges the stderr and stdout and calls `tee --append` to save them to a log file, `/tmp/*-gpu-$gpu_id.log`, which is dedicated to `$gpu_id`.

The script is recommended to be run with `nohup` for lengthy executions.

To check load balancing:

```shell
cat nohup.out | grep 'Add task' | awk '{print $3}' | sort | uniq -c | sort -nr
```

To see the total number of tasks:

```shell
cat nohup.out | grep 'Add task' | wc -l
```

To see how many tasks have completed:

```shell
cat /tmp/$TITLE-gpu-[0-7].log | grep '^Tokens/second' | wc -l
```
