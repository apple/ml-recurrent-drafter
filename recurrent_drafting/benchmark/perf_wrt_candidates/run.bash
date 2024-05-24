#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# Usage:
#
# Following file comments in generate.py to download testdata and model.
#
# rm nohup.out
# nohup bash -c recurrent_drafting/benchmark/perf_wrt_candidates/run.bash &
# tail -f nohup.out

set -e # Any command fails, this script fails.
set -o pipefail # Any stage of a pipe fails, the command fails.

BATCH_SIZE=8
MAX_NUM_PROMPTS=32
MAX_BEAM_WIDTH=48
MAX_BEAM_LENGTH=16
TITLE="bs-$BATCH_SIZE-np-$MAX_NUM_PROMPTS-beam-$MAX_BEAM_WIDTH-len-$MAX_BEAM_LENGTH"
echo "$TITLE" # Instructions in README.md requires this to be the first line in nohup.out.

if [[ ! -d ~/m/artifacts ]]; then
    echo "Please download model following file comments of generate.py"
fi

function run_on_gpu() {
    local gpu_id=$1
    local beam_length=$2
    local beam_width=$3

    echo "On GPU $gpu_id, beam_length: $beam_length, beam_width $beam_width"
    python3 \
	-m recurrent_drafting.cmd.generate \
	--hf_llm=lmsys/vicuna-7b-v1.3 \
	--hf_drafter=$HOME/m/artifacts/torch_drafter/ \
	--hf_tokenizer=lmsys/vicuna-7b-v1.3 \
    --eval_mt_bench=True \
	--max_prompt_length=256 \
	--max_generation_length=1024 \
	--beam_width="$beam_width" \
	--beam_length="$beam_length" \
	--max_num_prompts=$MAX_NUM_PROMPTS \
	--batch_size=$BATCH_SIZE \
	--use_gpu="$gpu_id" \
	--dtype="bf16" \
	2>&1 \
	| tee --append /tmp/"$TITLE-gpu-$gpu_id".log
}

# Create an array of FIFO queues, one per GPU
for gpu_id in {0..7}; do
    rm -f /tmp/"$TITLE-gpu-$gpu_id".queue
    rm -f /tmp/"$TITLE-gpu-$gpu_id".log
    mkfifo /tmp/"$TITLE-gpu-$gpu_id".queue
done

for beam_length in $(eval echo {2..$MAX_BEAM_LENGTH}); do
    for beam_width in $(eval echo {1..$MAX_BEAM_WIDTH}); do
		hash=$(echo "$beam_length-$beam_width" | md5sum | cut -c 1-8)
		gpu_id=$((16#$hash % 8))
		echo Add task "$gpu_id" "$beam_length" "$beam_width" to /tmp/"$TITLE-gpu-$gpu_id".queue
		echo run_on_gpu "$gpu_id" "$beam_length" "$beam_width" \
			>> /tmp/"$TITLE-gpu-$gpu_id".queue &
    done
done

# Process the queues in parallel
for gpu_id in {0..7}; do
    echo "echo done on GPU_$gpu_id" >> /tmp/"$TITLE-gpu-$gpu_id".queue &
	while read -r task; do
		eval "$task"
	done < /tmp/"$TITLE-gpu-$gpu_id".queue &
done

# Wait for all background jobs to complete
wait

# Clean up the FIFO queues
for gpu_id in {0..7}; do
    rm /tmp/"$TITLE-gpu-$gpu_id".queue
done
