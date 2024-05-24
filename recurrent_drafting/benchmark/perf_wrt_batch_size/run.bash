#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# Usage:
# rm nohup.out
# nohup bash -c recurrent_drafting/benchmark/perf_wrt_batch_size/run.bash &
# tail -f nohup.out

rm /tmp/batch_size-*.log

for i in {0..6}; do
    python3 -m  recurrent_drafting.cmd.generate \
	--hf_llm=lmsys/vicuna-7b-v1.3 \
	--hf_drafter=$HOME/m/artifacts/torch_drafter/ \
	--hf_tokenizer=lmsys/vicuna-7b-v1.3 \
	--eval_mt_bench=True \
	--max_prompt_length=1024 \
	--max_generation_length=1024 \
	--beam_width=8 \
	--beam_length=5 \
	--max_num_prompts=32 \
	--batch_size=$((2**i)) \
	--greedy_search=False \
	--dtype="bf16" \
	--use_gpu=$i \
	2>&1 \
	| tee --append /tmp/batch_size-$((2**i)).log &
done

wait
