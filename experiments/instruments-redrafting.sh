#!/usr/bin/env bash

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate mlx_recurrent_drafting

echo -e "What color is the sea?\nexit\n" \
  | python3 -m mlx_recurrent_drafting.cmd.generate \
    --hf_tokenizer=$HOME/m/vicuna-7b-v1.3-bf16 \
    --hf_llm=$HOME/m/vicuna-7b-v1.3-bf16 \
    --hf_drafter=$HOME/m/redrafter \
    --eval_mt_bench=False \
    --max_prompt_length=500 \
    --max_num_prompts=1 \
    --max_generation_length=100 \
    --beam_width=4 \
    --beam_length=5 \
    --greedy_search=True \
    --autoregression=False \
    --batch_size=1 \
    --dtype=bf16
