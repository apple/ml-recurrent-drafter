#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
set -e # exit on error

llm_name=lmsys/vicuna-7b-v1.3

n_layers=2
n_epochs=2
lr=0.001
weight_decay=0.0
current_time=$(date '+%Y-%m-%d_%H:%M:%S')
torchrun --nproc_per_node=8 recurrent_drafting/cmd/train.py \
    --llm_name_or_path $llm_name \
    --bf16 True \
    --output_dir $current_time \
    --num_train_epochs $n_epochs \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --drafter_predict_n_tokens 5 \
    --drafter_num_layers $n_layers \
    --rnn True \
    --phase train

torchrun --nproc_per_node=1 recurrent_drafting/cmd/train.py \
    --llm_name_or_path $llm_name \
    --drafter_name_or_path ${current_time}_redrafter_vicuna-7b-v1.3_n_5_lr_${lr}_layers_${n_layers} \
    --bf16 True \
    --output_dir /tmp/test_redrafter \
    --per_device_eval_batch_size 1 \
    --tf32 True \
    --model_max_length 2048 \
    --drafter_predict_n_tokens 5 \
    --drafter_num_layers $n_layers \
    --rnn True \
    --phase eval
