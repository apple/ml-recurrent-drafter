#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

"""
Training arguments:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""

import math
import multiprocessing
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from transformers import Trainer

from recurrent_drafting.configuration_drafter import DrafterConfig
from recurrent_drafting.modeling_drafter import Drafter
from recurrent_drafting.train import data
from recurrent_drafting.train.loss import drafter_loss
from recurrent_drafting.train.model import ReDrafter


class ReDrafterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]:
                The computed loss, optionally with model outputs.
        """
        next_n = self.args.drafter_predict_n_tokens
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], next_n=next_n
        )  # [drafter_predict_n_tokens, batch_size, seq_len, vocab_size]
        # [batch_size, seq_len]
        loss, log, eval_log = drafter_loss(
            logits, inputs["labels"], next_n, self.args.drafter_top_k
        )
        self.log(log)
        return (loss, eval_log) if return_outputs else loss


@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    drafter_name_or_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. "
            "Sequences will be right padded (and possibly truncated)."
        },
    )
    drafter_predict_n_tokens: int = field(
        default=5,
        metadata={"help": "Drafter predicts k extra tokens."},
    )
    drafter_top_k: int = field(
        default=5,
        metadata={"help": "Drafter top k accuracy for each token."},
    )
    drafter_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for the drafter."},
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={"help": "Include inputs for metrics."},
    )
    phase: str = field(
        default="train",
        metadata={"help": "train or eval"},
    )
    rnn: bool = field(
        default=False,
        metadata={"help": "Include rnn in drafter."},
    )


def get_tokenizer(model_args, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def generate_drafter_config_from_base(llm, training_args):
    return DrafterConfig(
        vocab_size=llm.lm_head.weight.shape[0],
        hidden_size=llm.lm_head.weight.shape[-1],
        exit_dim=2 * llm.lm_head.weight.shape[-1],
        num_draft_layers=training_args.drafter_num_layers,
        rnn=training_args.rnn,
    )


def get_compute_metrics(training_args):
    predict_n_tokens = training_args.drafter_predict_n_tokens

    def compute_metrics(all_preds):
        return_val = {}
        for i in range(predict_n_tokens):
            for k in range(1, training_args.drafter_top_k + 1):
                return_val[f"redrafter{i}_top{k}"] = np.mean(
                    all_preds.predictions[i * predict_n_tokens + k - 1]
                )
        return return_val

    return compute_metrics


def train(model_args, training_args):
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    # Load data
    train_dataset = datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train").map(
        lambda x: data.sharegpt_record_to_vicuna_training_instance(x, tokenizer),
        num_proc=multiprocessing.cpu_count(),
    )
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(model_args.llm_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # Load and freeze the base model
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.llm_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    any((setattr(param, "requires_grad", False) for param in llm.base_model.parameters()))
    drafter_config = generate_drafter_config_from_base(llm, training_args)
    drafter = Drafter(drafter_config)
    redrafter = ReDrafter(llm, drafter)
    # Format output dir
    training_args.output_dir = (
        f"{training_args.output_dir}"
        f"_redrafter_{model_args.llm_name_or_path.split('/')[-1]}"
        f"_n_{training_args.drafter_predict_n_tokens}"
        f"_lr_{training_args.learning_rate}"
        f"_layers_{training_args.drafter_num_layers}"
    )
    trainer = ReDrafterTrainer(
        model=redrafter,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
    )
    trainer.train(
        resume_from_checkpoint=bool(
            list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        )
    )
    # Save ReDrafter
    drafter.save_pretrained(training_args.output_dir)


def eval(model_args, training_args):
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    # Load data
    eval_dataset = (
        datasets.load_dataset("tatsu-lab/alpaca_eval", split="eval")
        .map(data.convert_alpaca_to_sharegpt, num_proc=multiprocessing.cpu_count())
        .map(
            lambda x: data.sharegpt_record_to_vicuna_training_instance(x, tokenizer),
            num_proc=multiprocessing.cpu_count(),
        )
    )
    # Load ReDrafter
    redrafter = ReDrafter.from_pretrained(
        model_args.llm_name_or_path,
        model_args.drafter_name_or_path,
        torch_dtype=torch.float16,
    )
    # Start trainer
    trainer = ReDrafterTrainer(
        model=redrafter,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset,
    )
    trainer.evaluate()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    assert training_args.phase in ["train", "eval"]
    run = train if training_args.phase == "train" else eval
    run(model_args, training_args)
