#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Usage:
1. Download the LLM, the drafter, and the tokenizer to $HOME/m

2. Run this script

   python recurrent_drafting/mlx/experiments/benchmark_recurrent_drafting.py \
    > /tmp/recurrent_drafting.csv

   Or, if you want to watch the output in the terminal, you need unbuffer, which comes with expect.

   brew install expect

   unbuffer python recurrent_drafting/mlx/experiments/benchmark_recurrent_drafting.py \
    | tee /tmp/recurrent_drafting.csv
"""
import itertools
import os
import sys
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import pandas
import transformers

from recurrent_drafting.mlx import (
    modeling_drafter,
    modeling_llama,
    recurrent_drafting,
    time_mlx,
)


@time_mlx.function("benchmark_recurrent_drafting.recurrent_drafting_generate")
def recurrent_drafting_generate(
    model: recurrent_drafting.ReDrafterModel,
    input_ids: mx.array,
    max_length: int,
    beam_shape: modeling_drafter.BeamShape,
    sampling_args: recurrent_drafting.SamplingArgs,
    special_tokens: recurrent_drafting.SpecialTokens = recurrent_drafting.SpecialTokens(0, 1),
) -> Tuple[mx.array, int]:
    output_generator = model.generate(
        input_ids,
        max_length,
        beam_shape,
        sampling_args,
        special_tokens,
    )
    output_token_ids = next(output_generator)
    steps = 0
    for output_token_ids in output_generator:
        mx.eval(output_token_ids)
        steps += 1
    return output_token_ids, steps


def timed_call(ledger: time_mlx.Ledger) -> float:
    assert len(ledger.records) == 1  # only one call to recurrent_drafting_generate
    return ledger.records[0].timing[0]  # in ms


if __name__ == "__main__":
    MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
    DRAFTER_PATH = os.path.expanduser("~/m/redrafter")
    TOKENIZER_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")

    # BUG: This piece of code utterly puzzled me. If I dare to move the instiantiation of model
    # after prompt, some loop steps will generate many <unk>'s.
    model = recurrent_drafting.ReDrafterModel(
        llm=modeling_llama.load_model(MODEL_PATH), drafter=modeling_drafter.load_model(DRAFTER_PATH)
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    new_ids = tokenizer(
        "A chat between a curious user and an artificial intelligence assistant. "
        + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        + "USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting "
        + "cultural experiences and must-see attractions. ASSISTANT:"
    ).input_ids
    prompt = mx.array([new_ids])  # batch size = 1

    table: List[Dict[str, Any]] = []
    ledger = time_mlx.ledger
    for run, greedy, dtype, max_length, beam_width, beam_length in itertools.product(
        range(2), [True, False], [mx.float16, mx.bfloat16], [200], [1, 2, 3, 4], [2, 3, 4, 5]
    ):
        ledger.reset()
        mx.random.seed(123)
        r = {
            "run": run,
            "beam_width": beam_width,
            "beam_length": beam_length,
            "dtype": dtype,
            "nmax_length": max_length,
            "greedy": greedy,
        }
        print("=" * 80, "\n", r)
        model.llm.set_dtype(dtype)
        model.drafter.set_dtype(dtype)
        mx.eval(model.llm.parameters())
        mx.eval(model.drafter.parameters())

        tokens, steps = recurrent_drafting_generate(
            model,
            input_ids=prompt,
            max_length=prompt.shape[1] + max_length,
            beam_shape=modeling_drafter.BeamShape(beam_width, beam_length),
            sampling_args=recurrent_drafting.SamplingArgs(1.0, greedy),
        )
        rr = {
            "steps": steps,
            "prompt_length": prompt.shape[1],
            "prompt_and_generated_length": tokens.shape[1],
            "comprehension_and_generation_time": timed_call(ledger),
        }
        print(rr, file=sys.stderr)
        r.update(rr)
        table.append(r)
        print(tokenizer.decode(tokens[0].tolist()), file=sys.stderr)
    pandas.DataFrame(table).to_csv(sys.stdout, index=False)
