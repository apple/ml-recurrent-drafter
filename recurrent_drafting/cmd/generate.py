#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""This script takes an recurrent drafter model and prompts from interactive inputs or
mt_bench(https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) dataset to run
the benchmark.

The script train.sh trains the recurrent drafter for vicuna-7b-v1.3.

Run the following command to benchmark the recurrent drafter algorithm using the default GPU.

python3 -m recurrent_drafting.cmd.generate \
    --hf_tokenizer=lmsys/vicuna-7b-v1.3 \
    --hf_llm=lmsys/vicuna-7b-v1.3 \
    --hf_drafter=$HOME/m/redrafter \
    --eval_mt_bench=True \
    --max_prompt_length=500 \
    --max_generation_length=2048 \
    --beam_width=45 \
    --beam_length=5 \
    --greedy_search=False \
    --batch_size=8 \
    --dtype=bf16

To use the first, say 64, evaluation data instances, use:
    --max_num_prompts=64

To specify a certain GPU, say, the fourth one, use:
    --use_gpu=3

To run generate.py on CPU, use a negative GPU index.
    --use_gpu=-1

"""

import os
from typing import Generator, List

import datasets
import torch
import tqdm
import transformers
from absl import app, flags

import recurrent_drafting
import recurrent_drafting.train
from recurrent_drafting import autoregressive
from recurrent_drafting.modeling_drafter import BeamShape
from recurrent_drafting.recurrent_drafting import SamplingArgs, SpecialTokens

FLAGS = flags.FLAGS


def load_llm(hf_llm: str, dtype: torch.dtype, device: torch.device) -> transformers.PreTrainedModel:
    cfg = transformers.AutoConfig.from_pretrained(hf_llm)
    assert cfg.model_type == "llama"
    return recurrent_drafting.modeling_llama.LlamaForCausalLM.from_pretrained(
        hf_llm, torch_dtype=dtype
    ).to(device)


def load_drafter(
    hf_drafter_dir: str, dtype: torch.dtype, device: torch.device
) -> transformers.PreTrainedModel:
    recurrent_drafting.modeling_drafter.register_auto_models()
    return transformers.AutoModel.from_pretrained(hf_drafter_dir, torch_dtype=dtype).to(device)


def load_tokenizer(hf_tokenizer_dir: str) -> tuple[transformers.PreTrainedTokenizer, SpecialTokens]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_tokenizer_dir, padding_side="left")
    return tokenizer, SpecialTokens(tokenizer.pad_token_id, tokenizer.eos_token_id)


VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


def instruct_vicuna_prompt(user_prompt: str) -> str:
    return f"{VICUNA_SYSTEM_PROMPT}USER: {user_prompt.lstrip()} ASSISTANT:"


def instruct_prompt(user_prompt: str, model_type: str = "vicuna") -> str:
    assert model_type == "vicuna"
    return instruct_vicuna_prompt(user_prompt)


def load_mt_bench_prompt(
    max_length: int, max_num_prompts: int, model_type: str = "vicuna"
) -> Generator[str, None, None]:
    eval_dataset = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    assert len(eval_dataset) >= max_num_prompts
    n_prompts = 0
    for row in eval_dataset:
        if max_num_prompts >= 0 and n_prompts >= max_num_prompts:
            break
        prompt = row["prompt"][0].strip()
        if len(prompt) > 2 and len(prompt) < max_length:
            n_prompts += 1
            yield instruct_prompt(prompt, model_type=model_type)
    if max_num_prompts >= 0:
        assert n_prompts == max_num_prompts


def batch(
    prompt_generator: Generator[str, None, None], batch_size: int
) -> Generator[List[str], None, None]:
    b: List[str] = []
    for prompt in prompt_generator:
        b.append(prompt)
        if len(b) >= batch_size:
            yield b
            b = []


def tokenize(
    batch_generator: Generator[List[str], None, None],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Generator[torch.Tensor, None, None]:
    for batch in batch_generator:
        yield tokenizer(batch, padding=True, return_tensors="pt").input_ids


def generate(
    ledger: recurrent_drafting.stats.Ledger,
    input_generator: Generator[torch.Tensor, None, None],
    model: recurrent_drafting.recurrent_drafting.RecurrentDrafting,
    max_length: int,
    beam_shape: BeamShape,
    sampling_args: SamplingArgs,
    special_tokens: SpecialTokens,
    autoregression: bool,
) -> Generator[Generator[torch.Tensor, None, None], None, None]:
    for input_ids in input_generator:
        input_ids = input_ids.to(model.llm.device)
        if autoregression:
            yield autoregressive.generate(
                model,
                input_ids,
                max_length,
                special_tokens,
            )
        else:
            yield model.generate(
                ledger,
                input_ids,
                max_length,
                beam_shape,
                sampling_args,
                special_tokens,
            )


def main(_: List[str]) -> None:
    recurrent_drafting.rng.seed_pytorch(FLAGS.rng_seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[FLAGS.dtype]
    device = torch.device(f"cuda:{FLAGS.use_gpu}" if int(FLAGS.use_gpu) >= 0 else "cpu")
    llm = load_llm(FLAGS.hf_llm, dtype, device)
    drafter = load_drafter(FLAGS.hf_drafter, dtype, device)
    tokenizer, special_tokens = load_tokenizer(os.path.expanduser(FLAGS.hf_tokenizer))
    model = recurrent_drafting.recurrent_drafting.RecurrentDrafting(llm=llm, drafter=drafter)
    ledger = recurrent_drafting.stats.Ledger()
    if FLAGS.eval_mt_bench:
        eval_mt_bench(ledger, model, tokenizer, special_tokens)
    else:
        chat(ledger, model, tokenizer, special_tokens)

    if not FLAGS.autoregression:
        ts, ps = recurrent_drafting.stats.summarize(ledger, FLAGS.batch_size, FLAGS.beam_length)
        title = f"beamwidth{FLAGS.beam_width}_beamlen{FLAGS.beam_length}"
        recurrent_drafting.stats.draw_table(title, ts)
        recurrent_drafting.stats.draw_figure(title, ps, "/tmp/title.png")


def model_type(model: transformers.PreTrainedModel) -> str:
    return {
        recurrent_drafting.modeling_llama.LlamaForCausalLM: "vicuna",
    }[type(model)]


def eval_mt_bench(
    ledger: recurrent_drafting.stats.Ledger,
    model: recurrent_drafting.recurrent_drafting.RecurrentDrafting,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens: SpecialTokens,
) -> None:
    for batch_output_generator in tqdm.tqdm(
        generate(
            ledger,
            tokenize(
                batch(
                    load_mt_bench_prompt(
                        max_length=FLAGS.max_prompt_length,
                        max_num_prompts=FLAGS.max_num_prompts,
                        model_type=model_type(model.llm),
                    ),
                    batch_size=FLAGS.batch_size,
                ),
                tokenizer,
            ),
            model,
            max_length=FLAGS.max_prompt_length + FLAGS.max_generation_length,
            beam_shape=BeamShape(FLAGS.beam_width, FLAGS.beam_length),
            sampling_args=SamplingArgs(FLAGS.temperature, FLAGS.greedy_search),
            special_tokens=special_tokens,
            autoregression=FLAGS.autoregression,
        ),
    ):
        batch_output_token_ids = next(batch_output_generator)
        for batch_output_token_ids in batch_output_generator:
            pass
        print(tokenizer.batch_decode(batch_output_token_ids))


def chat(
    ledger: recurrent_drafting.stats.Ledger,
    model: recurrent_drafting.recurrent_drafting.RecurrentDrafting,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens: SpecialTokens,
) -> None:
    while True:
        user_input = input("chat> ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting REPL.")
            break

        prompt = instruct_prompt(user_input, model_type=model_type(model.llm))
        input_generator = tokenize(batch((prompt for _ in range(1)), batch_size=1), tokenizer)

        output_generator = next(
            generate(
                ledger,
                input_generator,
                model,
                max_length=FLAGS.max_prompt_length + FLAGS.max_generation_length,
                beam_shape=BeamShape(FLAGS.beam_width, FLAGS.beam_length),
                sampling_args=SamplingArgs(FLAGS.temperature, FLAGS.greedy_search),
                special_tokens=special_tokens,
                autoregression=FLAGS.autoregression,
            )
        )
        print(prompt, end="")
        for output_token_ids in output_generator:
            decoded_output = tokenizer.decode(output_token_ids.to("cpu")[0])
            print(decoded_output[len(prompt) :], end="")
            prompt = decoded_output
        print()


def define_flags() -> None:
    flags.DEFINE_string("hf_tokenizer", "lmsys/vicuna-7b-v1.3", "The tokenizer used by hf_llm.")
    flags.DEFINE_string("hf_llm", "lmsys/vicuna-7b-v1.3", "The base LLM model.")
    flags.DEFINE_string("hf_drafter", None, "The recurrent drafter of hf_llm.", required=True)
    flags.DEFINE_bool("eval_mt_bench", False, "Use mt_bench dataset as the input.")
    flags.DEFINE_integer("max_prompt_length", 50, "Only prompts shorter than this value are used.")
    flags.DEFINE_integer("max_num_prompts", -1, "Use the first n prompts in mt_bench dataset.")
    flags.DEFINE_integer("max_generation_length", 60, "The length of output text for each prompt.")
    flags.DEFINE_integer("batch_size", 1, "Batch size")
    flags.DEFINE_integer("beam_length", -1, "Beam length of drafter model beam search")
    flags.DEFINE_integer("beam_width", 10, "Number of candidates of drafter model beam search")
    flags.DEFINE_bool("autoregression", False, "Turn on to generate with auto-regression.")
    flags.DEFINE_bool("greedy_search", False, "Greedy search for ReDrafter.")
    flags.DEFINE_float("temperature", 1.0, "Sampling randomness for ReDrafter.")
    flags.DEFINE_integer("rng_seed", 123, "RNG seed.")
    flags.DEFINE_integer("use_gpu", 0, "If negative, use CPU")
    flags.DEFINE_enum(
        "dtype", "bf16", enum_values=["fp32", "fp16", "bf16"], help="Save the model into dtype."
    )


if __name__ == "__main__":
    define_flags()
    app.run(main)
