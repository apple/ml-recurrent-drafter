# Copyright © 2024 Apple Inc.
"""This script takes an recurrent drafter model and prompts from interactive inputs or
mt_bench(https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) dataset to run
the benchmark.

Run the following command to benchmark the recurrent drafter algorithm using the default GPU.

python3 -m mlx_recurrent_drafting.cmd.generate \
    --hf_tokenizer=wangkuiyi/vicuna-7b-v1.3  \
    --hf_llm=wangkuiyi/vicuna-7b-v1.3 \
    --hf_drafter=$HOME/m/redrafter \
    --eval_mt_bench=True \
    --max_prompt_length=500 \
    --max_num_prompts=1 \
    --max_generation_length=2048 \
    --beam_width=45 \
    --beam_length=5 \
    --greedy_search=False \
    --batch_size=1 \
    --dtype=bf16

To use the first, say 64, evaluation data instances, use:
    --max_num_prompts=64

To specify a certain GPU, say, the fourth one, use:
    --use_gpu=3

To run generate.py on CPU, use a negative GPU index.
    --use_gpu=-1

"""

import json
import os
from typing import Generator, List

import datasets
import mlx.core as mx
import mlx.nn
import tqdm
from absl import app, flags
from mlx_lm.utils import get_model_path
from sentencepiece import SentencePieceProcessor

import mlx_recurrent_drafting
import mlx_recurrent_drafting.modeling_llama
import mlx_recurrent_drafting.recurrent_drafting
from mlx_recurrent_drafting.modeling_drafter import BeamShape

FLAGS = flags.FLAGS


def load_llm(llm_dir: str, dtype: mx.Dtype) -> mlx.nn.Module:
    model_path = get_model_path(llm_dir)
    config = {}
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
    assert config["model_type"] == "llama"
    llm = mlx_recurrent_drafting.modeling_llama.load_model(model_path)
    llm.setdefault(dtype)
    return llm


def load_drafter(
    drafter_dir: str, dtype: mx.Dtype
) -> mlx_recurrent_drafting.modeling_drafter.Drafter:
    drafter = mlx_recurrent_drafting.modeling_drafter.load_model(drafter_dir)
    drafter.set_dtype(dtype)
    return drafter


def load_tokenizer(tokenizer_dir: str) -> SentencePieceProcessor:
    model_path = get_model_path(tokenizer_dir)
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return tokenizer, mlx_recurrent_drafting.recurrent_drafting.SpecialTokens(
        tokenizer.pad_id(), tokenizer.eos_id()
    )


def tokenize(
    batch_generator: Generator[List[str], None, None],
    tokenizer: SentencePieceProcessor,
) -> Generator[mx.array, None, None]:
    for batch in batch_generator:
        assert len(batch) == 1
        yield mx.array([[tokenizer.bos_id()] + tokenizer.encode(batch)[0]])


def generate(
    input_generator: Generator[mx.array, None, None],
    model: mlx_recurrent_drafting.recurrent_drafting.ReDrafterModel,
    max_length: int,
    beam_shape: BeamShape,
    sampling_args: mlx_recurrent_drafting.recurrent_drafting.SamplingArgs,
    special_tokens: mlx_recurrent_drafting.recurrent_drafting.SpecialTokens,
) -> Generator[Generator[mx.array, None, None], None, None]:
    for input_ids in input_generator:
        yield model.generate(
            input_ids,
            max_length,
            beam_shape,
            sampling_args,
            special_tokens,
        )


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


def main(_: List[str]) -> None:
    # Only supports batch_size=1
    assert FLAGS.batch_size == 1
    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[FLAGS.dtype]
    device = mx.Device(mx.gpu, FLAGS.use_gpu) if int(FLAGS.use_gpu) >= 0 else mx.Device(mx.cpu)
    mx.set_default_device(device)
    llm = load_llm(os.path.expanduser(FLAGS.hf_llm), dtype)
    drafter = load_drafter(os.path.expanduser(FLAGS.hf_drafter), dtype)
    tokenizer, special_tokens = load_tokenizer(os.path.expanduser(FLAGS.hf_tokenizer))
    model = mlx_recurrent_drafting.recurrent_drafting.ReDrafterModel(llm=llm, drafter=drafter)
    if FLAGS.eval_mt_bench:
        prompt_generator = load_mt_bench_prompt(
            max_length=FLAGS.max_prompt_length,
            max_num_prompts=FLAGS.max_num_prompts,
            model_type=model_type(model.llm),
        )
        eval_mt_bench(model, tokenizer, special_tokens, prompt_generator)
    else:
        chat(model, tokenizer, special_tokens)


def model_type(model: mlx.nn.Module) -> str:
    return {mlx_recurrent_drafting.modeling_llama.Model: "vicuna"}[type(model)]  # type:ignore


def eval_mt_bench(
    model: mlx_recurrent_drafting.recurrent_drafting.ReDrafterModel,
    tokenizer: SentencePieceProcessor,
    special_tokens: mlx_recurrent_drafting.recurrent_drafting.SpecialTokens,
    prompt_generator: Generator[str, None, None],
) -> None:
    for batch_output_generator in tqdm.tqdm(
        generate(
            tokenize(
                batch(
                    prompt_generator,
                    batch_size=FLAGS.batch_size,
                ),
                tokenizer,
            ),
            model,
            max_length=FLAGS.max_prompt_length + FLAGS.max_generation_length,
            beam_shape=BeamShape(FLAGS.beam_width, FLAGS.beam_length),
            sampling_args=mlx_recurrent_drafting.recurrent_drafting.SamplingArgs(
                FLAGS.temperature, FLAGS.greedy_search
            ),
            special_tokens=special_tokens,
        ),
    ):
        batch_output_token_ids = next(batch_output_generator)
        for batch_output_token_ids in batch_output_generator:
            mx.eval(batch_output_token_ids)
            pass
        print(tokenizer.decode([t.item() for t in batch_output_token_ids[0]]))


def chat(
    model: mlx_recurrent_drafting.recurrent_drafting.ReDrafterModel,
    tokenizer: SentencePieceProcessor,
    special_tokens: mlx_recurrent_drafting.recurrent_drafting.SpecialTokens,
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
                input_generator,
                model,
                max_length=FLAGS.max_prompt_length + FLAGS.max_generation_length,
                beam_shape=BeamShape(FLAGS.beam_width, FLAGS.beam_length),
                sampling_args=mlx_recurrent_drafting.recurrent_drafting.SamplingArgs(
                    FLAGS.temperature, FLAGS.greedy_search
                ),
                special_tokens=special_tokens,
            )
        )
        print(prompt, end="")
        for output_token_ids in output_generator:
            mx.eval(output_token_ids)
            decoded_output = tokenizer.decode([t.item() for t in output_token_ids[0]])
            print(decoded_output[len(prompt) :], end="")
            prompt = decoded_output
        print()


def define_flags() -> None:
    flags.DEFINE_string("hf_tokenizer", "lmsys/vicuna-7b-v1.3", "The tokenizer used by hf_llm.")
    flags.DEFINE_string("hf_llm", "lmsys/vicuna-7b-v1.3", "The base LLM model.")
    flags.DEFINE_string("hf_drafter", None, "The recurrent drafter of hf_llm.", required=True)
    flags.DEFINE_bool("eval_mt_bench", False, "Use mt_bench dataset as the input.")
    flags.DEFINE_integer("max_prompt_length", 50, "Only prompts shorter than this value are used.")
    flags.DEFINE_integer("max_num_prompts", -1, "Use the first n prompts in the prompt dataset.")
    flags.DEFINE_integer("max_generation_length", 60, "The length of output text for each prompt.")
    flags.DEFINE_integer("batch_size", 1, "Batch size")
    flags.DEFINE_integer("beam_length", -1, "Beam length of drafter model beam search")
    flags.DEFINE_integer("beam_width", 10, "Number of candidates of drafter model beam search")
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
