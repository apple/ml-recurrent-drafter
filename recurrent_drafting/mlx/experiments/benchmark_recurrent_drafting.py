import itertools
import os

import mlx.core as mx
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
) -> mx.array:
    output_generator = model.generate(
        input_ids,
        max_length,
        beam_shape,
        sampling_args,
        special_tokens,
    )
    output_token_ids = next(output_generator)
    step = 0
    for output_token_ids in output_generator:
        mx.eval(output_token_ids)
        step += 1
    print(f"num_steps: {step}")
    return output_token_ids


def benchmark_recurrent_drafting(
    model,
    prompt: mx.array,
    beam_width: int,
    beam_length: int,
    dtype: mx.Dtype,
    max_length: int,
    greedy: bool,
):
    print(
        f"beam_width: {beam_width}\nbeam_length: {beam_length}\n"
        + f"dtype: {dtype}\nmax_length: {max_length}\ngreedy: {greedy}"
    )
    mx.random.seed(123)
    # model = _get_recurrent_drafting_model(llm_dir, drafter_dir)
    model.llm.set_dtype(dtype)
    model.drafter.set_dtype(dtype)
    mx.eval(model.llm.parameters())
    mx.eval(model.drafter.parameters())
    return recurrent_drafting_generate(
        model,
        input_ids=prompt,
        max_length=prompt.shape[1] + max_length,
        beam_shape=modeling_drafter.BeamShape(beam_width, beam_length),
        sampling_args=recurrent_drafting.SamplingArgs(1.0, greedy),
    )


def timed_call(ledger: time_mlx.Ledger) -> float:
    assert len(ledger.records) == 1  # only one call to recurrent_drafting_generate
    return ledger.records[0].timing[0]  # in ms


if __name__ == "__main__":
    llm_dir = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
    drafter_dir = os.path.expanduser("~/m/redrafter")
    tokenizer_path = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")

    model = recurrent_drafting.ReDrafterModel(
        llm=modeling_llama.load_model(llm_dir), drafter=modeling_drafter.load_model(drafter_dir)
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    new_ids = tokenizer(
        "A chat between a curious user and an artificial intelligence assistant. "
        + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        + "USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting "
        + "cultural experiences and must-see attractions. ASSISTANT:"
    ).input_ids
    prompt = mx.array([new_ids])  # batch size = 1
    ledger = time_mlx.ledger
    # for greedy in (True, False):
    #     for dtype in [mx.float16]:
    #         for max_length in [200]:
    #             for bw in (1, 2, 4):
    #                 for bl in (2, 4):
    for greedy, dtype, max_length, bw, bl in itertools.product(
        [True, False], [mx.float16, mx.bfloat16], [200], [1, 2, 4], [2, 4]
    ):

        print("=" * 80)
        ledger.reset()
        tokens = benchmark_recurrent_drafting(model, prompt, bw, bl, dtype, max_length, greedy)
        print(f"num_ total_tokens: {tokens.shape[1]}")
        print(f"parse_and_generation_time: {timed_call(ledger)}")
        print(tokenizer.decode(tokens[0].tolist()))
