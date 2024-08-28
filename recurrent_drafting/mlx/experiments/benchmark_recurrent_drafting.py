import os
from typing import Dict, Tuple

import mlx.core as mx
import transformers
from mlx_recurrent_drafting import (
    kv_cache,
    modeling_drafter,
    modeling_llama,
    recurrent_drafting,
    time_mlx,
)

MODEL_PATH = os.path.expanduser("~/m/vicuna-7b-v1.3-bf16")
DRAFTER_PATH = os.path.expanduser("~/m/redrafter")
"""<s> A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user\'s questions.
USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting
cultural experiences and must-see attractions. ASSISTANT:
"""
# Disable the multi-line reformatting by black.
# fmt: off
PROMPT = mx.array(
    [
        [
            529, 29879, 29958, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082
            , 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568
            , 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 3831
            , 852, 385, 3033, 6751, 9850, 12618, 1400, 1048, 263, 7786, 17487, 304, 26901
            , 29875, 29892, 12141, 292, 16375, 27482, 322, 1818, 29899, 4149, 19650, 1953
            , 29889, 319, 1799, 9047, 13566, 29901
        ]
    ]
)
# fmt: on

_test_recurrent_drafting_model_singleton: Dict[str, recurrent_drafting.ReDrafterModel] = {}


def _get_recurrent_drafting_model(model_type: str = "llama") -> recurrent_drafting.ReDrafterModel:
    global _test_recurrent_drafting_model_singleton
    while _test_recurrent_drafting_model_singleton.get(model_type) is None:
        llm = modeling_llama.load_model(MODEL_PATH)
        llm.set_dtype(mx.bfloat16)  # Set dtype bfloat16 for testing
        drafter = modeling_drafter.load_model(DRAFTER_PATH)
        drafter.set_dtype(mx.bfloat16)  # Set dtype bfloat16 for testing
        _test_recurrent_drafting_model_singleton[model_type] = recurrent_drafting.ReDrafterModel(
            llm, drafter
        )
    return _test_recurrent_drafting_model_singleton[model_type]


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
    print(f"total steps = {step}")
    return output_token_ids


def benchmark_recurrent_drafting(
    beam_width: int, beam_length: int, dtype: mx.Dtype, max_length: int, greedy: bool
):
    print(
        f"benchmark_recurrent_drafting beam_width={beam_width}, beam_length={beam_length}, \
            dtype={dtype}, max_length={max_length}, greedy={greedy}"
    )
    mx.random.seed(123)
    model = _get_recurrent_drafting_model()
    model.llm.set_dtype(dtype)
    model.drafter.set_dtype(dtype)
    mx.eval(model.llm.parameters())
    mx.eval(model.drafter.parameters())
    return recurrent_drafting_generate(
        model,
        input_ids=PROMPT,
        max_length=PROMPT.shape[1] + max_length,
        beam_shape=modeling_drafter.BeamShape(beam_width, beam_length),
        sampling_args=recurrent_drafting.SamplingArgs(1.0, greedy),
    )


def benchmark_verify_candidates(beam_width: int, beam_length: int):
    print(f"benchmark_verify_candidates beam_width={beam_width}, beam_length={beam_length}")
    mx.random.seed(123)
    model = _get_recurrent_drafting_model()
    mx.eval(model.llm.parameters())
    mx.eval(model.drafter.parameters())
    # prepare inputs
    beams = mx.random.randint(
        shape=(1, beam_width, beam_length), low=1, high=model.llm.args.vocab_size
    )
    cache = kv_cache.Cache(
        batch_size=1,
        max_length=PROMPT.shape[1] + beams.shape[1] * beams.shape[2],
        n_layers=model.llm.args.num_hidden_layers,
        n_heads=model.llm.args.num_key_value_heads or model.llm.args.num_attention_heads,
        head_dim=model.llm.args.hidden_size // model.llm.args.num_attention_heads,
        dtype=model.llm.lm_head.weight.dtype,
    )
    pad_token_id = 0
    for k_cache, v_cache in cache.sliced:
        mx.eval(k_cache._cache)
        mx.eval(v_cache._cache)
    # populate the kv cache
    mx.eval(
        recurrent_drafting._comprehend_prompt(
            model.llm,
            PROMPT,
            cache,
            sampling_args=recurrent_drafting.SamplingArgs(1.0, True),
            pad_token_id=0,
        )
    )

    @time_mlx.function("time verify_candidates")
    def _verify_candidates_wrapper(*args) -> Tuple[mx.array, mx.array]:
        return recurrent_drafting._verify_candidates(*args)

    _verify_candidates_wrapper(model.llm, PROMPT, beams, cache, pad_token_id)


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("wangkuiyi/vicuna-7b-v1.3")
    ledger = time_mlx.ledger
    for i in range(2):
        print(f"RUN {i}")
        for greedy in (True, False):
            for dtype in (mx.float16, mx.bfloat16):
                for max_length in (100, 200, 400, 800):
                    for bw in (1, 2):
                        for bl in (2, 4, 6):
                            print(
                                "--------------------------START OF BENCHMARK\
                                    --------------------------"
                            )
                            ledger.reset()
                            tokens = benchmark_recurrent_drafting(bw, bl, dtype, max_length, greedy)
                            print(f"num total tokens={tokens.shape[1]}")
                            print(tokenizer.decode(tokens[0].tolist()))
                            print()
                            ledger.print_summary()
                            print()
