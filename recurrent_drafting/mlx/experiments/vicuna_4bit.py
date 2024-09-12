import os

import mlx.core as mx
import mlx_lm
import torch
import transformers

import recurrent_drafting
import recurrent_drafting.mlx

PYTORCH_BIN_REPO = "lmsys/vicuna-7b-v1.3"
SAFETENSOR_FP16_DIR = os.path.expanduser("~/m/vicuna-7b-v1.3-fp16")
SAFETENSOR_4BIT_DIR = os.path.expanduser("~/m/vicuna-7b-v1.3-4bit")

NUM_NEW_TOKENS = 200

# QUESTION = "When was Apple Inc. founded?"
QUESTION = "Write a story about Einstein."


def quantize_model_if_not():
    if not os.path.isdir(SAFETENSOR_4BIT_DIR):
        if not os.path.isdir(SAFETENSOR_FP16_DIR):
            model = transformers.AutoModelForCausalLM.from_pretrained(
                PYTORCH_BIN_REPO, torch_dtype=torch.float16
            )
            tkner = transformers.AutoTokenizer.from_pretrained(PYTORCH_BIN_REPO)
            model.save_pretrained(SAFETENSOR_FP16_DIR, push_to_hub=False)
            tkner.save_pretrained(SAFETENSOR_FP16_DIR, push_to_hub=False)
        mlx_lm.convert(SAFETENSOR_FP16_DIR, mlx_path=SAFETENSOR_4BIT_DIR, quantize=True)


def mlx_generate():
    model, tokenizer = mlx_lm.load(SAFETENSOR_4BIT_DIR)
    prompt = recurrent_drafting.chat.vicuna_prompt([QUESTION], add_generation_prompt=True)
    for t in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=NUM_NEW_TOKENS):
        print(t, end="", flush=True)
    print()


def our_generate():
    model = recurrent_drafting.mlx.modeling_llama.load_model(SAFETENSOR_4BIT_DIR)
    tokenizer = transformers.AutoTokenizer.from_pretrained(SAFETENSOR_FP16_DIR)
    p = recurrent_drafting.chat.vicuna_prompt([QUESTION], add_generation_prompt=True)
    ids = tokenizer(p).input_ids
    prompt = mx.array([ids])

    cache = recurrent_drafting.mlx.kv_cache.Cache(
        batch_size=1,
        max_length=prompt.shape[1] + NUM_NEW_TOKENS,
        n_layers=model.args.num_hidden_layers,
        n_heads=model.args.num_key_value_heads or model.args.num_attention_heads,
        head_dim=model.args.hidden_size // model.args.num_attention_heads,
        dtype=mx.float16,
    )

    mask = recurrent_drafting.mlx.attention.causal_mask(
        mx.ones(shape=(1, prompt.shape[1]), dtype=mx.bool_), prompt.shape[1]
    )
    logits = model(prompt, prompt.shape[1], mask, cache.sliced)[1][:, -1, :]
    for i in range(NUM_NEW_TOKENS - 1):
        y = mx.argmax(mx.softmax(logits), axis=-1)
        print(tokenizer._convert_id_to_token(y.item()), end="")
        mask = recurrent_drafting.mlx.attention.causal_mask(
            mx.ones(shape=(1, prompt.shape[1] + i + 1), dtype=mx.bool_), 1
        )
        logits = model(y[None], 1, mask, cache.sliced)[1][:, -1, :]
    y = mx.argmax(logits, axis=-1)
    print(tokenizer._convert_id_to_token(y.item()))


quantize_model_if_not()
print("MLX generate:")
mlx_generate()
print("Our generate:")
our_generate()
