import os

import mlx_lm
import torch
import transformers

import recurrent_drafting

PYTORCH_BIN_REPO = "lmsys/vicuna-7b-v1.3"
SAFETENSOR_FP16_DIR = os.path.expanduser("~/m/vicuna-7b-v1.3-fp16")
SAFETENSOR_4BIT_DIR = os.path.expanduser("~/m/vicuna-7b-v1.3-4bit")


if not os.path.isdir(SAFETENSOR_4BIT_DIR):
    if not os.path.isdir(SAFETENSOR_FP16_DIR):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            PYTORCH_BIN_REPO, torch_dtype=torch.float16
        )
        tkner = transformers.AutoTokenizer.from_pretrained(PYTORCH_BIN_REPO)
        model.save_pretrained(SAFETENSOR_FP16_DIR, push_to_hub=False)
        tkner.save_pretrained(SAFETENSOR_FP16_DIR, push_to_hub=False)
    mlx_lm.convert(SAFETENSOR_FP16_DIR, mlx_path=SAFETENSOR_4BIT_DIR, quantize=True)


model, tokenizer = mlx_lm.load(SAFETENSOR_4BIT_DIR)
PROMPT = recurrent_drafting.chat.vicuna_prompt(
    ["Write a story about Einstein."], add_generation_prompt=True
)
print(PROMPT)
for t in mlx_lm.stream_generate(model, tokenizer, PROMPT, max_tokens=512):
    print(t, end="", flush=True)
print()
