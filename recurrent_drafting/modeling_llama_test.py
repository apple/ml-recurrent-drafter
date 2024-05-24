#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os

import torch
import transformers.models.llama.configuration_llama
import transformers.models.llama.modeling_llama
from transformers import AutoModelForCausalLM

import recurrent_drafting


@torch.inference_mode()
def _parity_check(
    model_hf: transformers.models.llama.modeling_llama.LlamaForCausalLM,
    model_my: recurrent_drafting.modeling_llama.LlamaForCausalLM,
):
    assert isinstance(model_hf, transformers.models.llama.modeling_llama.LlamaForCausalLM)
    assert isinstance(model_my, recurrent_drafting.modeling_llama.LlamaForCausalLM)

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], device=model_hf.device)  # in vocab_size
    pad_token_id = 0
    batch_size, input_length = input_ids.shape

    cache = recurrent_drafting.kv_cache.Cache(
        batch_size=batch_size,
        max_length=input_length + 1,
        n_layers=model_hf.config.num_hidden_layers,
        n_heads=model_hf.config.num_key_value_heads,
        head_dim=model_hf.config.hidden_size // model_hf.config.num_attention_heads,
        dtype=model_hf.dtype,
        device=input_ids.device,
    )

    logits_hf = model_hf(input_ids, attention_mask=input_ids != pad_token_id).logits
    attn_mask = recurrent_drafting.attention.causal_mask(
        input_ids != pad_token_id, input_ids.shape[1], input_ids.device
    )
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).expand(input_ids.shape)
    logits_my = model_my(
        input_ids,
        past_key_values=cache.sliced,
        attention_mask=attn_mask,
        position_ids=position_ids,
    ).logits
    torch.testing.assert_close(logits_hf, logits_my, atol=1e-4, rtol=1e-2)


def test_parity_llama_with_hf():
    # It takes too long time to run the model using CPU or the Metal GPU. Also, we need a CUDA GPU.
    # We don't want to download the large model when running CI and the CI may not have CUDA.
    if not torch.cuda.is_available():
        return
    cuda = torch.device("cuda")

    model = "lmsys/vicuna-7b-v1.3"
    model_hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(cuda)
    model_my = recurrent_drafting.modeling_llama.LlamaForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16
    ).to(cuda)
    _parity_check(model_hf, model_my)


def test_golden_test_scale_model():
    cfg = transformers.models.llama.configuration_llama.LlamaConfig(
        **{
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 8,
            "initializer_range": 0.02,
            "intermediate_size": 5,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 7,
        }
    )
    model_my = recurrent_drafting.modeling_llama.LlamaForCausalLM(cfg)
    model_dir = "recurrent_drafting/testdata/golden/llama/tiny"
    if not os.path.isfile(model_dir + "/model.safetensors"):
        os.makedirs(model_dir, exist_ok=True)
        model_my.save_pretrained(model_dir)

    model_hf = AutoModelForCausalLM.from_pretrained(model_dir)
    model_my = recurrent_drafting.modeling_llama.LlamaForCausalLM.from_pretrained(model_dir)
    _parity_check(model_hf, model_my)
