import tempfile
from typing import Tuple

import mlx.core as mx
import mlx.nn
import mlx_recurrent_drafting
import mlx_recurrent_drafting.attention
import mlx_recurrent_drafting.kv_cache
import mlx_recurrent_drafting.modeling_llama
import numpy
import pytest
import torch
import transformers

import recurrent_drafting


def test_rope() -> None:
    """Document how to use mlx.nn.RoPE
    TODO: we need to have position_ids supported"""
    past_kv_len = 32
    bz, n_heads, q_len, head_dim = 1, 1, 1, 4
    position_ids = torch.arange(past_kv_len, past_kv_len + q_len).unsqueeze(0).repeat(bz, q_len)
    q, k = torch.rand((bz, n_heads, q_len, head_dim)), torch.rand((bz, n_heads, q_len, head_dim))
    ref_rope = recurrent_drafting.modeling_llama.LlamaRotaryEmbedding(dim=head_dim)
    cos, sin = ref_rope(q.device, q.dtype, past_kv_len + q_len)
    ref_q_embed, ref_k_embed = recurrent_drafting.modeling_llama.apply_rotary_pos_emb(
        q, k, cos, sin, position_ids
    )

    mlx_rope = mlx.nn.RoPE(dims=head_dim)
    q_embed = mlx_rope(mx.array(q.numpy()), past_kv_len)
    k_embed = mlx_rope(mx.array(k.numpy()), past_kv_len)
    assert mx.allclose(q_embed, mx.array(ref_q_embed.numpy()), rtol=1e-3, atol=1e-2)
    assert mx.allclose(k_embed, mx.array(ref_k_embed.numpy()), rtol=1e-3, atol=1e-2)


@torch.inference_mode()
def _parity_check(
    ref_model: recurrent_drafting.modeling_llama.LlamaForCausalLM,
    mlx_model: mlx_recurrent_drafting.modeling_llama.Model,
    input_ids: numpy.ndarray,
    position_ids: numpy.ndarray,
    mask: numpy.ndarray,
) -> None:
    assert input_ids.shape == position_ids.shape
    batch_size, input_length = input_ids.shape
    ref_input_ids = torch.tensor(input_ids)
    ref_cache = recurrent_drafting.kv_cache.Cache(
        batch_size=batch_size,
        max_length=input_length + 1,
        n_layers=ref_model.config.num_hidden_layers,
        n_heads=ref_model.config.num_key_value_heads,
        head_dim=ref_model.config.hidden_size // ref_model.config.num_attention_heads,
        dtype=ref_model.dtype,
        device=ref_input_ids.device,
    )
    ref_attn_mask = torch.tensor(mask)
    ref_position_ids = torch.tensor(position_ids)
    ref_output = ref_model(
        ref_input_ids,
        past_key_values=ref_cache.sliced,
        attention_mask=ref_attn_mask,
        position_ids=ref_position_ids,
    )
    ref_logits = ref_output.logits
    ref_hidden_states = ref_output.hidden_states

    mlx_cache = mlx_recurrent_drafting.kv_cache.Cache(
        batch_size=batch_size,
        max_length=input_length + 1,
        n_layers=mlx_model.args.num_hidden_layers,
        n_heads=ref_model.config.num_key_value_heads,
        head_dim=mlx_model.args.hidden_size // mlx_model.args.num_attention_heads,
        dtype=mlx_model.model.embed_tokens.weight.dtype,
    )
    mlx_input_ids = mx.array(input_ids)
    mlx_mask = mx.array(mask)
    beam_len = (position_ids[0][-1] - position_ids[0][0] + 1).item()
    mlx_hidden_states, mlx_logits = mlx_model(mlx_input_ids, beam_len, mlx_mask, mlx_cache.sliced)
    assert mx.all(
        mx.allclose(mlx_hidden_states, mx.array(ref_hidden_states.numpy()), atol=1e-4, rtol=1e-4)
    )
    assert mx.all(mx.allclose(mlx_logits, mx.array(ref_logits.numpy()), atol=1e-4, rtol=1e-4))
    for i, (c1, c2) in enumerate(zip(mlx_cache.sliced, ref_cache.sliced)):
        q1, v1 = c1
        q2, v2 = c2
        assert q1.length == q2.length
        assert v1.length == v2.length
        assert mx.all(
            mx.allclose(q1._cache, mx.array(q2._cache.numpy()[i][0]), atol=1e-4, rtol=1e-4)
        )
        assert mx.all(
            mx.allclose(v1._cache, mx.array(v2._cache.numpy()[i][1]), atol=1e-4, rtol=1e-4)
        )


_test_llama_config = {
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


def create_test_models() -> Tuple[
    recurrent_drafting.modeling_llama.LlamaForCausalLM,
    mlx_recurrent_drafting.modeling_llama.Model,
]:
    recurrent_drafting.rng.seed_pytorch(123)
    mx.random.seed(123)
    ref_cfg = transformers.models.llama.configuration_llama.LlamaConfig(**_test_llama_config)
    ref_model = recurrent_drafting.modeling_llama.LlamaForCausalLM(ref_cfg)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ref_model.save_pretrained(tmpdirname)
        mlx_model = mlx_recurrent_drafting.modeling_llama.load_model(tmpdirname)
    return ref_model, mlx_model


@pytest.mark.parametrize(
    ["input_ids", "position_ids", "beam_shape"],
    [
        pytest.param(
            numpy.array([[1, 2, 3, 4, 5, 6]]),
            numpy.array([[0, 1, 2, 3, 4, 5]]),
            mlx_recurrent_drafting.modeling_drafter.BeamShape(1, 6),
        ),
        pytest.param(
            numpy.array([[1, 2, 3, 4, 5, 6]]),
            numpy.array([[0, 1, 2, 0, 1, 2]]),
            mlx_recurrent_drafting.modeling_drafter.BeamShape(2, 3),
        ),
        pytest.param(
            numpy.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]),
            numpy.array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]),
            mlx_recurrent_drafting.modeling_drafter.BeamShape(5, 3),
        ),
    ],
)
@pytest.mark.parametrize("device", [mx.gpu])  # mx.cpu fails the unit test
def test_parity_with_no_compression(
    input_ids: numpy.ndarray,
    position_ids: numpy.ndarray,
    beam_shape: mlx_recurrent_drafting.modeling_drafter.BeamShape,
    device: mx.Device,
) -> None:
    mx.set_default_device(device)
    ref_model, mlx_model = create_test_models()
    mask = numpy.array(
        mlx_recurrent_drafting.attention.causal_mask(
            mx.ones(shape=(1, beam_shape.length), dtype=mx.bool_), beam_shape.length
        )
    )
    eye = numpy.eye(beam_shape.width, dtype=int)
    mask = numpy.kron(eye, mask)
    _parity_check(ref_model, mlx_model, input_ids, position_ids, mask)
