# Copyright © 2024 Apple Inc.
import os
import tempfile
from typing import Any, Dict, Tuple

import mlx
import mlx.core as mx
import numpy
import pytest
import recurrent_drafting
import torch

import mlx_recurrent_drafting

VOCAB_SIZE = 128
HIDDEN_SIZE = 16
_test_recurrent_drafting_config: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "exit_dim": 24,
    "num_draft_layers": 1,
}


@pytest.mark.parametrize(
    ["batch_size", "beam_width", "vocab_size"], [pytest.param(2, 3, 4), pytest.param(7, 13, 14)]
)
@pytest.mark.parametrize("device", [mx.gpu, mx.cpu])
def test_maintain_logits(
    batch_size: int, beam_width: int, vocab_size: int, device: mx.Device
) -> None:
    if not mx.metal.is_available() and device == mx.gpu:
        return
    mx.set_default_device(device)
    numpy.random.seed(123)
    logits = numpy.random.rand(batch_size, beam_width, vocab_size)
    ref_logits = recurrent_drafting.modeling_drafter.maintain_logits(torch.tensor(logits))
    mlx_logits = mlx_recurrent_drafting.modeling_drafter.maintain_logits(mx.array(logits))
    assert mx.allclose(mlx_logits, mx.array(ref_logits.numpy()), atol=1e-2, rtol=1e-2).item()


@pytest.mark.parametrize(
    ["batch_size", "beam_width", "vocab_size"],
    [pytest.param(2, 3, 4), pytest.param(7, 13, 14), pytest.param(27, 43, 45)],
)
@pytest.mark.parametrize("device", [mx.gpu, mx.cpu])
def test_warp_logits(batch_size: int, beam_width: int, vocab_size: int, device: mx.Device) -> None:
    if not mx.metal.is_available() and device == mx.gpu:
        return
    mx.set_default_device(device)
    numpy.random.seed(123)
    logits = numpy.random.rand(batch_size, beam_width, vocab_size)
    ref_logits = recurrent_drafting.modeling_drafter.warp_logits(torch.tensor(logits))
    mlx_logits = mlx_recurrent_drafting.modeling_drafter.warp_logits(mx.array(logits))
    assert mx.allclose(mlx_logits, mx.array(ref_logits.numpy()), atol=1e-2, rtol=1e-2).item()


def load_test_models(
    rnn: bool,
) -> Tuple[
    recurrent_drafting.modeling_drafter.Drafter, mlx_recurrent_drafting.modeling_drafter.Drafter
]:
    ref_cfg = recurrent_drafting.configuration_drafter.DrafterConfig(
        **_test_recurrent_drafting_config, rnn=rnn
    )
    ref_model = recurrent_drafting.modeling_drafter.Drafter(ref_cfg)
    mlx_args = mlx_recurrent_drafting.modeling_drafter.ModelArgs(
        **_test_recurrent_drafting_config, rnn=rnn
    )
    mlx_model = mlx_recurrent_drafting.modeling_drafter.Drafter(mlx_args)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ref_model.save_pretrained(tmpdirname)
        path = os.path.join(tmpdirname, "model.safetensors")
        mlx_model.load_weights(path)
    mlx_model.assert_valid()
    return ref_model, mlx_model


@torch.inference_mode()
@pytest.mark.parametrize("rnn", [True, False])
@pytest.mark.parametrize(
    ["batch_size", "beam_width", "beam_length"],
    [pytest.param(1, 2, 2), pytest.param(2, 6, 8), pytest.param(7, 9, 14)],
)
@pytest.mark.parametrize("device", [mx.gpu, mx.cpu])
def test_drafter_beam_search(
    rnn: bool, batch_size: int, beam_width: int, beam_length: int, device: mx.Device
) -> None:
    if not mx.metal.is_available() and device == mx.gpu:
        return
    mx.set_default_device(device)
    recurrent_drafting.rng.seed_pytorch(123)
    ref_model, mlx_model = load_test_models(rnn)
    init_token = numpy.random.randint(low=0, high=VOCAB_SIZE, size=(batch_size,))
    prompt_state = numpy.random.rand(batch_size, HIDDEN_SIZE).astype(numpy.float32)
    embeddings_weight = numpy.random.rand(VOCAB_SIZE, HIDDEN_SIZE).astype(numpy.float32)

    torch_embeddings = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    torch_embeddings.weight = torch.nn.Parameter(torch.tensor(embeddings_weight))
    ref_beams, ref_log_probs = ref_model.beam_search_candidates(
        torch.tensor(prompt_state),
        torch.tensor(init_token),
        torch_embeddings,
        beam_shape=recurrent_drafting.modeling_drafter.BeamShape(beam_width, beam_length),
    )

    mlx_embeddings = mlx.nn.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    mlx_embeddings.load_weights([("weight", mx.array(embeddings_weight))])
    mlx_beams, mlx_log_probs = mlx_model.beam_search_candidates(
        mx.array(prompt_state),
        mx.array(init_token),
        mlx_embeddings,
        beam_shape=mlx_recurrent_drafting.modeling_drafter.BeamShape(beam_width, beam_length),
    )
    # Sort the beams by token id because the order of tokens from mx.argpartition is undefined
    # https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html#mlx-core-argpartition
    mlx_output_pairs = [
        sorted(
            [(mlx_beams[i][j], mlx_log_probs[i][j]) for j in range(beam_width)],
            key=lambda x: x[0].tolist(),
        )
        for i in range(batch_size)
    ]
    ref_output_pairs = [
        sorted(
            [(ref_beams[i][j], ref_log_probs[i][j]) for j in range(beam_width)],
            key=lambda x: x[0].tolist(),
        )
        for i in range(batch_size)
    ]
    for i in range(batch_size):
        for j in range(beam_width):
            assert mx.all(mlx_output_pairs[i][j][0] == mx.array(ref_output_pairs[i][j][0].numpy()))
            assert mx.allclose(
                mlx_output_pairs[i][j][1],
                mx.array(ref_output_pairs[i][j][1].numpy()),
                atol=1e-2,
                rtol=1e-2,
            ).item()
