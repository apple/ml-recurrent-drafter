#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import tempfile
from typing import Any, Dict

import transformers

import recurrent_drafting

_test_recurrent_drafting_config: Dict[str, Any] = {
    "vocab_size": 128,
    "hidden_size": 16,
    "exit_dim": 24,
    "num_draft_layers": 1,
}


def test_register_drafter_auto_models() -> None:
    recurrent_drafting.modeling_drafter.register_auto_models()
    dft_cfg = recurrent_drafting.configuration_drafter.DrafterConfig(
        **_test_recurrent_drafting_config
    )
    drafter_model = recurrent_drafting.modeling_drafter.Drafter(dft_cfg)
    with tempfile.TemporaryDirectory() as tmpdirname:
        drafter_model.save_pretrained(tmpdirname)
        new_model = transformers.AutoModel.from_pretrained(tmpdirname)
        assert isinstance(new_model, recurrent_drafting.modeling_drafter.Drafter)
        new_model.assert_valid()
