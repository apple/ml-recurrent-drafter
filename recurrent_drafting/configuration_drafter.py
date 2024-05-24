#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Any, Dict

import transformers


class DrafterConfig(transformers.PretrainedConfig):
    # https://huggingface.co/docs/transformers/custom_models#writing-a-custom-model
    model_type = "recurrent_drafting_drafter"

    def __init__(
        self,
        vocab_size: int = -1,
        hidden_size: int = -1,
        exit_dim: int = -1,
        num_draft_layers: int = -1,
        num_lookback_tokens: int = -1,
        rnn: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.exit_dim = exit_dim
        self.num_draft_layers = num_draft_layers
        self.num_lookback_tokens = num_lookback_tokens
        self.rnn = rnn
