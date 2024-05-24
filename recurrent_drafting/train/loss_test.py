#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import pytest
import torch

from . import loss


@pytest.mark.parametrize(
    ["logits", "labels", "next_n", "top_k", "expected_loss"],
    [
        pytest.param(
            torch.tensor(
                [
                    [
                        [
                            [-4.2755, -4.0159, -5.4326],
                            [-7.3534, -5.2731, -4.7144],
                            [-6.3034, -5.6929, -4.1427],
                            [-4.4837, -4.3300, -6.1975],
                        ]
                    ],
                    [
                        [
                            [-5.6221, -2.1564, -5.9357],
                            [-2.7976, -5.3193, -4.0888],
                            [-4.9951, -4.7358, -6.5123],
                            [-4.8608, -4.8744, -6.2276],
                        ]
                    ],
                ]
            ),
            torch.tensor([[0, 1, 2, 1]]),
            2,
            3,
            torch.tensor(1.6388),
        )
    ],
)
def test_drafter_loss(logits, labels, next_n, top_k, expected_loss) -> None:
    loss_, _, _ = loss.drafter_loss(logits, labels, next_n, top_k)
    assert torch.allclose(loss_, expected_loss)
