#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from . import generate


def test_load_mt_bench_prompt() -> None:
    assert 80 == len(
        [
            _
            for _ in generate.load_mt_bench_prompt(
                max_length=2048, max_num_prompts=-1  # read all prompts
            )
        ]
    )


_NUM_TEST_PROMPTS = 4


def test_batch() -> None:
    assert 1 == len(
        [
            _
            for _ in generate.batch(
                (f"${i}" for i in range(_NUM_TEST_PROMPTS)),
                batch_size=_NUM_TEST_PROMPTS - 1,  # so only one batch
            )
        ]
    )
    assert 2 == len(
        [
            _
            for _ in generate.batch(
                (f"${i}" for i in range(_NUM_TEST_PROMPTS)),
                batch_size=_NUM_TEST_PROMPTS // 2,
            )
        ]
    )
