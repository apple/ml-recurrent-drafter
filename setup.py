#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from setuptools import find_packages, setup

setup(
    name="recurrent_drafting",
    version="0.1.0",
    packages=find_packages(include=["recurrent_drafting", "recurrent_drafting.*"]),
    python_requires=">=3.10.1, <3.11",  # Python 3.10.0 has a bug
    install_requires=[
        "transformers",
        "torch",
        "absl-py",
        "pandas",
        "tabulate",
        "jsonlines",
        "sentencepiece",
        "protobuf",
    ],
    extras_require={
        "train": [
            "fschat",
            "accelerate",
            "scipy",
            "datasets",
        ],
        "mlx": [
            "mlx",
            "mlx-lm",
            "absl-py",
            "datasets",
        ],
        "dev": [
            "pre-commit",
            "mypy",  # Install mypy in native env instead of pre-commit's to parse PyTorch code.
            "pytest",
            "pytest-xdist",
        ],
    },
)
