#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy
import torch


def seed_pytorch(random_seed: int) -> None:
    """Seed the RNGs for PyTorch-based programs. c.f.
    https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba"""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(random_seed)
