# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for components of modules."""

from .memory import Memory
from .mlp import MLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization, EmpiricalNormalizationDict
from algorithms.feature_extract.depth_net import Actor_net, Critic_net