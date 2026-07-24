# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Squeeze-Evolve evolutionary test-time-scaling as a native Dynamo component.

Native port of the Squeeze-Evolve diversity loop (https://arxiv.org/abs/2604.07725):
runs the evolutionary loop as a chat model, routing candidate groups across model
tiers via one ``KvRouter`` per tier. See README.md.
"""

from dynamo.squeeze_evolve.orchestrator import (
    SqueezeEvolveConfig,
    SqueezeEvolveOrchestrator,
    Tier,
)

__all__ = ["SqueezeEvolveConfig", "SqueezeEvolveOrchestrator", "Tier"]
