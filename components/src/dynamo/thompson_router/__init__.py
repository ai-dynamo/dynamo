# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thompson Sampling KV-aware router for Dynamo.

Layers Thompson Sampling (Beta bandits + LinTS contextual bandits) on top of
Dynamo's native KvRouter (PyO3) for learning-based worker selection.

Usage:
    # Standalone service
    python -m dynamo.thompson_router --endpoint dynamo.worker.generate

    # In-process (import directly)
    from dynamo.thompson_router import KvThompsonRouter, RoutingDecision
"""

from dynamo.thompson_router.config import load_config
from dynamo.thompson_router.hints import extract_hints
from dynamo.thompson_router.learners import (
    BetaLearner,
    LatencyTracker,
    LinTSLearner,
    PendingDecisions,
)
from dynamo.thompson_router.router import KvThompsonRouter, RoutingDecision

__all__ = [
    "KvThompsonRouter",
    "RoutingDecision",
    "BetaLearner",
    "LinTSLearner",
    "LatencyTracker",
    "PendingDecisions",
    "extract_hints",
    "load_config",
]
