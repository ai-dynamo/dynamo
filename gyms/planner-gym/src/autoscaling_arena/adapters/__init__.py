# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler adapters — ``EngineProtocol`` implementations under test.

Each adapter is a closed-loop policy the harness drives every tick (it observes
fleet state and returns a fresh replica decision); we never replay a recorded
decision log.
"""

from autoscaling_arena.adapters.keda import KedaAutoscaler
from autoscaling_arena.adapters.planner import planner_engine_factory
from autoscaling_arena.adapters.reactive import ReactiveAutoscaler
from autoscaling_arena.adapters.static import StaticAutoscaler

__all__ = [
    "StaticAutoscaler",
    "ReactiveAutoscaler",
    "KedaAutoscaler",
    "planner_engine_factory",
]
