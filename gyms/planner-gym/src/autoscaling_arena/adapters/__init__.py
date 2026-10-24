# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler adapters — ``EngineProtocol`` implementations under test.

Each adapter is a closed-loop policy the harness drives every tick (it observes
fleet state and returns a fresh replica decision); we never replay a recorded
decision log.

The CloudAI adapters (MPC v3 and the offline-RL planner) pull in numpy and
torch, which a Planner-only environment has no reason to install. They are
exported lazily: ``from autoscaling_arena.adapters import CloudAIMPCV3Autoscaler``
works, but the heavy imports only happen on first access.
"""

from __future__ import annotations

import importlib
from typing import Any

from autoscaling_arena.adapters.keda import KedaAutoscaler
from autoscaling_arena.adapters.planner import planner_engine_factory
from autoscaling_arena.adapters.reactive import ReactiveAutoscaler
from autoscaling_arena.adapters.static import StaticAutoscaler

_LAZY_EXPORTS = {
    "CloudAIMPCV3Autoscaler": "autoscaling_arena.adapters.cloudai_mpc_v3",
    "CloudAIRLAutoscaleLSTM": "autoscaling_arena.adapters.cloudai_rl_lstm",
    "CloudAIRLDisaggAutoscaler": "autoscaling_arena.adapters.cloudai_rl_disagg",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "StaticAutoscaler",
    "ReactiveAutoscaler",
    "KedaAutoscaler",
    "CloudAIMPCV3Autoscaler",
    "CloudAIRLAutoscaleLSTM",
    "CloudAIRLDisaggAutoscaler",
    "planner_engine_factory",
]
