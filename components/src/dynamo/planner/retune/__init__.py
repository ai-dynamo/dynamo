# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner retune: periodic replay-based tuning of router and planner parameters."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from dynamo.planner.retune.domains import (
    PlannerDomain,
    expand_planner_domain,
    expand_router_domain,
    grid_points,
    grid_size,
    required_history_seconds,
)

# The controller, sweep and window reader are re-exported lazily (PEP 562).
# ``PlannerConfig`` validation imports ``retune.domains``, and must not load
# the native bindings, AISimulate or the replay stack that these modules need.
if TYPE_CHECKING:
    from dynamo.planner.retune.controller import RetuneController, RetuneTickResult
    from dynamo.planner.retune.sweep import (
        SweepRequest,
        SweepResult,
        evaluate_candidate,
        run_sweep,
    )
    from dynamo.planner.retune.window import (
        HistoryFile,
        WindowFile,
        read_trace_history,
        read_trace_window,
        trace_origin_ms,
    )

_LAZY_EXPORTS = {
    "RetuneController": "dynamo.planner.retune.controller",
    "RetuneTickResult": "dynamo.planner.retune.controller",
    "SweepRequest": "dynamo.planner.retune.sweep",
    "SweepResult": "dynamo.planner.retune.sweep",
    "evaluate_candidate": "dynamo.planner.retune.sweep",
    "run_sweep": "dynamo.planner.retune.sweep",
    "HistoryFile": "dynamo.planner.retune.window",
    "WindowFile": "dynamo.planner.retune.window",
    "read_trace_history": "dynamo.planner.retune.window",
    "read_trace_window": "dynamo.planner.retune.window",
    "trace_origin_ms": "dynamo.planner.retune.window",
}

__all__ = [
    "HistoryFile",
    "PlannerDomain",
    "RetuneController",
    "RetuneTickResult",
    "SweepRequest",
    "SweepResult",
    "WindowFile",
    "evaluate_candidate",
    "expand_planner_domain",
    "expand_router_domain",
    "grid_points",
    "grid_size",
    "read_trace_history",
    "read_trace_window",
    "required_history_seconds",
    "run_sweep",
    "trace_origin_ms",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
