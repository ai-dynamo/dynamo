# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocalPlannerOrchestrator skeleton (DEP-XXXX PR 5).

This module composes the PR 1-4 pieces (proto/types, transport/clock,
registry/scheduler/circuit breaker, merge algorithms) into a single
orchestrator that drives the 4-stage plugin pipeline (PREDICT / PROPOSE
/ RECONCILE / CONSTRAIN) per tick and emits an EXECUTE decision.

Scope in this PR 5 session (partial):
  - Skeleton class + pipeline driver with M-1 / M-4 / M-7 enforcement
  - PipelineContext-native ``tick`` API (stub-plugin tests)
  - ``register_internal`` passthrough + ``load_in_process_plugins``
  - Concurrency / failure / timeout unit tests

Deferred:
  - 5-7 placeholder builtin plugins wrapping existing PSM mixin methods
  - 5-8 G3 behaviour-parity fixture replay
  - External ``tick(tick_input) -> PlannerEffects`` bridging — PR 7
    (NativePlannerBase integration will adapt between PR 1's
    ``PipelineContext`` and existing ``core/types.py``'s TickInput /
    PlannerEffects)
"""

from dynamo.planner.plugins.orchestrator.orchestrator import (
    LocalPlannerOrchestrator,
)
from dynamo.planner.plugins.orchestrator.pipeline import (
    ExecuteAction,
    PipelineOutcome,
    run_pipeline,
)

__all__ = [
    "LocalPlannerOrchestrator",
    "PipelineOutcome",
    "ExecuteAction",
    "run_pipeline",
]
