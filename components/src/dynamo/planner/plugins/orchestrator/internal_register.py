# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-level convenience function wrapping orchestrator register_internal
(DEP-XXXX PR 5 sub-task 5-5).

The actual logic lives in ``PluginRegistryServer.register_internal``
(PR 3); this file exists so PR 6 builtin implementations and PR 7
``NativePlannerBase`` have a single import path — ``from
dynamo.planner.plugins.orchestrator.internal_register import
register_internal`` — consistent with the PR 5 spec in the DEP.
"""

from __future__ import annotations

from typing import Any, Optional

from dynamo.planner.plugins.orchestrator.orchestrator import (
    LocalPlannerOrchestrator,
)
from dynamo.planner.plugins.registry.types import RegisteredPlugin
from dynamo.planner.plugins.types import HoldPolicy


def register_internal(
    orchestrator: LocalPlannerOrchestrator,
    plugin_id: str,
    plugin_type: str,
    priority: int,
    instance: Any,
    *,
    execution_interval_seconds: float = 0.0,
    hold_policy: HoldPolicy = HoldPolicy.ACCEPT_WHEN_IDLE,
    is_builtin: bool = True,
    version: str = "builtin",
    needs: Optional[list[str]] = None,
) -> RegisteredPlugin:
    """Register an in-process plugin on the given orchestrator.

    Equivalent to ``orchestrator.register_internal(...)``; kept as a
    standalone function so the pr 5-5 spec's signature is directly
    callable by code that prefers a functional style.
    """
    return orchestrator.register_internal(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        priority=priority,
        instance=instance,
        execution_interval_seconds=execution_interval_seconds,
        hold_policy=hold_policy,
        is_builtin=is_builtin,
        version=version,
        needs=needs,
    )


__all__ = ["register_internal"]
