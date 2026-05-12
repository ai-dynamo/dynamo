# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-level convenience function wrapping orchestrator
``register_internal``.

The actual logic lives in ``PluginRegistryServer.register_internal``;
this file exists so builtin plugin implementations and
``NativePlannerBase`` have a single import path — ``from
dynamo.planner.plugins.orchestrator.internal_register import
register_internal``.
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
    standalone function so the signature is directly callable by code
    that prefers a functional style.
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
