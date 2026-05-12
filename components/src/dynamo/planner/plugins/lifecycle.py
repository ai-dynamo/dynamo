# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PluginLifecycle`` ABC.

Plugins implement ``Bootstrap`` + ``Reset`` so the orchestrator / mode
subclass can seed model state at startup and zero it on config reload.
``Snapshot`` / ``Restore`` are intentionally NOT part of this contract
(YAGNI) — add them back only when a real debugging-replay or
fast-recovery use case shows up.

Relation to stage methods (``Predict`` / ``Propose`` / ``Reconcile`` /
``Constrain``): those live on the plugin class too but are NOT declared
on this ABC — they're dispatched via ``PluginTransport.call(method, ...)``
and not every plugin implements every stage. ``Bootstrap`` / ``Reset``
are universal so making them abstract is fine.
"""

from __future__ import annotations

import abc

from dynamo.planner.plugins.types import (
    BootstrapRequest,
    BootstrapResponse,
    ResetRequest,
    ResetResponse,
)


class PluginLifecycle(abc.ABC):
    """Per-plugin lifecycle RPCs the orchestrator drives at startup + reset."""

    @abc.abstractmethod
    async def Bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        """Seed plugin state. Called once during orchestrator startup,
        typically with pre-deployment benchmark data (FPM / historical
        traffic). Implementations may no-op by returning
        ``BootstrapResponse(ok=True)`` — RECONCILE / CONSTRAIN plugins
        usually don't need state."""
        raise NotImplementedError

    @abc.abstractmethod
    async def Reset(self, request: ResetRequest) -> ResetResponse:
        """Zero plugin state. Called on config reload (v11 cache
        invalidation row 5) or after error recovery. Implementations
        may no-op by returning ``ResetResponse(ok=True)``."""
        raise NotImplementedError


__all__ = ["PluginLifecycle"]
