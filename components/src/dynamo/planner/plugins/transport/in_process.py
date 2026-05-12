# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process transport: direct Python call.

Used by:
- Builtin plugins — registered via ``register_internal``, zero RPC overhead
- ``in_process`` user plugins — loaded by config
- Replay / unit tests — first-class production transport, NOT a test fallback
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.errors import (
    PluginCallError,
    PluginTimeoutError,
    PluginUnknownMethodError,
)


class InProcessTransport(PluginTransport):
    """Direct Python invocation; zero serialization cost.

    Supports both ``async def`` and sync (``def``) plugin methods; sync
    methods are dispatched via ``asyncio.to_thread`` to avoid blocking
    the orchestrator event loop.

    **⚠ Sync plugin red line** (see transport/README.md): sync plugin
    methods MUST NOT do blocking IO (HTTP, file, ``time.sleep > 100ms``);
    otherwise thread pool exhaustion will block the orchestrator. If
    your plugin needs IO, write it as ``async def``.
    """

    def __init__(
        self,
        plugin_id: str,
        instance: Any,
        timeout_seconds: float = 5.0,
    ) -> None:
        if instance is None:
            raise ValueError(
                f"InProcessTransport(plugin_id={plugin_id!r}): instance must not be None"
            )
        if timeout_seconds <= 0:
            raise ValueError(
                f"InProcessTransport(plugin_id={plugin_id!r}): "
                f"timeout_seconds must be positive, got {timeout_seconds}"
            )
        self.plugin_id = plugin_id
        self.endpoint = f"inproc://{plugin_id}"
        self.timeout_seconds = timeout_seconds
        self._instance = instance
        self._closed = False

    async def call(self, method: str, request: Any) -> Any:
        # Method lookup
        fn = getattr(self._instance, method, None)
        if fn is None or not callable(fn):
            raise PluginUnknownMethodError(
                f"method {method!r} not found on plugin {self.plugin_id!r} "
                f"(type {type(self._instance).__name__})",
                plugin_id=self.plugin_id,
                method=method,
            )

        # Dispatch — async vs sync
        try:
            if inspect.iscoroutinefunction(fn):
                coro = fn(request)
            else:
                # Sync plugin: run in default thread pool to avoid blocking event loop
                coro = asyncio.to_thread(fn, request)
            return await asyncio.wait_for(coro, self.timeout_seconds)
        except asyncio.TimeoutError as e:
            raise PluginTimeoutError(
                f"plugin {self.plugin_id!r} method {method!r} exceeded "
                f"timeout_seconds={self.timeout_seconds}",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e
        except PluginCallError:
            # Plugin already raised a typed call error; propagate as-is
            raise
        except Exception as e:
            raise PluginCallError(
                f"plugin {self.plugin_id!r} method {method!r} raised "
                f"{type(e).__name__}: {e}",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e

    async def close(self) -> None:
        # In-process plugin instance lifecycle is owned by orchestrator;
        # transport has nothing to release. Idempotent flag set for safety.
        self._closed = True


__all__ = ["InProcessTransport"]
