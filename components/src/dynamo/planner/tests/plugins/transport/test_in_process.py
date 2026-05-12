# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``InProcessTransport``."""

from __future__ import annotations

import asyncio

import pytest

from dynamo.planner.plugins.transport import (
    InProcessTransport,
    PluginCallError,
    PluginTimeoutError,
    PluginUnknownMethodError,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _AsyncEchoPlugin:
    async def Predict(self, req):
        return f"predict-{req}"

    async def Propose(self, req):
        return f"propose-{req}"


class _SyncEchoPlugin:
    def Bootstrap(self, req):
        return f"bootstrap-{req}"


class _RaisingPlugin:
    async def Propose(self, req):
        raise ValueError("plugin internal failure")


class _SlowPlugin:
    async def Predict(self, req):
        await asyncio.sleep(2.0)
        return req


# ----- construction validation -----


def test_construct_requires_instance():
    with pytest.raises(ValueError, match="instance must not be None"):
        InProcessTransport("p", None)  # type: ignore[arg-type]


def test_construct_requires_positive_timeout():
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        InProcessTransport("p", _AsyncEchoPlugin(), timeout_seconds=0.0)


def test_endpoint_uses_inproc_scheme():
    t = InProcessTransport("p", _AsyncEchoPlugin())
    assert t.endpoint == "inproc://p"
    assert t.plugin_id == "p"


# ----- happy path -----


@pytest.mark.asyncio
async def test_async_plugin_call():
    t = InProcessTransport("a", _AsyncEchoPlugin(), timeout_seconds=1.0)
    assert await t.call("Predict", "x") == "predict-x"
    assert await t.call("Propose", "y") == "propose-y"


@pytest.mark.asyncio
async def test_sync_plugin_dispatched_via_to_thread():
    """Sync plugin methods must work (dispatched via asyncio.to_thread)."""
    t = InProcessTransport("s", _SyncEchoPlugin(), timeout_seconds=1.0)
    assert await t.call("Bootstrap", "data") == "bootstrap-data"


# ----- error paths -----


@pytest.mark.asyncio
async def test_unknown_method_raises():
    t = InProcessTransport("a", _AsyncEchoPlugin())
    with pytest.raises(PluginUnknownMethodError) as exc:
        await t.call("Nonexistent", "x")
    assert exc.value.plugin_id == "a"
    assert exc.value.method == "Nonexistent"


@pytest.mark.asyncio
async def test_plugin_exception_wrapped():
    t = InProcessTransport("r", _RaisingPlugin())
    with pytest.raises(PluginCallError) as exc:
        await t.call("Propose", "x")
    assert "plugin internal failure" in str(exc.value)
    assert exc.value.plugin_id == "r"
    # Original exception preserved
    assert isinstance(exc.value.cause, ValueError)


@pytest.mark.asyncio
async def test_timeout_raises_typed_error():
    t = InProcessTransport("slow", _SlowPlugin(), timeout_seconds=0.05)
    with pytest.raises(PluginTimeoutError) as exc:
        await t.call("Predict", "x")
    assert exc.value.plugin_id == "slow"
    assert exc.value.method == "Predict"


# ----- close idempotent -----


@pytest.mark.asyncio
async def test_close_idempotent():
    t = InProcessTransport("a", _AsyncEchoPlugin())
    await t.close()
    await t.close()
    await t.close()  # multiple close calls must not raise
