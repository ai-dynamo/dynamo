# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graceful_shutdown.py

Tests the callback ordering and failure handling used during graceful shutdown.
The drain callback prevents decode worker segfaults when a prefill worker scales
down before in-flight NIXL KV transfers complete (issue #7319). The cleanup
callback releases engine resources before the runtime shuts down.

These tests import graceful_shutdown directly (bypassing the dynamo package hierarchy)
so they work without GPU, NIXL, or TensorRT-LLM installed.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# ---------------------------------------------------------------------------
# Module loading: import graceful_shutdown without triggering the full dynamo
# package (which requires dynamo.llm, CUDA, etc.)
#
# We cannot do `from dynamo.common.utils import graceful_shutdown` because the
# dynamo package __init__ transitively imports dynamo._core, which is a native
# extension (PyO3) requiring CUDA/NIXL libraries that are not available in
# unit test environments. Instead, we stub dynamo._core and load the module
# directly from its file path via importlib.
# ---------------------------------------------------------------------------

_GRACEFUL_SHUTDOWN_PATH = Path(__file__).parent.parent / "graceful_shutdown.py"

# Provide a minimal dynamo._core stub while loading graceful_shutdown.py without
# importing the native extension. The stub is restored immediately after module
# load so pytest package collection can still import the real runtime bindings.
_dynamo_core_stub = types.ModuleType("dynamo._core")
_dynamo_core_stub.DistributedRuntime = object
_previous_dynamo = sys.modules.get("dynamo")
_previous_dynamo_core = sys.modules.get("dynamo._core")
_added_dynamo_stub = False
if importlib.util.find_spec("dynamo") is None:
    _dynamo_stub = types.ModuleType("dynamo")
    _dynamo_stub.__path__ = []  # mark as package for dynamo._core imports
    sys.modules["dynamo"] = _dynamo_stub
    _added_dynamo_stub = True
sys.modules["dynamo._core"] = _dynamo_core_stub


def _load_graceful_shutdown():
    spec = importlib.util.spec_from_file_location(
        "dynamo.common.utils.graceful_shutdown",
        _GRACEFUL_SHUTDOWN_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gs = _load_graceful_shutdown()
if _previous_dynamo_core is None:
    sys.modules.pop("dynamo._core", None)
else:
    sys.modules["dynamo._core"] = _previous_dynamo_core
if _added_dynamo_stub:
    if _previous_dynamo is None:
        sys.modules.pop("dynamo", None)
    else:
        sys.modules["dynamo"] = _previous_dynamo

graceful_shutdown_with_discovery = _gs.graceful_shutdown_with_discovery
install_signal_handlers = _gs.install_signal_handlers
is_shutdown_in_progress = _gs.is_shutdown_in_progress
fast_failover_exit_enabled = _gs.fast_failover_exit_enabled


# ---------------------------------------------------------------------------
# Helper: reset the module-level _shutdown_started event between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    _gs._shutdown_started.clear()
    yield
    _gs._shutdown_started.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_drain_callback_called_before_shutdown():
    """Drain callback must be awaited before runtime.shutdown().

    This is the key regression test for issue #7319: prefill workers holding
    active NIXL RDMA references must drain in-flight transfers before their
    process exits, otherwise decode workers segfault accessing freed GPU memory.
    """
    call_order = []

    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

    async def mock_drain():
        call_order.append("drain")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=mock_drain,
        )

    asyncio.run(_run())

    assert "drain" in call_order, "drain_callback was not called"
    assert "shutdown" in call_order, "runtime.shutdown was not called"
    drain_idx = call_order.index("drain")
    shutdown_idx = call_order.index("shutdown")
    assert drain_idx < shutdown_idx, (
        "drain_callback must be called before runtime.shutdown() to ensure "
        "in-flight NIXL transfers complete before GPU memory is freed"
    )


def test_no_drain_callback_still_shuts_down():
    """Backward compatibility: shutdown still works without drain_callback."""
    mock_runtime = MagicMock()

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=None,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_drain_callback_exception_does_not_block_shutdown():
    """Drain callback exceptions must not block shutdown.

    Even if draining fails (e.g., timeout), the shutdown must still proceed
    so the process exits cleanly.
    """
    mock_runtime = MagicMock()

    async def failing_drain():
        raise RuntimeError("drain timed out")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=failing_drain,
        )

    # Should not raise
    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_cleanup_callback_runs_after_drain():
    """Drain, cleanup, and runtime shutdown must run in that order."""
    call_order = []

    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

    async def mock_drain():
        call_order.append("drain")

    async def mock_cleanup():
        call_order.append("cleanup")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            drain_callback=mock_drain,
            cleanup_callback=mock_cleanup,
        )

    asyncio.run(_run())

    assert call_order == ["drain", "cleanup", "shutdown"]


def test_pre_shutdown_callback_withdraws_before_worker_teardown():
    """Lease-owned records disappear before the worker shutdown event is set."""
    call_order = []
    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("runtime"))

    async def withdraw():
        assert not shutdown_event.is_set()
        call_order.append("withdraw")

    async def cleanup():
        assert shutdown_event.is_set()
        call_order.append("engine")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)
        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=shutdown_event,
            grace_period_s=0,
            pre_shutdown_callback=withdraw,
            cleanup_callback=cleanup,
        )

    shutdown_event = asyncio.Event()
    asyncio.run(_run())
    assert call_order == ["withdraw", "engine", "runtime"]


def test_cleanup_callback_exception_does_not_block_shutdown():
    """A cleanup failure must not prevent runtime shutdown."""
    mock_runtime = MagicMock()

    async def failing_cleanup():
        raise RuntimeError("engine cleanup failed")

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            cleanup_callback=failing_cleanup,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


@pytest.mark.timeout(1)
def test_cleanup_callback_timeout_does_not_block_shutdown(monkeypatch):
    """A hanging cleanup callback must time out before runtime shutdown."""
    monkeypatch.setattr(_gs, "_DEFAULT_CLEANUP_TIMEOUT_SECS", 0.05)
    mock_runtime = MagicMock()

    async def hanging_cleanup():
        await asyncio.sleep(10)

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
            cleanup_callback=hanging_cleanup,
        )

    asyncio.run(_run())
    mock_runtime.shutdown.assert_called_once()


def test_shutdown_in_progress_reflects_active_shutdown():
    """The shutdown-in-progress flag is visible before shutdown_event is set."""
    assert is_shutdown_in_progress() is False

    async def _run():
        mock_runtime = MagicMock()
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(return_value=None)
        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=None,
            grace_period_s=0,
        )

    asyncio.run(_run())

    assert is_shutdown_in_progress() is True


def test_fast_failover_exit_env_parsing(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_FAST_EXIT_ON_SIGTERM", raising=False)
    assert fast_failover_exit_enabled() is False

    for value in ("1", "true", "yes", "on", "enabled"):
        monkeypatch.setenv("DYN_GMS_FAILOVER_FAST_EXIT_ON_SIGTERM", value)
        assert fast_failover_exit_enabled() is True

    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("DYN_GMS_FAILOVER_FAST_EXIT_ON_SIGTERM", value)
        assert fast_failover_exit_enabled() is False


def test_fast_failover_exit_runs_after_unregister_before_runtime_shutdown(monkeypatch):
    class FastExit(Exception):
        pass

    call_order = []
    shutdown_event = asyncio.Event()
    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

    async def _run():
        mock_endpoint = AsyncMock()
        mock_endpoint.unregister_endpoint_instance = AsyncMock(
            side_effect=lambda: call_order.append("unregister")
        )

        def fake_fast_exit(event):
            assert event is shutdown_event
            event.set()
            call_order.append("fast_exit")
            raise FastExit

        monkeypatch.setenv("DYN_GMS_FAILOVER_FAST_EXIT_ON_SIGTERM", "1")
        monkeypatch.setattr(_gs, "_fast_exit_after_failover_unregister", fake_fast_exit)

        await graceful_shutdown_with_discovery(
            runtime=mock_runtime,
            endpoints=[mock_endpoint],
            shutdown_event=shutdown_event,
            grace_period_s=0,
        )

    with pytest.raises(FastExit):
        asyncio.run(_run())

    assert call_order == ["unregister", "fast_exit"]
    assert shutdown_event.is_set()
    mock_runtime.shutdown.assert_not_called()
