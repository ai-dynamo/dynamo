# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graceful_shutdown.py

Tests the drain_callback mechanism added to prevent decode worker segfaults when
a prefill worker scales down before in-flight NIXL KV transfers complete (issue
#7319), and the cleanup_callback mechanism that releases engine resources (GPU
memory, PyTorch process groups) before the Rust runtime tears down.

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

# Provide a minimal dynamo._core stub so the module can be loaded
_dynamo_stub = types.ModuleType("dynamo")
_dynamo_core_stub = types.ModuleType("dynamo._core")
_dynamo_core_stub.DistributedRuntime = object
sys.modules.setdefault("dynamo", _dynamo_stub)
sys.modules.setdefault("dynamo._core", _dynamo_core_stub)


def _load_graceful_shutdown():
    spec = importlib.util.spec_from_file_location(
        "dynamo.common.utils.graceful_shutdown",
        _GRACEFUL_SHUTDOWN_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gs = _load_graceful_shutdown()
graceful_shutdown_with_discovery = _gs.graceful_shutdown_with_discovery
install_signal_handlers = _gs.install_signal_handlers


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


# ---------------------------------------------------------------------------
# cleanup_callback tests
#
# Regression coverage for the unified backend leak: when the Rust runtime
# tears down on SIGTERM, the Python Worker.run() finally block cannot finish
# engine.cleanup() before the loop's native backing collapses. cleanup_callback
# moves engine.cleanup() into the signal-handler path, *before* runtime.shutdown().
# ---------------------------------------------------------------------------


def test_cleanup_callback_called_before_shutdown():
    """Cleanup callback must be awaited before runtime.shutdown().

    This is the regression test for the unified backend GPU/process-group leak:
    engine.cleanup() must run before the Rust runtime tears down, otherwise
    PyTorch's destroy_process_group() never fires.
    """
    call_order = []

    mock_runtime = MagicMock()
    mock_runtime.shutdown = MagicMock(side_effect=lambda: call_order.append("shutdown"))

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
            cleanup_callback=mock_cleanup,
        )

    asyncio.run(_run())

    assert "cleanup" in call_order, "cleanup_callback was not called"
    assert "shutdown" in call_order, "runtime.shutdown was not called"
    cleanup_idx = call_order.index("cleanup")
    shutdown_idx = call_order.index("shutdown")
    assert cleanup_idx < shutdown_idx, (
        "cleanup_callback must be called before runtime.shutdown() so the "
        "Python engine releases GPU memory and process groups before the "
        "Rust runtime tears down"
    )


def test_cleanup_callback_runs_after_drain():
    """When both callbacks are set, drain runs before cleanup.

    Drain releases in-flight NIXL transfers; cleanup releases engine
    resources. Cleanup must wait until drain finishes so the engine isn't
    torn down with active RDMA references pointing into its GPU memory.
    """
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

    assert call_order == [
        "drain",
        "cleanup",
        "shutdown",
    ], f"expected drain -> cleanup -> shutdown, got {call_order}"


def test_cleanup_callback_exception_does_not_block_shutdown():
    """Cleanup callback exceptions must not block shutdown.

    A failing engine.cleanup() (e.g., engine handle already half-torn-down)
    must not strand the process; runtime.shutdown() still has to run.
    """
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
    """A hanging cleanup callback must time out and let shutdown proceed.

    If engine.cleanup() blocks (e.g., NCCL stuck), graceful shutdown still
    needs to call runtime.shutdown() so the orchestrator can reap the pod.
    """
    # Shorten the timeout so the test runs quickly.
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
