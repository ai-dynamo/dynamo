# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the opt-in `register_shutdown` kwarg on the `dynamo_worker` decorator:

- Default (`register_shutdown` omitted) leaves signal handling untouched.
- `register_shutdown=True` registers SIGINT and SIGTERM handlers that call
  the runtime's graceful shutdown path.
- The option composes with the existing deprecated `enable_nats` kwarg.

DistributedRuntime is mocked throughout so these tests don't need a real
etcd/discovery backend or a live Tokio runtime.
"""

import asyncio
import inspect
import signal
import warnings
from unittest.mock import MagicMock, patch

import pytest

from dynamo.runtime import dynamo_worker

pytestmark = [
    pytest.mark.unit,
]


def test_dynamo_worker_accepts_register_shutdown_kwarg():
    """dynamo_worker() should accept register_shutdown as an optional kwarg, defaulting to False."""
    sig = inspect.signature(dynamo_worker)
    assert "register_shutdown" in sig.parameters
    param = sig.parameters["register_shutdown"]
    assert param.default is False


@patch("dynamo.runtime.DistributedRuntime")
def test_dynamo_worker_default_does_not_register_shutdown_handlers(mock_runtime_cls):
    """Omitting register_shutdown should not call add_signal_handler at all."""
    mock_runtime_cls.return_value = MagicMock()

    @dynamo_worker()
    async def _worker(runtime):
        pass

    async def _run():
        loop = asyncio.get_running_loop()
        with patch.object(loop, "add_signal_handler") as mock_add_handler:
            await _worker()
        mock_add_handler.assert_not_called()

    asyncio.run(_run())


@patch("dynamo.runtime.DistributedRuntime")
def test_dynamo_worker_register_shutdown_true_registers_both_signals(mock_runtime_cls):
    """register_shutdown=True should register SIGINT and SIGTERM against runtime.shutdown."""
    mock_runtime_instance = MagicMock()
    mock_runtime_cls.return_value = mock_runtime_instance

    @dynamo_worker(register_shutdown=True)
    async def _worker(runtime):
        return "ok"

    async def _run():
        loop = asyncio.get_running_loop()
        with patch.object(loop, "add_signal_handler") as mock_add_handler:
            await _worker()
        return mock_add_handler

    mock_add_handler = asyncio.run(_run())

    registered_signals = {call.args[0] for call in mock_add_handler.call_args_list}
    assert registered_signals == {signal.SIGINT, signal.SIGTERM}
    for call in mock_add_handler.call_args_list:
        assert call.args[1] == mock_runtime_instance.shutdown


@patch("dynamo.runtime.DistributedRuntime")
def test_dynamo_worker_register_shutdown_composes_with_enable_nats(mock_runtime_cls):
    """register_shutdown=True should still work alongside the deprecated enable_nats kwarg."""
    mock_runtime_cls.return_value = MagicMock()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @dynamo_worker(enable_nats=True, register_shutdown=True)
        async def _worker(runtime):
            return "ok"

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "enable_nats" in str(deprecation_warnings[0].message)

    async def _run():
        loop = asyncio.get_running_loop()
        with patch.object(loop, "add_signal_handler") as mock_add_handler:
            await _worker()
        return mock_add_handler

    mock_add_handler = asyncio.run(_run())
    assert mock_add_handler.call_count == 2
