# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for the Valkey worker startup transaction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.valkey_startup import (
    UnsafeKvEventPublisherSetupError,
    complete_valkey_startup,
)

pytestmark = [pytest.mark.unit, pytest.mark.core, pytest.mark.pre_merge]


def _handler():
    handler = SimpleNamespace(
        valkey_worker_registration=None,
        kv_publishers=None,
        shutdown_valkey_worker_registration=AsyncMock(),
        cleanup=Mock(),
    )
    return handler


@pytest.mark.asyncio
async def test_publisher_failure_unregisters_the_acquired_lease():
    registration = SimpleNamespace(shutdown=AsyncMock())
    handler = _handler()

    async def shutdown() -> None:
        current = handler.valkey_worker_registration
        handler.valkey_worker_registration = None
        if current is not None:
            await current.shutdown()

    handler.shutdown_valkey_worker_registration.side_effect = shutdown

    with pytest.raises(RuntimeError, match="injected publisher setup failure"):
        await complete_valkey_startup(
            handler,
            acquire_registration=AsyncMock(return_value=registration),
            start_publishers=AsyncMock(
                side_effect=RuntimeError("injected publisher setup failure")
            ),
            register_model=AsyncMock(),
        )

    registration.shutdown.assert_awaited_once_with()
    handler.cleanup.assert_called_once_with()


@pytest.mark.asyncio
async def test_model_failure_drains_publishers_before_unregistering():
    lifecycle: list[str] = []

    async def shutdown() -> None:
        lifecycle.append("publisher")
        handler.kv_publishers = None
        lifecycle.append("registration")
        handler.valkey_worker_registration = None

    handler = _handler()
    handler.shutdown_valkey_worker_registration.side_effect = shutdown
    publisher = SimpleNamespace(shutdown_and_wait=AsyncMock())

    with pytest.raises(RuntimeError, match="injected model failure"):
        await complete_valkey_startup(
            handler,
            acquire_registration=AsyncMock(return_value=object()),
            start_publishers=AsyncMock(return_value=[publisher]),
            register_model=AsyncMock(
                side_effect=RuntimeError("injected model failure")
            ),
        )

    assert lifecycle == ["publisher", "registration"]
    handler.cleanup.assert_called_once_with()


@pytest.mark.asyncio
async def test_unsafe_partial_setup_leaves_lease_to_expire():
    registration = SimpleNamespace(shutdown=AsyncMock())
    handler = _handler()

    with pytest.raises(UnsafeKvEventPublisherSetupError):
        await complete_valkey_startup(
            handler,
            acquire_registration=AsyncMock(return_value=registration),
            start_publishers=AsyncMock(
                side_effect=UnsafeKvEventPublisherSetupError("unsafe partial setup")
            ),
            register_model=AsyncMock(),
        )

    registration.shutdown.assert_not_awaited()
    assert handler.valkey_worker_registration is None
    handler.shutdown_valkey_worker_registration.assert_awaited_once_with()
    handler.cleanup.assert_called_once_with()
