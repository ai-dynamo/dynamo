# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Event, get_ident
from unittest.mock import MagicMock

import pytest

from dynamo.planner.environment.runtime import RuntimeNamespaceBinding

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.asyncio
async def test_runtime_namespace_binding_offloads_resolution():
    event_loop_thread = get_ident()
    resolver = MagicMock()

    def resolve_runtime_namespace(namespace):
        assert get_ident() != event_loop_thread
        assert namespace == "base-ns"
        return "base-ns-workerhash"

    resolver.get_worker_runtime_namespace.side_effect = resolve_runtime_namespace
    binding = RuntimeNamespaceBinding(namespace="base-ns", resolver=resolver)

    assert await binding.refresh_runtime_namespace() is True
    assert binding.runtime_namespace() == "base-ns-workerhash"


@pytest.mark.asyncio
async def test_runtime_namespace_refresh_cancellation_does_not_wait_for_blocked_read():
    started = Event()
    release = Event()
    resolver = MagicMock()

    def resolve_runtime_namespace(_namespace):
        started.set()
        release.wait()
        return "base-ns-workerhash"

    resolver.get_worker_runtime_namespace.side_effect = resolve_runtime_namespace
    binding = RuntimeNamespaceBinding(namespace="base-ns", resolver=resolver)
    refresh = asyncio.create_task(binding.refresh_runtime_namespace())

    assert await asyncio.to_thread(started.wait, 1)
    refresh.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(refresh, timeout=0.2)
    finally:
        release.set()
