# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import aiohttp
import pytest
from kubernetes import client
from kubernetes.utils.retry import Backoff

from dynamo.common import kubernetes_asyncio

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientConnectionError(), asyncio.TimeoutError()],
)
def test_async_transport_errors_are_retryable(error) -> None:
    assert kubernetes_asyncio.is_transient_async_api_error(error)


@pytest.mark.asyncio
async def test_retry_honors_retry_after() -> None:
    error = client.ApiException(status=429)
    error.headers = {"Retry-After": "3"}
    attempts = 0
    sleeps = []

    async def fn():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        return "ok"

    async def sleep(delay):
        sleeps.append(delay)

    result = await kubernetes_asyncio._retry(
        Backoff(steps=2, duration=0.01, factor=1.0),
        lambda _: True,
        fn,
        use_retry_after=True,
        sleep_func=sleep,
    )

    assert result == "ok"
    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_retry_uses_exponential_backoff_without_retry_after() -> None:
    error = client.ApiException(status=500)
    attempts = 0
    sleeps = []

    async def fn():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error
        return "ok"

    async def sleep(delay):
        sleeps.append(delay)

    result = await kubernetes_asyncio._retry(
        Backoff(steps=3, duration=0.01, factor=5.0),
        lambda _: True,
        fn,
        use_retry_after=True,
        sleep_func=sleep,
    )

    assert result == "ok"
    assert sleeps == [0.01, 0.05]


@pytest.mark.asyncio
async def test_write_reruns_complete_closure_on_conflict(monkeypatch) -> None:
    monkeypatch.setattr(
        kubernetes_asyncio,
        "DEFAULT_RETRY",
        Backoff(steps=2, duration=0.0, factor=1.0),
    )
    attempts = 0

    async def write():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise client.ApiException(status=409)
        return "updated"

    assert await kubernetes_asyncio.retry_kubernetes_write(write) == "updated"
    assert attempts == 2
