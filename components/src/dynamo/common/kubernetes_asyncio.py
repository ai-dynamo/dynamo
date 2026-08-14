# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry policy for the separately packaged ``kubernetes_asyncio`` client."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aiohttp
from kubernetes.utils.retry import (
    DEFAULT_BACKOFF,
    DEFAULT_RETRY,
    Backoff,
    is_conflict,
    retry_after_seconds,
)

from dynamo.common.kubernetes import is_transient_api_error

T = TypeVar("T")


def is_transient_async_api_error(error: Exception) -> bool:
    return is_transient_api_error(error) or isinstance(
        error, (aiohttp.ClientError, asyncio.TimeoutError)
    )


# kubernetes_asyncio cannot import kubernetes.aio.utils.retry because that tree
# is not part of the published kubernetes wheel. This is the async execution of
# the same client-go algorithms, matching:
# https://github.com/kubernetes/client-go/blob/master/util/retry/util.go
async def _retry(
    backoff: Backoff,
    retriable: Callable[[Exception], bool],
    fn: Callable[[], Awaitable[T]],
    *,
    use_retry_after: bool,
    sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_func: Callable[[], float] = random.random,
) -> T:
    steps = backoff.steps
    duration = backoff.duration
    last_error: Exception | None = None

    while steps > 0:
        try:
            return await fn()
        except Exception as error:
            if not retriable(error):
                raise
            last_error = error

            steps -= 1
            if steps == 0:
                break

            delay = duration
            if backoff.jitter:
                delay += random_func() * backoff.jitter * delay
            if use_retry_after:
                retry_after = retry_after_seconds(error)
                if retry_after is not None and retry_after > delay:
                    delay = retry_after
            await sleep_func(delay)

            if backoff.factor != 0:
                duration *= backoff.factor
                if backoff.cap > 0 and duration > backoff.cap:
                    duration = backoff.cap
                    steps = 0

    assert last_error is not None
    raise last_error


async def retry_kubernetes_request(fn: Callable[[], Awaitable[T]]) -> T:
    """Retry a transient async request with Retry-After-aware backoff."""

    return await _retry(
        DEFAULT_BACKOFF,
        is_transient_async_api_error,
        fn,
        use_retry_after=True,
    )


async def retry_kubernetes_read(fn: Callable[[], Awaitable[T]]) -> T:
    """Retry transient async reads with Retry-After-aware backoff."""

    return await retry_kubernetes_request(fn)


async def retry_kubernetes_write(fn: Callable[[], Awaitable[T]]) -> T:
    """Retry an async read-modify-write closure on conflicts and throttling."""

    async def retry_transient() -> T:
        return await retry_kubernetes_request(fn)

    return await _retry(
        DEFAULT_RETRY,
        is_conflict,
        retry_transient,
        use_retry_after=False,
    )
