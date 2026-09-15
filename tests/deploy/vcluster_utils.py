# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared retry helpers for Kubernetes APIs reached through vCluster."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

VCLUSTER_CONNECTION_RETRY_LIMIT = 3
VCLUSTER_CONNECTION_RETRY_DELAY_SECONDS = 5

_Result = TypeVar("_Result")
_RetryableErrors = tuple[type[Exception], ...]


def retry_vcluster_api(
    operation: str,
    request: Callable[[], _Result],
    retryable_errors: _RetryableErrors,
    logger: logging.Logger,
) -> _Result:
    """Retry a synchronous vCluster API operation while its tunnel recovers."""
    for attempt in range(VCLUSTER_CONNECTION_RETRY_LIMIT + 1):
        try:
            return request()
        except retryable_errors as error:
            if attempt == VCLUSTER_CONNECTION_RETRY_LIMIT:
                raise
            logger.warning(
                "vCluster API connection failed while %s; the port-forward "
                "watchdog may restore the tunnel, retrying in %ss (%s/%s): %s",
                operation,
                VCLUSTER_CONNECTION_RETRY_DELAY_SECONDS,
                attempt + 1,
                VCLUSTER_CONNECTION_RETRY_LIMIT,
                error,
            )
            time.sleep(VCLUSTER_CONNECTION_RETRY_DELAY_SECONDS)

    raise AssertionError("unreachable")


async def retry_vcluster_api_async(
    operation: str,
    request: Callable[[], Awaitable[_Result]],
    retryable_errors: _RetryableErrors,
    logger: logging.Logger,
) -> _Result:
    """Retry an asynchronous vCluster API operation while its tunnel recovers."""
    for attempt in range(VCLUSTER_CONNECTION_RETRY_LIMIT + 1):
        try:
            return await request()
        except retryable_errors as error:
            if attempt == VCLUSTER_CONNECTION_RETRY_LIMIT:
                raise
            logger.warning(
                "vCluster API connection failed while %s; the port-forward "
                "watchdog may restore the tunnel, retrying in %ss (%s/%s): %s",
                operation,
                VCLUSTER_CONNECTION_RETRY_DELAY_SECONDS,
                attempt + 1,
                VCLUSTER_CONNECTION_RETRY_LIMIT,
                error,
            )
            await asyncio.sleep(VCLUSTER_CONNECTION_RETRY_DELAY_SECONDS)

    raise AssertionError("unreachable")
