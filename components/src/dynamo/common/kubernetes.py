# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared retry policy for synchronous Kubernetes API clients."""

from collections.abc import Callable
from typing import TypeVar

from kubernetes import client
from kubernetes.client import rest
from kubernetes.utils.retry import (
    DEFAULT_BACKOFF,
    on_retry_after_error,
    retry_on_conflict,
)

T = TypeVar("T")


def client_go_api_client() -> client.ApiClient:
    """Build an API client with client-go-compatible read retries enabled."""

    configuration = client.Configuration.get_default_copy()
    configuration.client_go_retries = True
    return client.ApiClient(configuration)


def is_transient_api_error(error: Exception) -> bool:
    if isinstance(error, rest.urllib3.exceptions.HTTPError):
        return True
    status = getattr(error, "status", None)
    return status == 0 or status == 429 or (isinstance(status, int) and status >= 500)


def retry_kubernetes_request(fn: Callable[[], T]) -> T:
    """Retry one API operation on throttling and transient server errors."""

    return on_retry_after_error(DEFAULT_BACKOFF, is_transient_api_error, fn)


def retry_kubernetes_write(fn: Callable[[], T]) -> T:
    """Restart a complete read-modify-write closure after conflicts.

    ``fn`` must re-read the object before every write and submit that read's
    resource version. The write request inside ``fn`` must use
    :func:`retry_kubernetes_request` for throttling and server errors. Keeping
    request retries at the write prevents automatic GET retries from being
    multiplied by retries of the complete closure.
    """

    return retry_on_conflict(fn)
