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


class RetriedRequestConflict(rest.ApiException):
    """A 409 observed after retrying an ambiguously completed request."""

    def __init__(self, conflict: Exception):
        super().__init__(
            status=getattr(conflict, "status", 409),
            reason=getattr(conflict, "reason", None),
            body=getattr(conflict, "body", None),
            data=getattr(conflict, "data", None),
        )
        self.headers = getattr(conflict, "headers", None)


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
    """Retry one API operation on throttling and transient server errors.

    If a transport or server failure is followed by a conflict, preserve that
    ambiguity for callers that cannot safely repeat a relative mutation.
    """

    ambiguous_completion = False

    def request() -> T:
        nonlocal ambiguous_completion
        try:
            return fn()
        except Exception as error:
            status = getattr(error, "status", None)
            if status == 409 and ambiguous_completion:
                raise RetriedRequestConflict(error) from error
            if (
                isinstance(error, rest.urllib3.exceptions.HTTPError)
                or status == 0
                or (isinstance(status, int) and status >= 500)
            ):
                ambiguous_completion = True
            raise

    return on_retry_after_error(DEFAULT_BACKOFF, is_transient_api_error, request)


def retry_kubernetes_write(fn: Callable[[], T]) -> T:
    """Restart a complete read-modify-write closure after conflicts.

    ``fn`` must re-read the object before every write and submit that read's
    resource version. The write request inside ``fn`` must use
    :func:`retry_kubernetes_request` for throttling and server errors. Keeping
    request retries at the write prevents automatic GET retries from being
    multiplied by retries of the complete closure.
    """

    return retry_on_conflict(fn)
