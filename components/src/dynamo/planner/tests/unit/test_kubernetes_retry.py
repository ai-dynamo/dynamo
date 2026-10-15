# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from kubernetes import client
from kubernetes.client import rest

from dynamo.common import kubernetes

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_client_go_api_client_enables_read_retries(monkeypatch) -> None:
    configuration = MagicMock()
    api_client = MagicMock()
    get_default_copy = MagicMock(return_value=configuration)
    api_client_factory = MagicMock(return_value=api_client)
    monkeypatch.setattr(
        kubernetes.client.Configuration, "get_default_copy", get_default_copy
    )
    monkeypatch.setattr(kubernetes.client, "ApiClient", api_client_factory)

    assert kubernetes.client_go_api_client() is api_client
    assert configuration.client_go_retries is True
    api_client_factory.assert_called_once_with(configuration)


@pytest.mark.parametrize("status", [0, 429, 500, 503])
def test_transient_api_errors_are_retryable(status) -> None:
    assert kubernetes.is_transient_api_error(client.ApiException(status=status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_non_transient_api_errors_are_not_retryable(status) -> None:
    assert not kubernetes.is_transient_api_error(client.ApiException(status=status))


def test_transport_errors_are_retryable() -> None:
    assert kubernetes.is_transient_api_error(
        rest.urllib3.exceptions.ProtocolError("connection reset", OSError("reset"))
    )


def test_write_retry_does_not_multiply_transient_request_retries() -> None:
    attempts = 0

    def write() -> None:
        nonlocal attempts
        attempts += 1
        raise client.ApiException(status=429)

    with pytest.raises(client.ApiException):
        kubernetes.retry_kubernetes_write(write)

    assert attempts == 1


def test_conflict_after_ambiguous_request_retry_is_distinguished() -> None:
    errors = iter([client.ApiException(status=500), client.ApiException(status=409)])

    def request() -> None:
        raise next(errors)

    with pytest.raises(kubernetes.RetriedRequestConflict):
        kubernetes.retry_kubernetes_request(request)


def test_conflict_after_throttling_remains_an_ordinary_conflict() -> None:
    errors = iter([client.ApiException(status=429), client.ApiException(status=409)])

    def request() -> None:
        raise next(errors)

    with pytest.raises(client.ApiException) as raised:
        kubernetes.retry_kubernetes_request(request)

    assert type(raised.value) is client.ApiException
