# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from kubernetes import client

from dynamo.common import kubernetes

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


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


def test_read_retry_does_not_duplicate_retry_after_handling() -> None:
    error = client.ApiException(status=429)
    error.headers = {"Retry-After": "2"}

    assert not kubernetes.is_transient_without_retry_after(error)


def test_read_retry_uses_backoff_without_retry_after() -> None:
    error = client.ApiException(status=429)

    assert kubernetes.is_transient_without_retry_after(error)
