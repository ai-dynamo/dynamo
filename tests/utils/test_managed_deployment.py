# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vCluster port-forward retry helper in managed_deployment."""

import logging

import pytest

try:
    import httpx

    from tests.utils.managed_deployment import _API_CONNECTION_ERRORS, _retry_api_call
except ImportError:
    pytest.skip(
        "managed_deployment client deps (kr8s/httpx/aiohttp) not available",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit]

_LOG = logging.getLogger(__name__)


def test_returns_result_on_success():
    assert _retry_api_call(lambda: 42, logger=_LOG) == 42


def test_retries_then_succeeds_after_connection_blip():
    """A dropped-tunnel error (e.g. from a replacement port-forward open) is
    retried, and the call succeeds once the tunnel is back."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("tunnel down")
        return "ok"

    assert _retry_api_call(flaky, logger=_LOG, attempts=5, delay=0) == "ok"
    assert calls["n"] == 3


def test_reraises_after_exhausting_attempts():
    with pytest.raises(ConnectionRefusedError):
        _retry_api_call(
            lambda: (_ for _ in ()).throw(ConnectionRefusedError(111, "refused")),
            logger=_LOG,
            attempts=2,
            delay=0,
        )


def test_does_not_retry_or_mask_non_connection_errors():
    calls = {"n": 0}

    def real_bug():
        calls["n"] += 1
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        _retry_api_call(real_bug, logger=_LOG, attempts=3, delay=0)
    assert calls["n"] == 1  # raised immediately, not retried


def test_connection_error_set_is_precise():
    # Covers the errors observed when the 127.0.0.1:8443 tunnel drops...
    assert issubclass(httpx.ConnectError, _API_CONNECTION_ERRORS)
    assert issubclass(httpx.RemoteProtocolError, _API_CONNECTION_ERRORS)
    assert issubclass(ConnectionRefusedError, _API_CONNECTION_ERRORS)
    # ...without swallowing unrelated OSErrors.
    assert not issubclass(FileNotFoundError, _API_CONNECTION_ERRORS)
