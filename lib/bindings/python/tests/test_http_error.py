# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import _raise_routed_http_error
from dynamo.llm import HttpError
from dynamo.llm.exceptions import RoutedEngineError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_raise_http_error():
    with pytest.raises(HttpError):
        raise HttpError(404, "Not Found")
    with pytest.raises(Exception):
        raise HttpError(500, "Internal Server Error")


def test_invalid_http_error_code():
    with pytest.raises(ValueError):
        HttpError(1700, "Invalid Code")


def test_invalid_http_error_message():
    with pytest.raises(ValueError):
        # The second argument must be a string, not bytes.
        HttpError(400, b"Bad Request")


def test_long_http_error_message():
    message = ("A" * 8192) + "B"
    error = HttpError(400, message)
    assert len(error.message) == 8192

    # Ensure the exception string uses the truncated message too.
    assert message[:8189] in str(error)
    assert "B" not in str(error)


def test_routed_engine_error_is_a_value_error():
    # Callers that caught the ValueError a routed stream raised before still
    # catch it. test_http_server covers the status it keeps.
    with pytest.raises(RoutedEngineError) as raised:
        _raise_routed_http_error(503, "worker busy")
    assert isinstance(raised.value, ValueError)


def test_only_a_routed_engine_stream_raises_routed_engine_error():
    # A worker that relays a `Client` or `KvRouter` stream must not forward the
    # other worker's error as its own, so those streams keep a plain ValueError.
    with pytest.raises(ValueError) as raised:
        _raise_routed_http_error(503, "worker busy", routed_engine=False)
    assert type(raised.value) is ValueError
