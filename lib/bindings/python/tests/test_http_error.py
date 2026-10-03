# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import HttpError

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


def test_http_error_parameter_is_explicit_public_context():
    error = HttpError(400, "private diagnostic", param="reasoning_effort")
    assert error.param == "reasoning_effort"
    assert error.message == "private diagnostic"
    assert HttpError(400, "legacy").param is None


@pytest.mark.parametrize("param", ["", "private/path", "x y", "x" * 129, 1, True, "é"])
def test_http_error_rejects_malformed_parameter(param):
    with pytest.raises(ValueError, match="top-level field"):
        HttpError(400, "private diagnostic", param=param)


@pytest.mark.parametrize("status", [200, 404, 422, 500])
def test_http_error_parameter_requires_validation_status(status):
    with pytest.raises(ValueError, match="status 400"):
        HttpError(status, "private diagnostic", param="reasoning_effort")
