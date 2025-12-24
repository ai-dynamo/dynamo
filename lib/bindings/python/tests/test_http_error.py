# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import HttpError

pytestmark = pytest.mark.pre_merge


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
