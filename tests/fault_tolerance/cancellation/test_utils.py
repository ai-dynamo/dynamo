# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for cancellation response validation."""

from collections.abc import Iterator

import pytest

from tests.fault_tolerance.cancellation.utils import (
    CancellableRequest,
    read_streaming_responses,
)

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


class _StreamingResponse:
    """Minimal response surface consumed by read_streaming_responses()."""

    status_code = 200

    def __init__(self, *lines: bytes):
        self._lines = lines

    def iter_lines(self) -> Iterator[bytes]:
        yield from self._lines


def _request_with_stream(*lines: bytes) -> CancellableRequest:
    request = CancellableRequest()
    request.response = _StreamingResponse(*lines)
    return request


def test_drained_stream_can_require_generated_content():
    """Role and terminal metadata alone must not prove worker health."""
    request = _request_with_stream(
        b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    )

    with pytest.raises(pytest.fail.Exception, match="without generated content"):
        read_streaming_responses(
            request,
            expected_count=1,
            drain=True,
            require_content=True,
        )


def test_drained_stream_accepts_generated_content():
    """A generated text fragment plus terminal metadata proves worker health."""
    request = _request_with_stream(
        b'data: {"choices":[{"delta":{"content":"healthy"},"finish_reason":null}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    )

    read_streaming_responses(
        request,
        expected_count=1,
        drain=True,
        require_content=True,
    )
