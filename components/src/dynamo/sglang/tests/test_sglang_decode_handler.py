# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager

import pytest

from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    _extract_media_urls,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_extract_media_urls_supports_string_and_wire_items():
    mm_data = {
        "video_url": [
            "file:///tmp/test.mp4",
            {"Url": "https://example.com/test.mp4"},
            {"ignored": "value"},
        ]
    }

    assert _extract_media_urls(mm_data, "video_url") == [
        "file:///tmp/test.mp4",
        "https://example.com/test.mp4",
    ]


def test_extract_media_urls_returns_none_for_missing_or_invalid_items():
    assert _extract_media_urls({}, "image_url") is None
    assert (
        _extract_media_urls({"image_url": [{"ignored": "value"}]}, "image_url") is None
    )


class _FakeContext:
    def id(self):
        return "ctx"

    def is_stopped(self):
        return False


@pytest.mark.asyncio
async def test_process_token_stream_treats_completion_usage_fields_as_optional():
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)

    @asynccontextmanager
    async def _noop_monitor(request_id_future, context):
        yield None

    handler._cancellation_monitor = _noop_monitor

    async def _stream():
        yield {
            "meta_info": {
                "id": "rid",
                "finish_reason": {"type": "stop"},
            },
            "output_ids": [],
        }

    outputs = [
        item async for item in handler._process_token_stream(_stream(), _FakeContext())
    ]

    assert outputs == [{"finish_reason": "stop", "token_ids": []}]
