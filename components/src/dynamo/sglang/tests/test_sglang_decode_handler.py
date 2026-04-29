# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_build_logprob_kwargs_token_request_uses_output_options_logprobs():
    assert DecodeWorkerHandler._build_logprob_kwargs(
        {"output_options": {"logprobs": 3}}
    ) == {
        "return_logprob": True,
        "top_logprobs_num": 3,
    }


def test_extract_logprobs_from_meta_info():
    log_probs, top_logprobs = DecodeWorkerHandler._extract_logprobs(
        {
            "output_token_logprobs": [[-0.25, 11, " hello"], [-0.5, 12, " world"]],
            "output_top_logprobs": [
                [[-0.25, 11, " hello"], [-1.0, 99, " hi"]],
                [[-0.5, 12, " world"]],
            ],
        },
        0,
        2,
    )

    assert log_probs == [-0.25, -0.5]
    assert top_logprobs is not None
    assert top_logprobs[0][0]["token"] == " hello"
    assert top_logprobs[0][0]["bytes"] == list(" hello".encode("utf-8"))


def test_extract_logprobs_uses_start_offset_for_streaming_chunks():
    log_probs, top_logprobs = DecodeWorkerHandler._extract_logprobs(
        {
            "output_token_logprobs": [
                [-0.1, 10, " a"],
                [-0.2, 11, " b"],
                [-0.3, 12, " c"],
            ],
            "output_top_logprobs": [
                [[-0.1, 10, " a"]],
                [[-0.2, 11, " b"]],
                [[-0.3, 12, " c"]],
            ],
        },
        1,
        1,
    )

    assert log_probs == [-0.2]
    assert top_logprobs == [
        [
            {
                "rank": 0,
                "token_id": 11,
                "token": " b",
                "logprob": -0.2,
                "bytes": list(" b".encode("utf-8")),
            }
        ]
    ]


def test_extract_logprobs_returns_none_when_offset_exceeds_available_entries():
    log_probs, top_logprobs = DecodeWorkerHandler._extract_logprobs(
        {"output_token_logprobs": [[-0.25, 11, " hello"]]},
        1,
        5,
    )

    assert log_probs is None
    assert top_logprobs is None
