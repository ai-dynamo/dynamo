# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    _extract_media_urls,
    _extract_sglang_stop_reason,
    _openai_stop_sampling_params,
    _user_stop_token_ids,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import StreamProcessor

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


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ({"type": "stop", "matched": "END"}, "END"),
        ({"type": "stop", "matched": 128001}, 128001),
        ({"type": "stop", "matched": [128001, 128009]}, [128001, 128009]),
        ({"type": "stop", "matched": True}, None),
        ({"type": "stop", "matched": ["END"]}, None),
        ({"type": "length"}, None),
        (None, None),
    ],
)
def test_extract_sglang_stop_reason(finish_reason, expected):
    assert _extract_sglang_stop_reason(finish_reason) == expected


def test_extract_sglang_stop_reason_filters_hidden_token_ids():
    finish_reason = {"type": "stop", "matched": 128001}

    assert _extract_sglang_stop_reason(finish_reason, {576}) is None
    assert _extract_sglang_stop_reason(finish_reason, {128001}) == 128001


def test_extract_sglang_stop_reason_filters_hidden_token_id_arrays():
    finish_reason = {"type": "stop", "matched": [128001, 128009]}

    assert _extract_sglang_stop_reason(finish_reason, {128001}) is None
    assert _extract_sglang_stop_reason(finish_reason, {128001, 128009}) == [
        128001,
        128009,
    ]


def test_user_stop_token_ids_ignores_hidden_ids():
    assert _user_stop_token_ids(
        {
            "stop_conditions": {
                "stop_token_ids": [576],
                "stop_token_ids_hidden": [128001],
            }
        }
    ) == {576}


def test_user_stop_token_ids_handles_null_fields():
    assert _user_stop_token_ids({"stop_conditions": {"stop_token_ids": None}}) == set()
    assert _user_stop_token_ids({"stop_token_ids": None}) == set()


def test_user_stop_token_ids_accepts_stop_token_id_array():
    assert _user_stop_token_ids({"stop": [32, 34]}) == {32, 34}


def test_user_stop_token_ids_treats_token_id_display_as_string_stop():
    assert _user_stop_token_ids({"stop": ["token_id:576"]}) == set()


def test_openai_stop_sampling_params_preserves_string_stops():
    assert _openai_stop_sampling_params({"stop": "END"}) == {"stop": "END"}
    assert _openai_stop_sampling_params({"stop": ["END"]}) == {"stop": ["END"]}
    assert _openai_stop_sampling_params({"stop": ["token_id:576"]}) == {
        "stop": ["token_id:576"]
    }


def test_openai_stop_sampling_params_maps_token_id_stop_array():
    assert _openai_stop_sampling_params({"stop": [32, 34]}) == {
        "stop_token_ids": [32, 34]
    }
    assert _openai_stop_sampling_params({"stop_token_ids": [32, 34]}) == {
        "stop_token_ids": [32, 34]
    }


def _new_decode_handler(*, use_sglang_tokenizer: bool = False):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = use_sglang_tokenizer
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model")
    )

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor
    return handler


async def _stream(items):
    for item in items:
        yield item


class _Context:
    def is_stopped(self):
        return False


def test_build_sampling_params_passes_n_for_token_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=False)

    sampling_params = handler._build_sampling_params(
        {
            "sampling_options": {"temperature": 0.2, "top_p": 0.9, "n": 3},
            "stop_conditions": {"max_tokens": 8},
        }
    )

    assert sampling_params["n"] == 3
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8


def test_build_sampling_params_passes_n_for_sglang_tokenizer_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=True)

    sampling_params = handler._build_sampling_params(
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "n": 2,
            "max_tokens": 8,
            "stop": [32, 34],
        }
    )

    assert sampling_params["n"] == 2
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8
    assert sampling_params["stop_token_ids"] == [32, 34]


def test_build_logprob_kwargs_allows_chosen_token_logprobs(monkeypatch):
    monkeypatch.delenv("DYN_SGL_ALLOW_TOP_LOGPROBS", raising=False)

    kwargs = DecodeWorkerHandler._build_logprob_kwargs(
        {"output_options": {"logprobs": 0}}
    )

    assert kwargs == {"return_logprob": True, "top_logprobs_num": 0}


def test_build_logprob_kwargs_rejects_top_logprobs_by_default(monkeypatch):
    monkeypatch.delenv("DYN_SGL_ALLOW_TOP_LOGPROBS", raising=False)

    with pytest.raises(ValueError, match="does not currently support logprobs >= 1"):
        DecodeWorkerHandler._build_logprob_kwargs({"output_options": {"logprobs": 1}})


def test_build_logprob_kwargs_allows_top_logprobs_with_escape_hatch(monkeypatch):
    monkeypatch.setenv("DYN_SGL_ALLOW_TOP_LOGPROBS", "1")

    kwargs = DecodeWorkerHandler._build_logprob_kwargs(
        {"output_options": {"logprobs": 2}}
    )

    assert kwargs == {"return_logprob": True, "top_logprobs_num": 2}


def test_extract_logprobs_formats_top_tokens_as_token_ids():
    log_probs, top_logprobs, total = DecodeWorkerHandler._extract_logprobs(
        {
            "output_token_logprobs": [(-0.1, 101, "a")],
            "output_top_logprobs": [[(-0.1, 101, "a"), (-0.2, 102, "b")]],
        },
        0,
        return_tokens_as_token_ids=True,
    )

    assert log_probs == [-0.1]
    assert top_logprobs == [
        [
            {"rank": 1, "token_id": 101, "token": "token_id:101", "logprob": -0.1},
            {"rank": 2, "token_id": 102, "token": "token_id:102", "logprob": -0.2},
        ]
    ]
    assert total == 1


@pytest.mark.asyncio
async def test_process_token_stream_tracks_logprobs_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.1, 101, "a")],
                        },
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.2, 201, "b")],
                        },
                    },
                    {
                        "index": 0,
                        "output_ids": [102],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [
                                (-0.1, 101, "a"),
                                (-0.3, 102, "c"),
                            ],
                        },
                    },
                ]
            ),
            _Context(),
        )
    )

    assert [chunk["index"] for chunk in chunks] == [0, 1, 0]
    assert [chunk["token_ids"] for chunk in chunks] == [[101], [201], [102]]
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.1], [-0.2], [-0.3]]


@pytest.mark.asyncio
async def test_process_text_stream_tracks_delta_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "He",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Go",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Good",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                ]
            ),
            _Context(),
        )
    )

    choices = [chunk["choices"][0] for chunk in chunks]
    assert [choice["index"] for choice in choices] == [0, 1, 0, 1]
    assert [choice["delta"]["content"] for choice in choices] == [
        "He",
        "Go",
        "llo",
        "od",
    ]


@pytest.mark.asyncio
async def test_process_text_stream_stop_reason_uses_response_nvext():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": "END"},
                        },
                    }
                ]
            ),
            _Context(),
            request={"nvext": {"extra_fields": ["stop_reason"]}},
        )
    )

    assert "stop_reason" not in chunks[0]["choices"][0]
    assert chunks[0]["nvext"]["stop_reason"] == "END"


@pytest.mark.asyncio
async def test_process_text_stream_stop_reason_requires_nvext_extra_field():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": "END"},
                        },
                    }
                ]
            ),
            _Context(),
        )
    )

    assert "stop_reason" not in chunks[0]["choices"][0]
    assert "nvext" not in chunks[0]


@pytest.mark.asyncio
async def test_process_token_stream_suppresses_hidden_stop_token_reason():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [128001],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": 128001},
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cached_tokens": None,
                        },
                    }
                ]
            ),
            _Context(),
            user_stop_token_ids={576},
        )
    )

    assert "stop_reason" not in chunks[0]


@pytest.mark.asyncio
async def test_multimodal_stream_keeps_reading_after_one_choice_finishes():
    chunks = await _collect(
        StreamProcessor.process_sglang_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "text": "a",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "text": "b",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 0,
                        "output_ids": [],
                        "text": "a",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                    {
                        "index": 1,
                        "output_ids": [],
                        "text": "b",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                ]
            )
        )
    )

    outputs = [json.loads(chunk) for chunk in chunks]

    assert [output["index"] for output in outputs] == [0, 1, 0, 1]
    assert [output["finished"] for output in outputs] == [False, False, True, True]
    assert [output.get("finish_reason") for output in outputs] == [
        None,
        None,
        "stop",
        "stop",
    ]


async def _collect(stream):
    return [item async for item in stream]
