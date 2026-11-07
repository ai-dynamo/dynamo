# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeEngineClient:
    tokenizer = None

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _stream():
            for response in self.responses:
                yield response

        return _stream()


class _FakeContext:
    def id(self):
        return "req-1"

    def trace_headers(self):
        return None


def _output(
    token_ids,
    *,
    index=0,
    finish_reason=None,
    stop_reason=None,
    logprobs=None,
):
    return SimpleNamespace(
        index=index,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        logprobs=logprobs,
    )


def _request_output(
    outputs, *, prompt_token_ids=None, num_cached_tokens=0, prompt_logprobs=None
):
    return SimpleNamespace(
        outputs=outputs,
        prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [101],
        num_cached_tokens=num_cached_tokens,
        kv_transfer_params=None,
        prompt_logprobs=prompt_logprobs,
    )


def _handler_with_responses(responses):
    def _ignore_log(*args, **kwargs):
        pass

    handler = SimpleNamespace()
    handler.engine_client = _FakeEngineClient(responses)
    handler.runtime = SimpleNamespace(shutdown=lambda: None)
    handler._extract_logprobs = BaseWorkerHandler._extract_logprobs
    handler._log_with_lora_context = _ignore_log
    # These delta-streaming tests exercise base-model requests only. Model the
    # no-LoRA branch without constructing the full legacy worker handler.
    handler._generate_with_lora_admission_lock = (
        lambda lora_request, create_generator: create_generator(lora_request)
    )
    return handler


async def _collect_handler_chunks(
    responses, *, sampling_params=None, max_chunks=None, **kwargs
):
    """Collect worker chunks. ``max_chunks`` stops reading early, as the
    frontend does when it matches a stop string and cancels the worker."""
    handler = _handler_with_responses(responses)
    stream = BaseWorkerHandler.generate_tokens(
        handler,
        prompt=None,
        sampling_params=sampling_params or SamplingParams(),
        request_id="req-1",
        **kwargs,
    )
    chunks = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
    finally:
        await stream.aclose()
    return chunks, handler


def test_build_sampling_params_forces_delta_token_mode():
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {"output_kind": RequestOutputKind.CUMULATIVE},
        "stop_conditions": {},
        "output_options": {},
    }

    sampling_params = build_sampling_params(
        request,
        default_sampling_params={
            "detokenize": True,
            "output_kind": RequestOutputKind.CUMULATIVE,
        },
    )

    assert sampling_params.detokenize is False
    assert sampling_params.output_kind == RequestOutputKind.DELTA


@pytest.mark.asyncio
async def test_generate_tokens_passes_delta_chunks_without_cumulative_slicing():
    responses = [
        _request_output([_output([1])], prompt_token_ids=[10, 11]),
        _request_output([_output([2, 3])], prompt_token_ids=[10, 11]),
        _request_output(
            [_output([4], finish_reason="length")], prompt_token_ids=[10, 11]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1], [2, 3], [4]]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 4,
        "total_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


@pytest.mark.asyncio
async def test_generate_tokens_reports_cached_usage_before_finish():
    responses = [
        _request_output(
            [output],
            prompt_token_ids=[10, 11, 12],
            num_cached_tokens=2,
        )
        for output in (
            _output([1]),
            _output([2, 3]),
            _output([], finish_reason="length"),
        )
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1], [2, 3], []]
    assert "finish_reason" not in chunks[0]
    assert "finish_reason" not in chunks[1]
    assert chunks[-1]["finish_reason"] == "length"
    assert "completion_usage" not in chunks[1]
    for chunk, completion_tokens in [(chunks[0], 1), (chunks[-1], 3)]:
        assert chunk["completion_usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": completion_tokens,
            "total_tokens": 3 + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 2},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_tokens", [0, 2])
async def test_generate_tokens_reports_cache_count_when_it_becomes_available(
    cached_tokens,
):
    responses = [
        _request_output(
            [_output([token_id])],
            prompt_token_ids=[10, 11, 12],
            num_cached_tokens=cache_count,
        )
        for token_id, cache_count in [(1, None), (2, cached_tokens)]
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert all("finish_reason" not in chunk for chunk in chunks)
    assert chunks[0]["completion_usage"]["prompt_tokens_details"] is None
    assert chunks[1]["completion_usage"]["prompt_tokens_details"] == {
        "cached_tokens": cached_tokens
    }


@pytest.mark.asyncio
async def test_generate_tokens_reports_prompt_and_cache_metadata_changes():
    responses = [
        _request_output(
            [_output([index])],
            prompt_token_ids=list(range(prompt_tokens)),
            num_cached_tokens=cached_tokens,
        )
        for index, (prompt_tokens, cached_tokens) in enumerate(
            [(3, 0), (3, 0), (4, 0), (4, 2), (4, 2)]
        )
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert ["completion_usage" in chunk for chunk in chunks] == [
        True,
        False,
        True,
        True,
        False,
    ]
    assert chunks[2]["completion_usage"]["prompt_tokens"] == 4
    assert chunks[3]["completion_usage"]["prompt_tokens_details"] == {
        "cached_tokens": 2
    }


@pytest.mark.asyncio
async def test_generate_tokens_keeps_final_empty_delta_chunk_for_usage():
    responses = [
        _request_output([_output([1, 2])], prompt_token_ids=[10]),
        _request_output([_output([], finish_reason="length")], prompt_token_ids=[10]),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1, 2], []]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 2
    assert chunks[-1]["completion_usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_generate_tokens_ignores_logprobs_on_empty_final_delta_chunk():
    logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
    ]
    responses = [
        _request_output([_output([1, 2])], prompt_token_ids=[10]),
        _request_output(
            [_output([], finish_reason="length", logprobs=logprobs)],
            prompt_token_ids=[10],
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[1, 2], []]
    assert "log_probs" not in chunks[-1]
    assert "top_logprobs" not in chunks[-1]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 2
    assert chunks[-1]["completion_usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_generate_tokens_tracks_interleaved_output_indexes_independently():
    responses = [
        _request_output([_output([1], index=0), _output([10, 11], index=1)]),
        _request_output(
            [
                _output([2], index=0, finish_reason="length"),
                _output([12], index=1),
            ]
        ),
        _request_output([_output([], index=1, finish_reason="length")]),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [(chunk["index"], chunk["token_ids"]) for chunk in chunks] == [
        (0, [1]),
        (1, [10, 11]),
        (0, [2]),
        (1, [12]),
        (1, []),
    ]
    assert chunks[2]["completion_usage"]["completion_tokens"] == 4
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 5
    assert "completion_usage" not in chunks[3]
    assert [
        chunk["completion_usage"]["completion_tokens"]
        for chunk in chunks
        if "completion_usage" in chunk
    ] == [
        1,
        3,
        4,
        5,
    ]


@pytest.mark.asyncio
async def test_generate_tokens_reads_delta_aligned_logprobs_from_zero_offset():
    logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
        {8: SimpleNamespace(logprob=-0.8, rank=1, decoded_token="b")},
    ]
    responses = [
        _request_output([_output([7, 8], finish_reason="length", logprobs=logprobs)])
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert chunks[0]["token_ids"] == [7, 8]
    assert chunks[0]["log_probs"] == [-0.7, -0.8]
    assert [entry[0]["token_id"] for entry in chunks[0]["top_logprobs"]] == [7, 8]


@pytest.mark.asyncio
async def test_generate_tokens_keeps_multichunk_delta_logprobs_aligned():
    first_logprobs = [
        {7: SimpleNamespace(logprob=-0.7, rank=1, decoded_token="a")},
    ]
    second_logprobs = [
        {8: SimpleNamespace(logprob=-0.8, rank=1, decoded_token="b")},
        {9: SimpleNamespace(logprob=-0.9, rank=1, decoded_token="c")},
    ]
    responses = [
        _request_output([_output([7], logprobs=first_logprobs)]),
        _request_output(
            [_output([8, 9], finish_reason="length", logprobs=second_logprobs)]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert [chunk["token_ids"] for chunk in chunks] == [[7], [8, 9]]
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.7], [-0.8, -0.9]]
    assert [
        [entry[0]["token_id"] for entry in chunk["top_logprobs"]] for chunk in chunks
    ] == [[7], [8, 9]]


# vLLM's raw prompt_logprobs for a two-token prompt, and its serialized form.
_RAW_PROMPT_LOGPROBS = [
    None,
    {11: SimpleNamespace(logprob=-0.5, rank=1, decoded_token="b")},
]
_PROMPT_LOGPROBS_PAYLOAD = [
    None,
    {"11": {"logprob": -0.5, "rank": 1, "decoded_token": "b"}},
]


def _chunks_with_prompt_logprobs(chunks):
    return [
        i
        for i, chunk in enumerate(chunks)
        if "prompt_logprobs" in (chunk.get("engine_data") or {})
    ]


@pytest.mark.asyncio
async def test_generate_tokens_sends_prompt_logprobs_before_terminal():
    # The frontend matches a stop string after the second chunk and cancels
    # the worker, so the terminal output is never read.
    responses = [
        _request_output(
            [_output([1])],
            prompt_token_ids=[10, 11],
            prompt_logprobs=_RAW_PROMPT_LOGPROBS,
        ),
        _request_output([_output([2])], prompt_token_ids=[10, 11]),
        _request_output(
            [_output([3], finish_reason="length")], prompt_token_ids=[10, 11]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses, max_chunks=2)

    assert [chunk["token_ids"] for chunk in chunks] == [[1], [2]]
    assert all("finish_reason" not in chunk for chunk in chunks)
    assert chunks[0]["engine_data"]["prompt_logprobs"] == _PROMPT_LOGPROBS_PAYLOAD
    assert "engine_data" not in chunks[1]


@pytest.mark.asyncio
async def test_generate_tokens_sends_late_prompt_logprobs_once():
    responses = [
        _request_output([_output([1])], prompt_token_ids=[10, 11]),
        _request_output(
            [_output([2])],
            prompt_token_ids=[10, 11],
            prompt_logprobs=_RAW_PROMPT_LOGPROBS,
        ),
        _request_output(
            [_output([3], finish_reason="length")], prompt_token_ids=[10, 11]
        ),
    ]

    chunks, _ = await _collect_handler_chunks(responses)

    assert _chunks_with_prompt_logprobs(chunks) == [1]
    assert chunks[1]["engine_data"]["prompt_logprobs"] == _PROMPT_LOGPROBS_PAYLOAD


@pytest.mark.asyncio
async def test_generate_tokens_sends_prompt_logprobs_once_per_choice():
    # The completions echo path reads the payload per choice index, so each
    # choice's first token output carries it.
    responses = [
        _request_output(
            [_output([1], index=0), _output([10], index=1)],
            prompt_token_ids=[10, 11],
            prompt_logprobs=_RAW_PROMPT_LOGPROBS,
        ),
        _request_output(
            [
                _output([2], index=0, finish_reason="length"),
                _output([12], index=1, finish_reason="length"),
            ],
            prompt_token_ids=[10, 11],
        ),
    ]

    chunks, _ = await _collect_handler_chunks(
        responses, sampling_params=SamplingParams(n=2)
    )

    assert [chunk["index"] for chunk in chunks] == [0, 1, 0, 1]
    assert _chunks_with_prompt_logprobs(chunks) == [0, 1]
    for chunk in chunks[:2]:
        assert chunk["engine_data"]["prompt_logprobs"] == _PROMPT_LOGPROBS_PAYLOAD
