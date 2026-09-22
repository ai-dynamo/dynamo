# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.experimental.llm import LLMUnaryClient, with_engine_data

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.core,
]


def test_with_engine_data_preserves_completion_and_existing_values() -> None:
    completion = {
        "token_ids": [7, 8],
        "finish_reason": "stop",
        "engine_data": {"backend_timing_ms": 2.5},
    }
    values = {"classifier_scores": {"positive": 0.9}}

    result = with_engine_data(completion, values)

    assert result == {
        "token_ids": [7, 8],
        "finish_reason": "stop",
        "engine_data": {
            "backend_timing_ms": 2.5,
            "classifier_scores": {"positive": 0.9},
        },
    }
    assert completion["engine_data"] == {"backend_timing_ms": 2.5}


def test_with_engine_data_rejects_duplicate_keys() -> None:
    completion = {"engine_data": {"classifier_scores": {"positive": 0.8}}}

    with pytest.raises(ValueError, match="classifier_scores"):
        with_engine_data(
            completion,
            {"classifier_scores": {"positive": 0.9}},
        )


@pytest.mark.parametrize(
    "completion, values, message",
    [
        ([], {}, "LLM completion must be an object"),
        ({}, [], "engine_data values must be an object"),
        (
            {"engine_data": "not-an-object"},
            {"classifier_scores": {}},
            "existing engine_data must be an object",
        ),
    ],
)
def test_with_engine_data_rejects_non_object_values(
    completion: Any,
    values: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        with_engine_data(completion, values)


class _Stream:
    def __init__(
        self, values: list[Any], *, terminal_error: BaseException | None = None
    ) -> None:
        self._values = iter(values)
        self._terminal_error = terminal_error
        self.closed = False

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._values)
        except StopIteration:
            if self._terminal_error is not None:
                raise self._terminal_error
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _Client:
    def __init__(self, stream: _Stream) -> None:
        self.stream = stream
        self.calls: list[tuple[Mapping[str, Any], bool, Any]] = []

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> _Stream:
        self.calls.append((request, annotated, context))
        return self.stream


def _request() -> dict[str, Any]:
    return {
        "token_ids": [1, 2],
        "sampling_options": {"n": 1},
        "output_options": {},
    }


@pytest.mark.asyncio
async def test_llm_unary_client_collects_tokens_and_terminal_metadata() -> None:
    stream = _Stream(
        [
            {"token_ids": [7], "index": 0},
            {
                "token_ids": [8, 9],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": {"completion_tokens": 3},
            },
        ]
    )
    client = _Client(stream)
    request = _request()
    context = object()

    result = await LLMUnaryClient(client).complete(request, context=context)

    assert result == {
        "token_ids": [7, 8, 9],
        "index": 0,
        "finish_reason": "stop",
        "completion_usage": {"completion_tokens": 3},
    }
    assert client.calls == [(request, False, context)]
    assert stream.closed


@pytest.mark.parametrize(
    "request_value, message",
    [
        ({"sampling_options": {"n": 2}}, "requires n=1"),
        ({"sampling_options": []}, "sampling_options must be an object"),
        ({"output_options": {"logprobs": 0}}, "does not support logprobs"),
        ({"output_options": []}, "output_options must be an object"),
    ],
)
@pytest.mark.asyncio
async def test_llm_unary_client_rejects_unsupported_request(
    request_value: Mapping[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await LLMUnaryClient(_Client(_Stream([]))).complete(request_value)


@pytest.mark.parametrize(
    "chunks, message",
    [
        (["bad"], "non-object chunk"),
        ([{"token_ids": [1], "index": 1}], "choice index 0"),
        ([{"token_ids": [True], "index": 0}], "invalid token_ids"),
        ([{"token_ids": [1], "index": 0}], "no terminal chunk"),
        (
            [{"token_ids": [1], "index": 0, "finish_reason": ""}],
            "invalid finish_reason",
        ),
        (
            [{"token_ids": [1], "index": 0, "log_probs": [-0.1]}],
            "unsupported logprobs",
        ),
        (
            [
                {"token_ids": [1], "index": 0, "finish_reason": "stop"},
                {"token_ids": [2], "index": 0},
            ],
            "data after terminal",
        ),
    ],
)
@pytest.mark.asyncio
async def test_llm_unary_client_rejects_invalid_stream(
    chunks: list[Any], message: str
) -> None:
    stream = _Stream(chunks)

    with pytest.raises(RuntimeError, match=message):
        await LLMUnaryClient(_Client(stream)).complete(_request())

    assert stream.closed


@pytest.mark.asyncio
async def test_llm_unary_client_propagates_cancellation() -> None:
    stream = _Stream([], terminal_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await LLMUnaryClient(_Client(stream)).complete(_request())

    assert stream.closed
