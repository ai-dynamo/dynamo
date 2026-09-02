# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang embedding input dispatch."""

from typing import Any

import pytest

pytest.importorskip(
    "sglang.srt.managers.io_struct", reason="sglang not installed in this container"
)

from sglang.srt.managers.io_struct import EmbeddingReqInput  # noqa: E402

from dynamo.sglang.request_handlers.embedding import (  # noqa: E402
    embedding_handler as eh,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _TokenizerManager:
    def __init__(self) -> None:
        self.requests: list[tuple[EmbeddingReqInput, Any]] = []

    async def generate_request(self, request: EmbeddingReqInput, context: Any):
        request.normalize_batch_and_arguments()
        self.requests.append((request, context))
        yield {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 2}}


class _Engine:
    def __init__(self) -> None:
        self.tokenizer_manager = _TokenizerManager()
        self.async_encode_calls: list[dict[str, Any]] = []

    async def async_encode(self, **kwargs: Any):
        self.async_encode_calls.append(kwargs)
        return {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 2}}


class _Context:
    trace_id = "embedding-trace"

    def trace_headers(self) -> dict[str, str]:
        return {"traceparent": "00-test"}


def _handler(*, enable_trace: bool = True) -> eh.EmbeddingWorkerHandler:
    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)
    handler.engine = _Engine()
    handler.enable_trace = enable_trace
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding_input", "expected_request_id"),
    [
        ("hello", "embedding-trace"),
        (["hello", "world"], ["embedding-trace-0", "embedding-trace-1"]),
    ],
)
async def test_text_inputs_use_async_encode(embedding_input, expected_request_id):
    handler = _handler()

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": embedding_input}, _Context()
        )
    ]

    assert len(outputs) == 1
    assert handler.engine.async_encode_calls == [
        {
            "prompt": embedding_input,
            "external_trace_header": {"traceparent": "00-test"},
            "rid": expected_request_id,
        }
    ]
    assert handler.engine.tokenizer_manager.requests == []


@pytest.mark.asyncio
async def test_single_tokenized_input_uses_native_input_ids():
    handler = _handler()

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": [11, 22, 33]}, _Context()
        )
    ]

    assert len(outputs) == 1
    assert handler.engine.async_encode_calls == []
    [(request, context)] = handler.engine.tokenizer_manager.requests
    assert context is None
    assert request.text is None
    assert request.input_ids == [11, 22, 33]
    assert request.rid == "embedding-trace"
    assert request.external_trace_header == {"traceparent": "00-test"}


@pytest.mark.asyncio
async def test_batched_tokenized_input_gets_unique_request_ids():
    handler = _handler(enable_trace=False)
    token_ids = [[11, 22], [33, 44]]

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": token_ids}, _Context()
        )
    ]

    assert len(outputs) == 1
    [(request, context)] = handler.engine.tokenizer_manager.requests
    assert context is None
    assert request.text is None
    assert request.input_ids == token_ids
    assert request.rid == ["embedding-trace-0", "embedding-trace-1"]
    assert request.external_trace_header is None
