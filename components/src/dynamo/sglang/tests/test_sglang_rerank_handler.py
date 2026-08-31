# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang cross-encoder reranking."""

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "sglang.srt.managers.io_struct", reason="sglang not installed in this container"
)

from dynamo.sglang.request_handlers.embedding import (  # noqa: E402
    EmbeddingWorkerHandler,
)
from dynamo.sglang.request_handlers.rerank import RerankWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _TokenizerManager:
    def __init__(
        self,
        scores: list[float],
        *,
        chat_template: str = "",
        model_path: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.scores = scores
        self.requests: list[tuple[Any, Any]] = []
        self.tokenizer = SimpleNamespace(chat_template=chat_template)
        self.model_config = SimpleNamespace(model_path=model_path)

    async def generate_request(self, request: Any, raw_request: Any):
        request.normalize_batch_and_arguments()
        self.requests.append((request, raw_request))
        yield [
            {
                "embedding": [score],
                "meta_info": {"prompt_tokens": index + 1},
            }
            for index, score in enumerate(self.scores)
        ]


class _Engine:
    def __init__(self, tokenizer_manager: _TokenizerManager) -> None:
        self.tokenizer_manager = tokenizer_manager


class _Context:
    trace_id = "rerank-trace"

    def trace_headers(self) -> dict[str, str]:
        return {"traceparent": "00-test"}


def _handler(
    scores: list[float],
    *,
    chat_template: str = "",
    model_path: str = "BAAI/bge-reranker-v2-m3",
) -> tuple[RerankWorkerHandler, _TokenizerManager]:
    manager = _TokenizerManager(
        scores, chat_template=chat_template, model_path=model_path
    )
    return RerankWorkerHandler(_Engine(manager), enable_trace=True), manager


@pytest.mark.asyncio
async def test_builds_pairs_sorts_scores_and_applies_top_n():
    handler, manager = _handler([0.2, 0.9, 0.5])
    outputs = [
        output
        async for output in handler.generate(
            {
                "model": "reranker",
                "query": "query",
                "documents": ["zero", "one", "two"],
                "top_n": 2,
                "return_documents": True,
            },
            _Context(),
        )
    ]

    assert [item["index"] for item in outputs[0]] == [1, 2]
    assert [item["document"] for item in outputs[0]] == ["one", "two"]
    [(request, raw_request)] = manager.requests
    assert raw_request is None
    assert request.text == [["query", "zero"], ["query", "one"], ["query", "two"]]
    assert request.is_cross_encoder_request is True
    assert request.rid == ["rerank-trace-0", "rerank-trace-1", "rerank-trace-2"]
    assert request.external_trace_header == {"traceparent": "00-test"}


@pytest.mark.asyncio
async def test_omits_documents_when_not_requested():
    handler, _ = _handler([0.1, 0.8])
    [output] = [
        output
        async for output in handler.generate(
            {
                "model": "reranker",
                "query": "query",
                "documents": ["a", "b"],
                "return_documents": False,
            },
            _Context(),
        )
    ]
    assert [item["index"] for item in output] == [1, 0]
    assert all("document" not in item for item in output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request",
    [
        {"model": "m", "query": " ", "documents": ["doc"]},
        {"model": "m", "query": "query", "documents": []},
        {"model": "m", "query": "query", "documents": [""]},
        {"model": "m", "query": "query", "documents": ["doc"], "top_n": 0},
    ],
)
async def test_rejects_invalid_requests_before_inference(request):
    handler, manager = _handler([0.5])
    with pytest.raises(ValueError):
        _ = [output async for output in handler.generate(request, _Context())]
    assert manager.requests == []


@pytest.mark.asyncio
async def test_rejects_decoder_only_qwen3_reranker():
    handler, manager = _handler(
        [0.5],
        chat_template='The answer can only be "yes" or "no"',
        model_path="Qwen/Qwen3-Reranker-0.6B",
    )
    with pytest.raises(ValueError, match="cross-encoder rerankers only"):
        _ = [
            output
            async for output in handler.generate(
                {"model": "m", "query": "query", "documents": ["doc"]},
                _Context(),
            )
        ]
    assert manager.requests == []


@pytest.mark.asyncio
async def test_embedding_endpoint_dispatches_rerank_shape():
    class _RerankHandler:
        async def generate(self, request, context):
            yield [{"score": 0.7, "index": 0}]

    handler = EmbeddingWorkerHandler.__new__(EmbeddingWorkerHandler)
    handler.rerank_handler = _RerankHandler()
    output = [
        item
        async for item in handler.generate(
            {"model": "m", "query": "query", "documents": ["doc"]},
            _Context(),
        )
    ]
    assert output == [[{"score": 0.7, "index": 0}]]
