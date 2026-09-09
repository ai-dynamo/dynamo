# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang cross-encoder reranking."""

import argparse
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip(
    "sglang.srt.managers.io_struct", reason="sglang not installed in this container"
)

pytest.importorskip(
    "sglang.srt.speculative.spec_info", reason="full SGLang runtime is required"
)

from dynamo.llm import ModelType  # noqa: E402
from dynamo.sglang import init_embedding as pooling_init  # noqa: E402
from dynamo.sglang.backend_args import (  # noqa: E402
    DynamoSGLangArgGroup,
    DynamoSGLangConfig,
)
from dynamo.sglang.health_check import SglangRerankHealthCheckPayload  # noqa: E402
from dynamo.sglang.init_rerank import init_rerank  # noqa: E402
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
    handler = RerankWorkerHandler.__new__(RerankWorkerHandler)
    handler.engine = _Engine(manager)
    handler.enable_trace = True
    return handler, manager


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
    "bad_request",
    [
        {"model": "m", "query": " ", "documents": ["doc"]},
        {"model": "m", "query": "query", "documents": []},
        {"model": "m", "query": "query", "documents": [""]},
        {"model": "m", "query": "query", "documents": ["doc"], "top_n": 0},
    ],
)
async def test_rejects_invalid_requests_before_inference(bad_request):
    handler, manager = _handler([0.5])
    with pytest.raises(ValueError):
        _ = [output async for output in handler.generate(bad_request, _Context())]
    assert manager.requests == []


def test_rejects_decoder_only_qwen3_reranker_at_startup():
    manager = _TokenizerManager([0.5], model_path="Qwen/Qwen3-Reranker-0.6B")
    with pytest.raises(ValueError, match="cross-encoder rerankers only"):
        RerankWorkerHandler(_Engine(manager), SimpleNamespace())
    assert manager.requests == []


@pytest.mark.asyncio
async def test_embedding_endpoint_does_not_dispatch_rerank():
    handler = EmbeddingWorkerHandler.__new__(EmbeddingWorkerHandler)
    with pytest.raises(ValueError):
        _ = [
            item
            async for item in handler.generate(
                {"model": "m", "query": "query", "documents": ["doc"]}, _Context()
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("rerank", [False, True])
@pytest.mark.parametrize("fail_startup", [False, True])
async def test_dedicated_pooling_registration_health_and_cleanup(
    monkeypatch, rerank, fail_startup
):
    module = pooling_init
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)

    engine = Mock()
    engine.tokenizer_manager.tokenizer.bos_token_id = 1
    endpoint = Mock()
    endpoint.serve_endpoint = AsyncMock()
    runtime = Mock()
    runtime.endpoint.return_value = endpoint
    config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="pooler", model_path="pooler"),
        dynamo_args=SimpleNamespace(
            namespace="test",
            component="rerank" if rerank else "backend",
            endpoint="generate",
            use_sglang_tokenizer=False,
        ),
    )
    handler = Mock()
    handlers = [Mock(return_value=handler), Mock(return_value=handler)]
    monkeypatch.setattr(module, "EmbeddingWorkerHandler", handlers[0])
    monkeypatch.setattr(module, "RerankWorkerHandler", handlers[1])
    monkeypatch.setattr(module.sgl, "Engine", Mock(return_value=engine))
    metrics_task = asyncio.create_task(asyncio.Event().wait())
    monkeypatch.setattr(
        module, "setup_sgl_metrics", AsyncMock(return_value=(None, metrics_task, []))
    )
    register = AsyncMock()
    monkeypatch.setattr(module, "register_model_with_readiness_gate", register)
    monkeypatch.setattr(module, "register_model_taint_route", Mock())
    monkeypatch.setattr(module, "register_engine_metrics_callback", Mock())
    monkeypatch.setattr(module, "init_embedding_metrics", Mock())
    deferred = AsyncMock()
    shutdown_endpoints = []

    init = init_rerank if rerank else module.init_embedding
    if fail_startup:
        handlers[int(rerank)].side_effect = ValueError("unsupported model")
        with pytest.raises(ValueError, match="unsupported model"):
            await init(runtime, config, asyncio.Event(), shutdown_endpoints, deferred)
        register.assert_not_called()
        endpoint.serve_endpoint.assert_not_called()
        engine.shutdown.assert_called_once()
        assert metrics_task.cancelled()
        deferred.assert_awaited_once()
        return
    await init(runtime, config, asyncio.Event(), shutdown_endpoints, deferred)

    assert register.call_args.kwargs["output_type"] == (
        ModelType.Rerank if rerank else ModelType.Embedding
    )
    assert shutdown_endpoints == [endpoint]
    handlers[int(rerank)].assert_called_once()
    handlers[int(not rerank)].assert_not_called()
    assert endpoint.serve_endpoint.call_args.args[0] == handler.generate
    payload = endpoint.serve_endpoint.call_args.kwargs["health_check_payload"]
    if rerank:
        assert payload["query"] and payload["documents"]
        assert "input" not in payload
    else:
        assert payload["input"] == [1]
        assert "query" not in payload
    handler.cleanup.assert_called_once()
    assert metrics_task.cancelled()
    deferred.assert_awaited_once()


def test_rerank_flag_defaults_and_rejects_embedding_combination(monkeypatch):
    monkeypatch.delenv("DYN_SGL_RERANK_WORKER", raising=False)
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    assert not parser.parse_args([]).rerank_worker
    args = parser.parse_args(["--rerank-worker"])
    config = DynamoSGLangConfig.from_cli_args(args)
    config.validate()
    assert config.rerank_worker and not config.embedding_worker
    args = parser.parse_args(["--rerank-worker", "--embedding-worker"])
    with pytest.raises(ValueError, match="cannot be combined"):
        DynamoSGLangConfig.from_cli_args(args).validate()


@pytest.mark.asyncio
async def test_rerank_health_check_runs_cross_encoder(monkeypatch):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    handler, manager = _handler([0.5])
    payload = SglangRerankHealthCheckPayload("pooler").to_dict()
    outputs = [item async for item in handler.generate(payload, _Context())]
    assert outputs[0][0]["score"] == 0.5
    assert manager.requests[0][0].is_cross_encoder_request
