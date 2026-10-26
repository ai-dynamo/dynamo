# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.outputs import PoolingOutput, PoolingRequestOutput

import dynamo.vllm.rerank_handler as mod
from dynamo.llm.exceptions import EngineShutdown

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


def _config(**overrides: Any) -> SimpleNamespace:
    values = {
        "model": "BAAI/bge-reranker-v2-m3",
        "served_model_name": "reranker",
        "model_source_path": "/models/bge-reranker-v2-m3",
        "custom_jinja_template": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_handler(
    serving: Any, shutdown_event: asyncio.Event | None = None
) -> mod.RerankWorkerHandler:
    with patch.object(mod, "VllmEngineMonitor"):
        return mod.RerankWorkerHandler(
            runtime=MagicMock(),
            engine=MagicMock(),
            config=_config(),
            serving=serving,
            shutdown_event=shutdown_event,
        )


def _make_context(stopped: bool = False) -> MagicMock:
    def _killed_or_stopped() -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        if stopped:
            future.set_result(None)
        return future

    context = MagicMock()
    context.id.return_value = "engine-request"
    context.async_killed_or_stopped.side_effect = _killed_or_stopped
    return context


def _vllm_response(results: list[tuple[int, float]]) -> SimpleNamespace:
    """Shape of the JSONResponse vLLM's ServingScores returns for a rerank."""
    body = {
        "id": "score-engine-request",
        "model": "reranker",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "results": [
            {"index": index, "document": {"text": None}, "relevance_score": score}
            for index, score in results
        ],
    }
    return SimpleNamespace(body=json.dumps(body).encode())


async def _collect(handler: mod.RerankWorkerHandler, request, context) -> list:
    return [item async for item in handler.generate(request, context)]


class TestGenerate:
    @pytest.mark.asyncio
    async def test_maps_vllm_results_to_ranked_documents(self):
        serving = AsyncMock(return_value=_vllm_response([(1, 0.9), (0, 0.1)]))
        handler = _make_handler(serving)

        outputs = await _collect(
            handler,
            {
                "model": "reranker",
                "query": "What is Dynamo?",
                "documents": ["bread", "Dynamo serves inference."],
            },
            _make_context(),
        )

        assert outputs == [
            [
                {"score": 0.9, "index": 1, "document": "Dynamo serves inference."},
                {"score": 0.1, "index": 0, "document": "bread"},
            ]
        ]
        (vllm_request,) = serving.await_args.args
        assert vllm_request.model == "reranker"
        assert vllm_request.query == "What is Dynamo?"
        assert vllm_request.documents == ["bread", "Dynamo serves inference."]
        assert vllm_request.top_n == 0
        assert vllm_request.request_id == "engine-request"

    @pytest.mark.asyncio
    async def test_forwards_top_n_and_omits_documents_on_request(self):
        serving = AsyncMock(return_value=_vllm_response([(1, 0.9)]))
        handler = _make_handler(serving)

        outputs = await _collect(
            handler,
            {
                "query": "q",
                "documents": ["a", "b"],
                "top_n": 1,
                "return_documents": False,
            },
            _make_context(),
        )

        assert outputs == [[{"score": 0.9, "index": 1}]]
        (vllm_request,) = serving.await_args.args
        assert vllm_request.top_n == 1
        # Without a model on the wire request, the served model name is used.
        assert vllm_request.model == "reranker"

    @pytest.mark.asyncio
    async def test_parses_response_built_by_vllm(self):
        """Feed the handler a response built by vLLM's own rerank serializer,
        so a change to vLLM's response shape fails here."""
        documents = ["bread", "Dynamo serves inference.", "oceans"]
        final_res_batch = [
            PoolingRequestOutput(
                request_id=f"score-engine-request-{index}",
                outputs=PoolingOutput(data=torch.tensor([score])),
                prompt_token_ids=[1, 2],
                num_cached_tokens=0,
                finished=True,
            )
            for index, score in enumerate([0.1, 0.9, 0.4])
        ]
        vllm_response = ServingScores._request_output_to_rerank_response(
            None,  # type: ignore[arg-type]
            final_res_batch,
            request_id="score-engine-request",
            model_name="reranker",
            documents=documents,
            top_n=2,
            num_prompt_tokens=6,
        )
        handler = _make_handler(AsyncMock(return_value=vllm_response))

        outputs = await _collect(
            handler,
            {"query": "q", "documents": documents, "top_n": 2},
            _make_context(),
        )

        assert [(item["index"], item["document"]) for item in outputs[0]] == [
            (1, "Dynamo serves inference."),
            (2, "oceans"),
        ]
        assert [item["score"] for item in outputs[0]] == pytest.approx([0.9, 0.4])

    @pytest.mark.asyncio
    async def test_scoring_errors_propagate(self):
        serving = AsyncMock(side_effect=ValueError("bad rerank input"))
        handler = _make_handler(serving)

        with pytest.raises(ValueError, match="bad rerank input"):
            await _collect(handler, {"query": "q", "documents": ["a"]}, _make_context())

    @pytest.mark.asyncio
    async def test_cancelled_request_stops_scoring_without_output(self):
        cancelled = asyncio.Event()

        async def _never_finishes(_request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        handler = _make_handler(_never_finishes)

        outputs = await _collect(
            handler, {"query": "q", "documents": ["a"]}, _make_context(stopped=True)
        )

        assert outputs == []
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_raises_engine_shutdown(self):
        async def _never_finishes(_request):
            await asyncio.Event().wait()

        shutdown_event = asyncio.Event()
        shutdown_event.set()
        handler = _make_handler(_never_finishes, shutdown_event=shutdown_event)

        with pytest.raises(EngineShutdown):
            await _collect(handler, {"query": "q", "documents": ["a"]}, _make_context())


class TestFromEngine:
    @staticmethod
    def _engine() -> MagicMock:
        engine = MagicMock()
        engine.get_supported_tasks = AsyncMock(return_value=("classify",))
        return engine

    @pytest.mark.asyncio
    async def test_rejects_model_without_scoring_support(self):
        with patch.object(mod, "VllmEngineMonitor"), patch.object(
            mod, "enable_scoring_api", return_value=False
        ):
            with pytest.raises(ValueError, match="does not support vLLM's scoring"):
                await mod.RerankWorkerHandler.from_engine(
                    runtime=MagicMock(), engine=self._engine(), config=_config()
                )

    @pytest.mark.asyncio
    async def test_builds_serving_scores_with_score_template(self):
        engine = self._engine()
        config = _config(custom_jinja_template="/templates/qwen3_reranker.jinja")
        with patch.object(mod, "VllmEngineMonitor"), patch.object(
            mod, "enable_scoring_api", return_value=True
        ), patch.object(
            mod, "load_chat_template", return_value="TEMPLATE"
        ) as load_template, patch.object(
            mod, "OpenAIServingModels"
        ) as serving_models, patch.object(
            mod, "ServingScores"
        ) as serving_scores:
            handler = await mod.RerankWorkerHandler.from_engine(
                runtime=MagicMock(), engine=engine, config=config
            )

        load_template.assert_called_once_with("/templates/qwen3_reranker.jinja")
        (base_model_path,) = serving_models.call_args.kwargs["base_model_paths"]
        assert base_model_path.name == "reranker"
        assert base_model_path.model_path == "/models/bge-reranker-v2-m3"
        args, kwargs = serving_scores.call_args
        assert args == (engine, serving_models.return_value)
        assert kwargs["supported_tasks"] == ("classify",)
        assert kwargs["chat_template_config"].chat_template == "TEMPLATE"
        assert handler.serving is serving_scores.return_value
