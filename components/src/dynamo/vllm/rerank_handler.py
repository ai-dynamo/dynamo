# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker handling for the ``/v1/rerank`` API.

Scoring is delegated to vLLM's own ``ServingScores``, so the worker builds
prompts the same way ``vllm serve`` does: cross-encoders, bi-encoders and
late-interaction models use their native scoring path, and decoder-only
rerankers such as Qwen3-Reranker get their score template from
``--custom-jinja-template`` (vLLM's ``--chat-template``).
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.utils import enable_scoring_api
from vllm.renderers.chat_utils import ChatTemplateConfig, load_chat_template

from dynamo._core import Context
from dynamo.llm.exceptions import EngineShutdown

from .args import Config
from .engine_monitor import VllmEngineMonitor

logger = logging.getLogger(__name__)


class RerankWorkerHandler:
    """Handle ``/v1/rerank`` on a vLLM pooling engine.

    Each request runs through vLLM's ``ServingScores`` and is converted to
    the SGLang-compatible response the frontend aggregates: a list of
    ``{score, index, document?}`` sorted by descending score.
    """

    def __init__(
        self,
        runtime: Any,
        engine: Any,
        config: Config,
        serving: ServingScores,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.runtime = runtime
        self.engine_client = engine
        self.config = config
        self.shutdown_event = shutdown_event
        # Dead-engine detection, as on the embedding and classify workers: a
        # crashed pooling engine must not leave the endpoint registered.
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.serving = serving
        self.model_name = config.served_model_name or config.model
        logger.info("Rerank worker handler initialized")

    def cleanup(self) -> None:
        """Release resources owned by this handler.

        AsyncLLM lifecycle is owned by the worker factory; the engine monitor
        cancels its background tasks via ``__del__``.
        """
        return None

    @classmethod
    async def from_engine(
        cls,
        runtime: Any,
        engine: Any,
        config: Config,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> "RerankWorkerHandler":
        """Build the handler, failing fast if the model cannot score."""
        supported_tasks = await engine.get_supported_tasks()
        model_config = engine.model_config
        if not enable_scoring_api(supported_tasks, model_config):
            raise ValueError(
                f"Model {config.model} does not support vLLM's scoring API "
                f"(supported tasks: {supported_tasks}). --rerank-worker needs a "
                "cross-encoder with one output label, an embedding model, or a "
                "late-interaction model, started with --runner pooling. Original "
                "Qwen3 rerankers also need --hf-overrides that load them as "
                "Qwen3ForSequenceClassification."
            )

        model_name = config.served_model_name or config.model
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=[
                BaseModelPath(name=model_name, model_path=config.model_source_path)
            ],
            lora_modules=None,
        )
        serving = ServingScores(
            engine,
            models,
            supported_tasks=supported_tasks,
            request_logger=None,
            chat_template_config=ChatTemplateConfig(
                chat_template=load_chat_template(config.custom_jinja_template),
            ),
        )
        return cls(
            runtime=runtime,
            engine=engine,
            config=config,
            serving=serving,
            shutdown_event=shutdown_event,
        )

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Handle one internal rerank wire request."""
        documents = request["documents"]
        top_n = request.get("top_n")
        rerank_request = RerankRequest(
            model=request.get("model") or self.model_name,
            query=request["query"],
            documents=documents,
            # vLLM uses 0 for "return every document".
            top_n=top_n if top_n is not None else 0,
            request_id=context.id(),
        )

        response = await self._score(rerank_request, context)
        if response is None:
            return

        return_documents = request.get("return_documents", True)
        ranked: list[dict[str, Any]] = []
        for result in json.loads(response.body)["results"]:
            item: dict[str, Any] = {
                "score": result["relevance_score"],
                "index": result["index"],
            }
            if return_documents:
                item["document"] = documents[result["index"]]
            ranked.append(item)
        yield ranked

    async def _score(self, rerank_request: RerankRequest, context: Context) -> Any:
        """Run one scoring request, cancelling it if the caller goes away.

        Cancelling the task cancels vLLM's per-prompt ``encode`` generators,
        which abort their engine requests. Returns ``None`` when the request
        was cancelled and raises ``EngineShutdown`` on worker shutdown.
        """
        score_task = asyncio.create_task(self.serving(rerank_request))
        stopped = asyncio.ensure_future(context.async_killed_or_stopped())
        waiters: set[asyncio.Future[Any]] = {score_task, stopped}
        shutdown_task: Optional[asyncio.Task[Any]] = None
        if self.shutdown_event is not None:
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())
            waiters.add(shutdown_task)

        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if score_task in done:
                return score_task.result()

            logger.debug("Aborting rerank request ID: %s", context.id())
            score_task.cancel()
            await asyncio.gather(score_task, return_exceptions=True)
            if shutdown_task is not None and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during rerank.")
            return None
        finally:
            pending = [waiter for waiter in waiters if not waiter.done()]
            for waiter in pending:
                waiter.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
