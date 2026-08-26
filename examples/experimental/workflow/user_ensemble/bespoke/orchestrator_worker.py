# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the user ensemble by manually coordinating Dynamo endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from dynamo.experimental.workflow import StageContext
from dynamo.experimental.workflow.generate import GenerateEndpointInvoker
from dynamo.experimental.workflow.vllm import (
    DynamoVllmStage,
    ExternalEncoderRequestStage,
)
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from examples.experimental.workflow.user_ensemble.common.config import (
    CHAT_TEMPLATE,
    GENERATOR_ENDPOINT,
    MODEL,
    ORCHESTRATOR_ENDPOINT,
    PUBLIC_MODEL_NAME,
    build_encoder_stage,
)
from examples.experimental.workflow.user_ensemble.common.stages import (
    DummyClassifier,
    EnsembleResponseStage,
)


class BespokeEnsembleHandler:
    """Manually implement the same fan-out, remote call, and merge lifecycle."""

    def __init__(
        self,
        *,
        encoder: Any,
        classifier: Any,
        request_adapter: Any,
        response: Any,
        generator: GenerateEndpointInvoker,
    ) -> None:
        self._encoder = encoder
        self._classifier = classifier
        self._request_adapter = request_adapter
        self._response = response
        self._generator = generator

    async def generate(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        execution = asyncio.create_task(
            self._run(request, context),
            name=f"bespoke-ensemble:{context.id()}",
        )
        cancellation = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait(
                {execution, cancellation}, return_when=asyncio.FIRST_COMPLETED
            )
            if execution not in done:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                return
            yield await execution
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise
        finally:
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

    async def _run(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        attempt_id = context.id()
        encoder_context = StageContext(
            workflow_name=None,
            stage_id="encoder",
            attempt_id=attempt_id,
        )
        encoded = await self._encoder.run({"request": request}, encoder_context)

        classifier_task = asyncio.create_task(
            self._classifier.run(
                {"encoder_features": encoded["encoder_features"]},
                StageContext(None, "classifier", attempt_id),
            ),
            name=f"bespoke-classifier:{attempt_id}",
        )
        generator_task = None
        try:
            prepared = await self._request_adapter.run(
                {
                    "request": request,
                    "encoder_features": encoded["encoder_features"],
                    "encoder_metadata": encoded["encoder_metadata"],
                },
                StageContext(None, "request_adapter", attempt_id),
            )
            generator_task = asyncio.create_task(
                self._generator.run(
                    "generator",
                    DynamoVllmStage.request_complete_contract,
                    {"request": prepared["request"]},
                    StageContext(None, "generator", attempt_id),
                    request_context=context,
                ),
                name=f"bespoke-generator:{attempt_id}",
            )
            classified, generated = await asyncio.gather(
                classifier_task,
                generator_task,
            )
        except BaseException:
            classifier_task.cancel()
            if generator_task is not None:
                generator_task.cancel()
            await asyncio.gather(
                classifier_task,
                *([generator_task] if generator_task is not None else []),
                return_exceptions=True,
            )
            raise

        result = await self._response.run(
            {
                "completion": generated["completion"],
                "scores": classified["scores"],
            },
            StageContext(None, "response", attempt_id),
        )
        return dict(result["chunk"])


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    encoder = build_encoder_stage()
    try:
        generator_client = await runtime.endpoint(GENERATOR_ENDPOINT).client()
        await generator_client.wait_for_instances()
        handler = BespokeEnsembleHandler(
            encoder=encoder,
            classifier=DummyClassifier(),
            request_adapter=ExternalEncoderRequestStage(),
            response=EnsembleResponseStage(),
            generator=GenerateEndpointInvoker(generator_client),
        )
        endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat,
            endpoint,
            MODEL,
            model_name=PUBLIC_MODEL_NAME,
            worker_type=WorkerType.Aggregated,
            custom_template_path=str(CHAT_TEMPLATE),
            ignore_weights=True,
        )
        await endpoint.serve_endpoint(handler.generate)
    finally:
        encoder.close()


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
