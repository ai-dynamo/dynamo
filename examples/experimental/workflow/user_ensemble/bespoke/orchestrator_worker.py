# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the user ensemble by manually coordinating Dynamo endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from dynamo.experimental.workflow import StageContext
from dynamo.experimental.workflow.vllm import ExternalEncoderRequestStage
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
        generator_client: Any,
    ) -> None:
        self._encoder = encoder
        self._classifier = classifier
        self._request_adapter = request_adapter
        self._response = response
        self._generator_client = generator_client

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
        _validate_generate_request(request)
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
        generator_context = None
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
            generator_context = context.detached(f"{attempt_id}:generator")
            stream = await self._generator_client.round_robin(
                prepared["request"],
                annotated=False,
                context=generator_context,
            )
            generator_task = asyncio.create_task(
                _collect_generation(stream),
                name=f"bespoke-generator:{attempt_id}",
            )
            classified, completion = await asyncio.gather(
                classifier_task,
                generator_task,
            )
        except BaseException:
            classifier_task.cancel()
            if generator_task is not None:
                generator_task.cancel()
            if generator_context is not None:
                generator_context.stop_generating()
            await asyncio.gather(
                classifier_task,
                *([generator_task] if generator_task is not None else []),
                return_exceptions=True,
            )
            raise

        result = await self._response.run(
            {
                "completion": completion,
                "scores": classified["scores"],
            },
            StageContext(None, "response", attempt_id),
        )
        return dict(result["chunk"])


def _validate_generate_request(request: Mapping[str, Any]) -> None:
    """Match the unary restrictions enforced by GenerateEndpointBinding."""

    sampling_options = request.get("sampling_options", {})
    if not isinstance(sampling_options, Mapping):
        raise TypeError("sampling_options must be an object")
    if sampling_options.get("n") not in (None, 1):
        raise ValueError("user ensemble requires n=1")
    output_options = request.get("output_options", {})
    if not isinstance(output_options, Mapping):
        raise TypeError("output_options must be an object")
    if (
        output_options.get("logprobs") is not None
        or output_options.get("prompt_logprobs") is not None
    ):
        raise ValueError("user ensemble does not support logprobs")


async def _collect_generation(stream: AsyncIterator[Any]) -> dict[str, Any]:
    """Fold stock Generate deltas without using workflow runtime adapters."""

    token_ids: list[int] = []
    terminal: dict[str, Any] | None = None
    async for value in stream:
        if terminal is not None:
            raise RuntimeError("generator returned data after its terminal chunk")
        if not isinstance(value, Mapping):
            raise TypeError("generator returned a non-object chunk")
        chunk = dict(value)
        if chunk.get("index") != 0:
            raise RuntimeError("user ensemble requires generator choice index 0")
        delta = chunk.get("token_ids")
        if not isinstance(delta, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in delta
        ):
            raise TypeError("generator returned invalid token_ids")
        if "log_probs" in chunk or "top_logprobs" in chunk:
            raise RuntimeError("user ensemble does not support generator logprobs")
        token_ids.extend(delta)
        finish_reason = chunk.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or not finish_reason:
                raise TypeError("generator returned invalid finish_reason")
            terminal = chunk
    if terminal is None:
        raise RuntimeError("generator returned no terminal chunk")
    terminal["token_ids"] = token_ids
    return terminal


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
            generator_client=generator_client,
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
