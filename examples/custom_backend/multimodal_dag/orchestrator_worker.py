# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI-facing orchestrator for the custom vision DAG."""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import uvloop
from transformers import AutoTokenizer, GenerationConfig

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_utils.external_qwen_artifact import ExternalQwenArtifact
from examples.custom_backend.multimodal_dag.protocol import (
    CLASSIFIER_ENDPOINT,
    DEFAULT_BACKEND_MODEL,
    ORCHESTRATOR_ENDPOINT,
    PUBLIC_MODEL_NAME,
    VISION_ENCODER_ENDPOINT,
    VLLM_ENDPOINT,
    VllmResult,
    apply_stop,
    build_vllm_request,
    chat_chunk,
    validate_chat_request,
)

configure_dynamo_logging(service_name="multimodal-dag-orchestrator")
logger = logging.getLogger(__name__)


def _normalize_eos_token_ids(value: int | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Sequence) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    raise ValueError(
        "generation config eos_token_id must be an integer or integer list"
    )


def _response_data(response: Any) -> Mapping[str, Any]:
    data = response.data()
    if not isinstance(data, Mapping):
        raise TypeError(
            f"downstream worker returned {type(data).__name__}, expected an object"
        )
    return data


class OrchestratorHandler:
    """Run encoder-first, then classifier and vLLM concurrently."""

    def __init__(
        self,
        *,
        backend_model: str,
        tokenizer: Any,
        eos_token_ids: Sequence[int],
        encoder_client: Client,
        classifier_client: Client,
        vllm_client: Client,
    ) -> None:
        self._backend_model = backend_model
        self._tokenizer = tokenizer
        self._eos_token_ids = list(eos_token_ids)
        self._encoder_client = encoder_client
        self._classifier_client = classifier_client
        self._vllm_client = vllm_client

    async def _collect_single(
        self,
        client: Client,
        request: Mapping[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        stream = await client.generate(dict(request), context=context)
        responses: list[dict[str, Any]] = []
        async for response in stream:
            responses.append(dict(_response_data(response)))
            if len(responses) > 1:
                raise RuntimeError(
                    "downstream unary worker returned multiple responses"
                )
        if not responses:
            raise RuntimeError("downstream unary worker returned no response")
        return responses[0]

    async def _collect_vllm(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> VllmResult:
        logger.info("Starting vLLM branch")
        stream = await self._vllm_client.generate(dict(request), context=context)
        token_ids: list[int] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        async for response in stream:
            chunk = _response_data(response)
            index = chunk.get("index", 0)
            if index != 0:
                raise ValueError(f"vLLM returned unsupported output index {index}")
            chunk_token_ids = chunk.get("token_ids", [])
            if not isinstance(chunk_token_ids, list) or any(
                not isinstance(token_id, int) or isinstance(token_id, bool)
                for token_id in chunk_token_ids
            ):
                raise ValueError("vLLM returned invalid token_ids")
            token_ids.extend(chunk_token_ids)

            chunk_finish_reason = chunk.get("finish_reason")
            if isinstance(chunk_finish_reason, str):
                if chunk_finish_reason.startswith("error:"):
                    raise RuntimeError(chunk_finish_reason)
                finish_reason = chunk_finish_reason
            chunk_usage = chunk.get("completion_usage")
            if chunk_usage is not None:
                if not isinstance(chunk_usage, Mapping):
                    raise ValueError("vLLM returned invalid completion_usage")
                usage = dict(chunk_usage)

        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        logger.info("Completed vLLM branch with %d output tokens", len(token_ids))
        return VllmResult(text=text, finish_reason=finish_reason, usage=usage)

    async def _run_parallel(
        self,
        *,
        artifact: Mapping[str, Any],
        vllm_request: Mapping[str, Any],
        context: Any,
    ) -> tuple[dict[str, Any], VllmResult]:
        task_group_type = getattr(asyncio, "TaskGroup", None)
        if task_group_type is not None:
            tasks: list[asyncio.Task[Any]] = []
            try:
                async with task_group_type() as task_group:
                    classifier_task = task_group.create_task(
                        self._collect_single(
                            self._classifier_client,
                            artifact,
                            context,
                        )
                    )
                    tasks.append(classifier_task)
                    vllm_task = task_group.create_task(
                        self._collect_vllm(vllm_request, context)
                    )
                    tasks.append(vllm_task)
            except BaseException as group_error:
                for task in tasks:
                    if not task.cancelled() and task.done():
                        error = task.exception()
                        if error is not None:
                            raise error from group_error
                raise
            return classifier_task.result(), vllm_task.result()

        # Dynamo supports Python 3.10, where asyncio.TaskGroup is unavailable.
        # Preserve TaskGroup's sibling-cancellation semantics on that interpreter.
        classifier_task = asyncio.create_task(
            self._collect_single(self._classifier_client, artifact, context)
        )
        vllm_task = asyncio.create_task(self._collect_vllm(vllm_request, context))
        tasks = (classifier_task, vllm_task)
        try:
            classifier_result, vllm_result = await asyncio.gather(*tasks)
            return classifier_result, vllm_result
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def generate(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        validated = validate_chat_request(request)
        artifact_payload = await self._collect_single(
            self._encoder_client, request, context
        )
        artifact = ExternalQwenArtifact.from_dict(artifact_payload)
        vllm_request = build_vllm_request(
            model=self._backend_model,
            token_ids=artifact.prompt_token_ids,
            external_mm_data=artifact_payload,
            request=validated,
            eos_token_ids=self._eos_token_ids,
        )

        logger.info("Starting parallel classifier and vLLM branches")
        classifier_result, vllm_result = await self._run_parallel(
            artifact=artifact_payload,
            vllm_request=vllm_request,
            context=context,
        )
        text, stopped = apply_stop(vllm_result.text, validated.stop)
        finish_reason = "stop" if stopped else vllm_result.finish_reason

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        yield chat_chunk(request_id=request_id, content=text)
        yield chat_chunk(
            request_id=request_id,
            finish_reason=finish_reason,
            usage=vllm_result.usage,
            classifier=classifier_result,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_BACKEND_MODEL)
    return parser.parse_args()


async def _connect(runtime: DistributedRuntime, endpoint_name: str) -> Client:
    client = await runtime.endpoint(endpoint_name).client()
    await client.wait_for_instances()
    return client


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = _parse_args()
    encoder_client, classifier_client, vllm_client = await asyncio.gather(
        _connect(runtime, VISION_ENCODER_ENDPOINT),
        _connect(runtime, CLASSIFIER_ENDPOINT),
        _connect(runtime, VLLM_ENDPOINT),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    generation_config = GenerationConfig.from_pretrained(args.model)
    handler = OrchestratorHandler(
        backend_model=args.model,
        tokenizer=tokenizer,
        eos_token_ids=_normalize_eos_token_ids(generation_config.eos_token_id),
        encoder_client=encoder_client,
        classifier_client=classifier_client,
        vllm_client=vllm_client,
    )

    endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
    await register_model(
        model_input=ModelInput.Text,
        model_type=ModelType.Chat,
        endpoint=endpoint,
        model_path=args.model,
        model_name=PUBLIC_MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )
    logger.info("Serving public model %s", PUBLIC_MODEL_NAME)
    await endpoint.serve_endpoint(
        handler.generate,
        graceful_shutdown=True,
        metrics_labels=[("service", "multimodal_dag_orchestrator")],
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
