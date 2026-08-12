# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow adapter for an in-process vLLM decoder runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, cast

from vllm.config import ModelConfig

from dynamo.common.backend import GenerateChunk, GenerateRequest
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.decoder_runtime import DecodeOutputMode, VllmDecoderRuntime
from dynamo.workflow import StageContext, StageContract, ValueSpec

logger = logging.getLogger(__name__)


class VllmDecoderStage:
    """Adapt a borrowed decoder runtime to the local workflow stage contract."""

    contract = StageContract(
        id="vllm-decoder",
        inputs={
            "request": ValueSpec(
                type="object", class_id="dynamo.common.backend.GenerateRequest"
            ),
            "prompt": ValueSpec(type="object", class_id="dynamo.vllm.PreparedPrompt"),
        },
        outputs={
            "chunk": ValueSpec(
                type="object", class_id="dynamo.common.backend.GenerateChunk"
            )
        },
    )

    def __init__(self, runtime: VllmDecoderRuntime) -> None:
        self._runtime = runtime
        self._active_requests: dict[str, str] = {}

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        native_request_id = self._native_request_id(context)
        if context.attempt_id in self._active_requests:
            raise RuntimeError(
                f"decoder attempt {context.attempt_id!r} is already active"
            )
        self._active_requests[context.attempt_id] = native_request_id
        try:
            chunk = await self._generate_final(
                cast(GenerateRequest, inputs["request"]),
                inputs["prompt"],
                native_request_id,
            )
            return {"chunk": chunk}
        finally:
            self._active_requests.pop(context.attempt_id, None)

    @property
    def model_config(self) -> ModelConfig:
        """Resolved decoder model configuration for prompt adapters."""

        return self._runtime.model_config

    async def abort_attempt(self, attempt_id: str) -> None:
        """Abort this stage's active native request for a workflow attempt."""

        request_id = self._active_requests.get(attempt_id)
        if request_id is not None:
            await self._runtime.abort(request_id)

    async def _generate_final(
        self,
        request: GenerateRequest,
        prompt: Any,
        request_id: str,
    ) -> GenerateChunk:
        prompt_token_ids = prompt.get("prompt_token_ids")
        if not isinstance(prompt_token_ids, list):
            raise InvalidArgument(
                "prepared decoder prompt must contain prompt_token_ids"
            )

        self._validate_final_request(request)
        sampling_params = self._runtime.prepare_sampling_params(
            request,
            prompt=prompt,
            output_mode=DecodeOutputMode.FINAL_ONLY,
        )
        completed = False
        final_chunk: GenerateChunk | None = None
        try:
            async for chunk in self._runtime.decode(
                prompt,
                sampling_params,
                request_id,
            ):
                if final_chunk is not None:
                    raise RuntimeError(
                        "vLLM returned multiple chunks for final-only generation"
                    )
                final_chunk = chunk
            if final_chunk is None:
                raise RuntimeError("vLLM returned no final output")
            finish_reason = final_chunk.get("finish_reason")
            if not finish_reason:
                raise RuntimeError("vLLM final output did not include a finish reason")
            if str(finish_reason).startswith("error"):
                raise RuntimeError(str(finish_reason))
            if final_chunk.get("index", 0) != 0:
                raise RuntimeError(
                    "vLLM returned unexpected output index "
                    f"{final_chunk.get('index')}"
                )
            completed = True
            return final_chunk
        finally:
            if not completed:
                try:
                    await asyncio.shield(self._runtime.abort(request_id))
                except Exception:
                    logger.exception("Failed to abort vLLM request %s", request_id)

    @staticmethod
    def _validate_final_request(request: GenerateRequest) -> None:
        sampling_options = request.get("sampling_options") or {}
        n = sampling_options.get("n")
        if n not in (None, 1):
            raise InvalidArgument(
                f"VllmDecoderStage supports exactly one choice; got n={n}"
            )

        output_options = request.get("output_options") or {}
        if (
            output_options.get("logprobs") is not None
            or output_options.get("prompt_logprobs") is not None
        ):
            raise InvalidArgument("VllmDecoderStage does not support logprobs")

    @staticmethod
    def _native_request_id(context: StageContext) -> str:
        return f"{context.attempt_id}:{context.stage_id}"
