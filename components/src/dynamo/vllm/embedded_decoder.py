# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable native vLLM decoder for in-process Dynamo workers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, cast

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.common.backend import GenerateChunk, GenerateRequest
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.handlers import build_sampling_params
from dynamo.workflow import StageContext, StageContract, ValueSpec

logger = logging.getLogger(__name__)


class EmbeddedVllmDecoder:
    """Run native vLLM as a reusable local stage without another endpoint."""

    contract = StageContract(
        id="embedded-vllm-decoder",
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

    def __init__(
        self,
        *,
        engine: AsyncLLM,
        vllm_config: VllmConfig,
    ) -> None:
        self._engine: AsyncLLM | None = engine
        self._vllm_config = vllm_config
        self._default_sampling_params = (
            vllm_config.model_config.get_diff_sampling_param() or {}
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
    ) -> EmbeddedVllmDecoder:
        """Create the vLLM config and native engine without a Dynamo endpoint."""

        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=[],
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )
        return cls(
            engine=engine,
            vllm_config=vllm_config,
        )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        """Run through the common workflow-stage interface."""

        chunk = await self.generate_final(
            cast(GenerateRequest, inputs["request"]),
            inputs["prompt"],
            context.attempt_id,
        )
        return {"chunk": chunk}

    @property
    def model_config(self) -> ModelConfig:
        """Resolved decoder model configuration for external prompt adapters."""

        return self._vllm_config.model_config

    async def generate_final(
        self,
        request: GenerateRequest,
        prompt: Any,
        request_id: str,
    ) -> GenerateChunk:
        """Run one non-streaming, single-choice decode and return a terminal chunk."""

        prompt_token_ids = prompt.get("prompt_token_ids")
        if not isinstance(prompt_token_ids, list):
            raise InvalidArgument(
                "prepared decoder prompt must contain prompt_token_ids"
            )

        sampling_params = self._build_final_sampling_params(
            request,
            prompt_tokens=len(prompt_token_ids),
        )
        engine = self._require_engine()
        completed = False
        final_output: Any | None = None
        try:
            async for request_output in engine.generate(
                prompt,
                sampling_params,
                request_id,
            ):
                final_output = request_output
            chunk = self._final_output_to_chunk(
                final_output,
                fallback_prompt_tokens=len(prompt_token_ids),
            )
            completed = True
            return chunk
        finally:
            if not completed:
                try:
                    await asyncio.shield(engine.abort(request_id))
                except Exception:
                    logger.exception("Failed to abort vLLM request %s", request_id)

    async def abort(self, request_id: str) -> None:
        """Abort a native request if the decoder is still running."""

        engine = self._engine
        if engine is not None:
            await engine.abort(request_id)

    def shutdown(self) -> None:
        """Shut down the native engine once."""

        engine = self._engine
        self._engine = None
        if engine is not None:
            engine.shutdown()

    def _build_final_sampling_params(
        self,
        request: GenerateRequest,
        *,
        prompt_tokens: int,
    ) -> SamplingParams:
        sampling_options = request.get("sampling_options") or {}
        n = sampling_options.get("n")
        if n not in (None, 1):
            raise InvalidArgument(
                f"EmbeddedVllmDecoder supports exactly one choice; got n={n}"
            )

        output_options = request.get("output_options") or {}
        if (
            output_options.get("logprobs") is not None
            or output_options.get("prompt_logprobs") is not None
        ):
            raise InvalidArgument("EmbeddedVllmDecoder does not support logprobs")

        max_model_len = self.model_config.max_model_len
        available_tokens = max_model_len - prompt_tokens
        if available_tokens < 1:
            raise InvalidArgument(
                "prepared decoder prompt leaves no room for generation: "
                f"prompt_tokens={prompt_tokens}, max_model_len={max_model_len}"
            )

        stop_conditions = request.get("stop_conditions") or {}
        requested_max_tokens = stop_conditions.get("max_tokens")
        if requested_max_tokens is not None and requested_max_tokens > available_tokens:
            raise InvalidArgument(
                "requested max_tokens exceeds the prepared prompt budget: "
                f"max_tokens={requested_max_tokens}, available={available_tokens}"
            )

        sampling_params = build_sampling_params(
            cast(dict[str, Any], request),
            self._default_sampling_params,
            model_max_len=None,
        )
        if requested_max_tokens is None:
            configured_max_tokens = self._default_sampling_params.get("max_tokens")
            sampling_params.max_tokens = min(
                configured_max_tokens
                if configured_max_tokens is not None
                else available_tokens,
                available_tokens,
            )
        sampling_params.n = 1
        sampling_params.detokenize = False
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        return sampling_params

    @staticmethod
    def _final_output_to_chunk(
        final_output: Any | None,
        *,
        fallback_prompt_tokens: int,
    ) -> GenerateChunk:
        if final_output is None or len(final_output.outputs) != 1:
            count = 0 if final_output is None else len(final_output.outputs)
            raise RuntimeError(
                f"vLLM returned {count} outputs; exactly one is required"
            )

        output = final_output.outputs[0]
        if getattr(output, "index", 0) not in (None, 0):
            raise RuntimeError(f"vLLM returned unexpected output index {output.index}")
        finish_reason = getattr(output, "finish_reason", None)
        if not finish_reason:
            raise RuntimeError("vLLM final output did not include a finish reason")

        output_token_ids = list(output.token_ids or [])
        final_prompt_token_ids = getattr(final_output, "prompt_token_ids", None)
        prompt_tokens = (
            len(final_prompt_token_ids)
            if final_prompt_token_ids is not None
            else fallback_prompt_tokens
        )
        completion_tokens = len(output_token_ids)
        return {
            "token_ids": output_token_ids,
            "index": 0,
            "finish_reason": normalize_finish_reason(str(finish_reason)),
            "completion_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _require_engine(self) -> AsyncLLM:
        if self._engine is None:
            raise RuntimeError("EmbeddedVllmDecoder is shut down")
        return self._engine
