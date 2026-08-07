# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable native vLLM decoder for in-process Dynamo workers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol, cast

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.common.backend import GenerateChunk, GenerateRequest, LlmRegistration
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.handlers import build_sampling_params
from dynamo.vllm.multimodal_utils.custom_encoder import (
    CustomEncoderAdapter,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)

logger = logging.getLogger(__name__)


class DecoderEngine(Protocol):
    """Native engine operations required by :class:`EmbeddedVllmDecoder`."""

    def generate(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[Any]:
        ...

    async def abort(self, request_id: str) -> None:
        ...

    def shutdown(self) -> None:
        ...


class EmbeddedVllmDecoder:
    """Run native vLLM as a child component without registering an endpoint."""

    def __init__(
        self,
        *,
        engine: DecoderEngine,
        engine_args: AsyncEngineArgs,
        vllm_config: VllmConfig,
        default_sampling_params: dict[str, Any],
    ) -> None:
        self._engine: DecoderEngine | None = engine
        self._engine_args = engine_args
        self._vllm_config = vllm_config
        self._default_sampling_params = default_sampling_params

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        *,
        usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    ) -> EmbeddedVllmDecoder:
        """Create the vLLM config and native engine without a Dynamo endpoint."""

        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
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
            engine_args=engine_args,
            vllm_config=vllm_config,
            default_sampling_params=(
                vllm_config.model_config.get_diff_sampling_param()
            ),
        )

    def create_prompt_adapter(
        self,
        backend: VisionEncoderBackend[Any, Any, Any],
    ) -> CustomEncoderAdapter[Any]:
        """Create the Dynamo adapter that maps encoder artifacts to this decoder."""

        return create_custom_encoder_adapter(
            backend,
            self._vllm_config.model_config,
            self._engine_args,
        )

    def registration(self) -> LlmRegistration:
        """Return model capacity metadata for the containing Dynamo worker."""

        scheduler_config = self._vllm_config.scheduler_config
        return LlmRegistration(
            context_length=self._vllm_config.model_config.max_model_len,
            kv_cache_block_size=self._vllm_config.cache_config.block_size,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            data_parallel_size=1,
            data_parallel_start_rank=0,
        )

    async def generate_final(
        self,
        request: GenerateRequest,
        prompt: Any,
        request_id: str,
    ) -> GenerateChunk:
        """Run one non-streaming, single-choice decode and return a terminal chunk."""

        prompt_token_ids = prompt.get("prompt_token_ids")
        if not isinstance(prompt_token_ids, list):
            raise InvalidArgument("decoder prompt must contain prompt_token_ids")

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
            if final_output is None or len(final_output.outputs) != 1:
                count = 0 if final_output is None else len(final_output.outputs)
                raise RuntimeError(
                    f"vLLM returned {count} outputs; exactly one is required"
                )

            output = final_output.outputs[0]
            if getattr(output, "index", 0) not in (None, 0):
                raise RuntimeError(
                    f"vLLM returned unexpected output index {output.index}"
                )
            finish_reason = getattr(output, "finish_reason", None)
            if not finish_reason:
                raise RuntimeError("vLLM final output did not include a finish reason")

            output_token_ids = list(output.token_ids or [])
            final_prompt_token_ids = getattr(final_output, "prompt_token_ids", None)
            prompt_tokens = (
                len(final_prompt_token_ids)
                if final_prompt_token_ids is not None
                else len(prompt_token_ids)
            )
            completion_tokens = len(output_token_ids)
            chunk: GenerateChunk = {
                "token_ids": output_token_ids,
                "index": 0,
                "finish_reason": str(finish_reason),
                "completion_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            completed = True
            return chunk
        finally:
            if not completed:
                try:
                    await asyncio.shield(engine.abort(request_id))
                except Exception:
                    logger.exception("Failed to abort vLLM request %s", request_id)

    async def abort(self, request_id: str) -> None:
        engine = self._engine
        if engine is not None:
            await engine.abort(request_id)

    def shutdown(self) -> None:
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

        sampling_params = build_sampling_params(
            cast(dict[str, Any], request),
            self._default_sampling_params,
            model_max_len=None,
        )
        stop_conditions = request.get("stop_conditions") or {}
        if stop_conditions.get("max_tokens") is None:
            available = max(
                1,
                self._vllm_config.model_config.max_model_len - prompt_tokens,
            )
            configured = self._default_sampling_params.get("max_tokens", available)
            if configured is None:
                configured = available
            sampling_params.max_tokens = min(configured, available)

        sampling_params.n = 1
        sampling_params.detokenize = False
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        return sampling_params

    def _require_engine(self) -> DecoderEngine:
        if self._engine is None:
            raise RuntimeError("EmbeddedVllmDecoder is shut down")
        return self._engine
