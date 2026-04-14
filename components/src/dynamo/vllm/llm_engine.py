# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import AsyncGenerator

from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.llm import ModelInput
from dynamo.vllm.args import parse_args

from .handlers import build_sampling_params

logger = logging.getLogger(__name__)


class VllmLLMEngine(LLMEngine):
    def __init__(self, engine_args):
        self.engine_args = engine_args
        self.engine_client = None
        self._vllm_config = None
        self._default_sampling_params = None
        self._prometheus_temp_dir = None
        self._model_max_len = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[VllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        if not config.served_model_name:
            config.served_model_name = (
                config.engine_args.served_model_name
            ) = config.model

        engine = cls(config.engine_args)
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            self._prometheus_temp_dir = tempfile.TemporaryDirectory(
                prefix="vllm_prometheus_"
            )
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = self._prometheus_temp_dir.name

        self._default_sampling_params = (
            self.engine_args.create_model_config().get_diff_sampling_param()
        )

        vllm_config = self.engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        self._vllm_config = vllm_config

        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        self._model_max_len = getattr(
            getattr(vllm_config, "model_config", None), "max_model_len", None
        )

        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        block_size = vllm_config.cache_config.block_size

        return EngineConfig(
            model=self.engine_args.model,
            served_model_name=self.engine_args.served_model_name,
            context_length=self._model_max_len,
            kv_cache_block_size=block_size,
            total_kv_blocks=num_gpu_blocks,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine_client is not None, "Engine not initialized"
        assert self._default_sampling_params is not None, "Engine not initialized"

        request_id = context.id()

        token_ids = request.get("token_ids", [])
        prompt = TokensPrompt(prompt_token_ids=token_ids)

        sampling_params = build_sampling_params(
            request, self._default_sampling_params, self._model_max_len
        )

        gen = self.engine_client.generate(prompt, sampling_params, request_id)

        num_output_tokens_so_far = 0
        async for res in gen:
            if not res.outputs:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "token_ids": [],
                }
                break

            output = res.outputs[0]
            next_total = len(output.token_ids)
            out: dict = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

            if output.finish_reason:
                out["finish_reason"] = str(output.finish_reason)
                prompt_tokens = len(res.prompt_token_ids) if res.prompt_token_ids else 0
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": next_total,
                    "total_tokens": prompt_tokens + next_total,
                }

            yield out
            num_output_tokens_so_far = next_total

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine_client is not None and request_id is not None:
            await self.engine_client.abort(request_id)
            logger.debug("Aborted request %s", request_id)

    async def cleanup(self) -> None:
        if self.engine_client is not None:
            self.engine_client.shutdown()
        if self._prometheus_temp_dir is not None:
            self._prometheus_temp_dir.cleanup()
        logger.info("vLLM engine shutdown")
