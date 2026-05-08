# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
from collections.abc import AsyncGenerator
from typing import Any

from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.sampling_params import GuidedDecodingParams
from torch.cuda import device_count

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions

logger = logging.getLogger(__name__)


class TrtllmLLMEngine(LLMEngine):
    def __init__(
        self,
        engine_args: dict[str, Any],
        model_name: str,
        served_model_name: str | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int | None = None,
        max_num_tokens: int | None = None,
        kv_block_size: int = 32,
    ):
        self.engine_args = engine_args
        self.model_name = model_name
        self.served_model_name = served_model_name
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.kv_block_size = kv_block_size
        self._engine: TensorRTLLMEngine | None = None
        self._default_sampling_params = SamplingParams(detokenize=False)
        self._active_requests: dict[str, Any] = {}

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TrtllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        gpus_per_node = config.gpus_per_node or device_count()

        engine_args = {
            "model": str(config.model),
            "scheduler_config": SchedulerConfig(),
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "backend": Backend.PYTORCH,
            "kv_cache_config": KvCacheConfig(
                free_gpu_memory_fraction=config.free_gpu_memory_fraction,
            ),
            "gpus_per_node": gpus_per_node,
            "max_num_tokens": config.max_num_tokens,
            "max_seq_len": config.max_seq_len,
            "max_beam_width": config.max_beam_width,
            "max_batch_size": config.max_batch_size,
        }

        # Apply --extra-engine-args (YAML) and --override-engine-args (JSON)
        # the same way the legacy `dynamo.trtllm` worker does
        # (workers/llm_worker.py:285-298). Without this, profiler / parallel
        # scheduler caps like `--override-engine-args '{"kv_cache_config":
        # {"max_tokens": N}}'` are silently ignored on the unified path,
        # causing tests to allocate at the engine config's default fraction
        # and OOM the GPU under parallel load.
        if config.extra_engine_args:
            engine_args = update_llm_args_with_extra_options(
                engine_args, config.extra_engine_args
            )
        if config.override_engine_args:
            try:
                overrides = json.loads(config.override_engine_args)
            except json.JSONDecodeError as e:
                logging.error("Failed to parse override_engine_args as JSON: %s", e)
                sys.exit(1)
            if not isinstance(overrides, dict):
                logging.error(
                    "override_engine_args must be a JSON object, got %s",
                    type(overrides).__name__,
                )
                sys.exit(1)
            logging.info("Applying engine arg overrides: %s", overrides)
            warn_override_collisions(engine_args, overrides)
            deep_update(engine_args, overrides)

        # Pull the *post-override* values from engine_args so the engine instance
        # (and the EngineConfig the frontend reads in start()) stays in sync with
        # what the underlying TRT-LLM engine actually got.
        engine = cls(
            engine_args=engine_args,
            model_name=config.model,
            served_model_name=config.served_model_name,
            max_seq_len=engine_args.get("max_seq_len", config.max_seq_len),
            max_batch_size=engine_args.get("max_batch_size", config.max_batch_size),
            max_num_tokens=engine_args.get("max_num_tokens", config.max_num_tokens),
            kv_block_size=config.kv_block_size,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        self._engine = TensorRTLLMEngine(self.engine_args)
        await self._engine.initialize()

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            context_length=self.max_seq_len,
            kv_cache_block_size=self.kv_block_size,
            max_num_seqs=self.max_batch_size,
            max_num_batched_tokens=self.max_num_tokens,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self._engine is not None, "Engine not initialized"

        token_ids = request.get("token_ids", [])
        sampling_params = self._override_sampling_params(
            self._default_sampling_params, request
        )

        stop_conditions = request.get("stop_conditions", {})
        max_tokens = stop_conditions.get("max_tokens")
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        elif self.max_seq_len is not None:
            sampling_params.max_tokens = max(1, self.max_seq_len - len(token_ids))

        ignore_eos = stop_conditions.get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        generation_result = self._engine.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=True,
        )

        request_id = context.id()
        if request_id is not None:
            self._active_requests[request_id] = generation_result

        try:
            # TensorRT-LLM reports cumulative token_ids for each output choice.
            # With n>1, choices are interleaved, so a single cursor would make
            # choice 1 inherit choice 0's offset. Track the emitted length per
            # output index and convert each cumulative list into a Dynamo delta.
            output_tokens_per_choice: dict[int, int] = {}
            async for res in generation_result:
                if not res.outputs and not res.finished:
                    yield {"finish_reason": "error", "token_ids": [], "index": 0}
                    break

                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                    next_total = len(output.token_ids)
                    # The engine returns all tokens generated so far for this
                    # choice. Calculate only the new tokens generated in this
                    # iteration to create the delta.
                    out: GenerateChunk = {
                        "token_ids": output.token_ids[tokens_so_far:],
                        "index": output_idx,
                    }

                    if output.finish_reason:
                        out["finish_reason"] = str(output.finish_reason)

                    if out.get("finish_reason") or res.finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        prompt_tokens = len(token_ids)
                        total_completion_tokens = sum(
                            len(o.token_ids) for o in res.outputs
                        )
                        out["completion_usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": prompt_tokens + total_completion_tokens,
                        }

                    # Yield the chunk to the client and update the token count
                    # for this output choice.
                    yield out
                    output_tokens_per_choice[output_idx] = next_total
        finally:
            if request_id is not None:
                self._active_requests.pop(request_id, None)

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if request_id is not None:
            generation_result = self._active_requests.get(request_id)
            if generation_result is not None:
                generation_result.abort()
                logger.debug("Aborted request %s", request_id)

    async def cleanup(self) -> None:
        if self._engine is not None:
            await self._engine.cleanup()
            logger.info("TensorRT-LLM engine shutdown")

    @staticmethod
    def _override_sampling_params(
        sampling_params: SamplingParams, request: GenerateRequest
    ) -> SamplingParams:
        overrides = {
            key: value
            for key, value in request.get("sampling_options", {}).items()
            if value is not None
        }

        guided_decoding = overrides.pop("guided_decoding", None)
        if guided_decoding is not None and isinstance(guided_decoding, dict):
            regex = guided_decoding.get("regex")
            choice = guided_decoding.get("choice")
            if choice and not regex:
                valid_choices = [c for c in choice if c is not None]
                if valid_choices:
                    regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"
            overrides["guided_decoding"] = GuidedDecodingParams(
                json=guided_decoding.get("json"),
                regex=regex,
                grammar=guided_decoding.get("grammar"),
                json_object=guided_decoding.get("json_object", False),
                structural_tag=guided_decoding.get("structural_tag"),
            )

        n = overrides.get("n")
        if (
            isinstance(n, int)
            and not isinstance(n, bool)
            and n > 1
            and hasattr(sampling_params, "best_of")
        ):
            # Dynamo does not expose best_of here, but TRT-LLM validates that
            # its internal best_of is at least n when cloning SamplingParams.
            # Keep that private field in lockstep so OpenAI n>1 requests do
            # not fail before generation starts.
            best_of = getattr(sampling_params, "best_of", None)
            if best_of is None or best_of < n:
                overrides["best_of"] = n

        return dataclasses.replace(sampling_params, **overrides)
