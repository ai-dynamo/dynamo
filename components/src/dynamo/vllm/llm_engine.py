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
from typing import Any, cast

from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.vllm.args import parse_args

from .handlers import build_sampling_params

logger = logging.getLogger(__name__)


class VllmLLMEngine(LLMEngine):
    def __init__(self, engine_args, disaggregation_mode: DisaggregationMode):
        self.engine_args = engine_args
        self.disaggregation_mode = disaggregation_mode
        self.engine_client: AsyncLLM | None = None
        self._vllm_config: Any = None
        self._default_sampling_params: Any = None
        self._prometheus_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._model_max_len: int | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[VllmLLMEngine, WorkerConfig]:
        config = parse_args(argv)

        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            raise NotImplementedError(
                "ENCODE is not supported by the unified vLLM entry point; "
                "use `python -m dynamo.vllm` for multimodal encode workers"
            )

        if not config.served_model_name:
            config.served_model_name = (
                config.engine_args.served_model_name
            ) = config.model

        # _resolve_disaggregation_mode() in DynamoVllmConfig has already
        # promoted the field to a DisaggregationMode enum; the field type
        # is still the input union, so narrow it here for mypy (cast
        # rather than assert so `-O` builds don't drop the narrowing).
        mode = cast(DisaggregationMode, config.disaggregation_mode)
        engine = cls(config.engine_args, mode)
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # vLLM's NixlConnector handles its own per-worker IDs
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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

        # TODO: remove dict() once build_sampling_params accepts GenerateRequest
        sampling_params = build_sampling_params(
            dict(request), self._default_sampling_params, self._model_max_len
        )

        # vLLM's KV transfer is internal to NixlConnector
        # (--kv-transfer-config). Dispatch only sets connector hints and
        # forwards the prefill→decode handoff payload.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            # `do_remote_decode` is prefill's to own; merge caller-supplied
            # values on top of the defaults so explicit overrides win.
            kv_defaults = {
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            }
            caller_kv = sampling_params.extra_args.get("kv_transfer_params", {})
            sampling_params.extra_args["kv_transfer_params"] = {
                **kv_defaults,
                **caller_kv,
                "do_remote_decode": True,
            }
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            prefill_result = require_prefill_result(request, self.disaggregation_mode)
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
            if kv_params is None:
                raise ValueError(
                    "decode worker received prefill_result without "
                    "kv_transfer_params; the prefill peer must populate "
                    "this for vLLM's NixlConnector to pull KV blocks"
                )
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params

        gen = self.engine_client.generate(prompt, sampling_params, request_id)

        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL

        num_output_tokens_so_far: dict[int, int] = {}
        async for res in gen:
            if not res.outputs:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "index": 0,
                    "token_ids": [],
                }
                break

            for output in res.outputs:
                output_idx = getattr(output, "index", 0) or 0
                previous_total = num_output_tokens_so_far.get(output_idx, 0)
                next_total = len(output.token_ids)
                out: GenerateChunk = {
                    "index": output_idx,
                    "token_ids": output.token_ids[previous_total:],
                }

                if output.finish_reason:
                    out["finish_reason"] = str(output.finish_reason)
                    prompt_tokens = (
                        len(res.prompt_token_ids) if res.prompt_token_ids else 0
                    )
                    completion_tokens = sum(
                        len(choice.token_ids) for choice in res.outputs
                    )
                    out["completion_usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    # Stamp the connector's transfer handle on the
                    # prefill terminal so PrefillRouter can forward it.
                    if is_prefill:
                        kv_transfer_params = getattr(res, "kv_transfer_params", None)
                        if kv_transfer_params is not None:
                            out["disaggregated_params"] = {
                                "kv_transfer_params": kv_transfer_params,
                            }

                yield out
                num_output_tokens_so_far[output_idx] = next_total

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine_client is not None and request_id is not None:
            await self.engine_client.abort(request_id)
            logger.debug("Aborted request %s", request_id)

    async def cleanup(self) -> None:
        try:
            if self.engine_client is not None:
                self.engine_client.shutdown()
        finally:
            self.engine_client = None
            if self._prometheus_temp_dir is not None:
                if (
                    os.environ.get("PROMETHEUS_MULTIPROC_DIR")
                    == self._prometheus_temp_dir.name
                ):
                    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
                self._prometheus_temp_dir.cleanup()
                self._prometheus_temp_dir = None
            logger.info("vLLM engine shutdown")
