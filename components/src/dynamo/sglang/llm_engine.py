# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, Dict

import sglang as sgl

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import ModelInput
from dynamo.sglang.args import parse_args

logger = logging.getLogger(__name__)


class SglangLLMEngine(LLMEngine):
    def __init__(self, server_args):
        self.server_args = server_args
        self.engine = None
        self._input_param_manager = None
        self._skip_tokenizer_init = server_args.skip_tokenizer_init

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SglangLLMEngine, WorkerConfig]:
        config = await parse_args(argv if argv is not None else sys.argv[1:])
        server_args = config.server_args
        dynamo_args = config.dynamo_args

        model_input = (
            ModelInput.Text
            if not server_args.skip_tokenizer_init
            else ModelInput.Tokens
        )

        engine = cls(server_args)
        worker_config = WorkerConfig.from_runtime_config(
            dynamo_args,
            model_name=server_args.model_path,
            served_model_name=server_args.served_model_name,
            model_input=model_input,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        self.engine = sgl.Engine(server_args=self.server_args)

        tokenizer = (
            self.engine.tokenizer_manager.tokenizer
            if not self._skip_tokenizer_init
            else None
        )
        self._input_param_manager = InputParamManager(tokenizer)

        # Capacity fields -- sourced the same way as register.py in the
        # non-unified path so the Rust runtime gets consistent values.
        total_kv_blocks = None
        scheduler_info = getattr(self.engine, "scheduler_info", None) or {}
        max_total_tokens = scheduler_info.get("max_total_num_tokens")
        page_size = self.server_args.page_size
        if max_total_tokens and page_size:
            total_kv_blocks = (max_total_tokens + page_size - 1) // page_size

        return EngineConfig(
            model=self.server_args.model_path,
            served_model_name=self.server_args.served_model_name,
            context_length=self.server_args.context_length,
            kv_cache_block_size=page_size,
            total_kv_blocks=total_kv_blocks,
            max_num_seqs=getattr(self.server_args, "max_running_requests", None),
            max_num_batched_tokens=getattr(
                self.server_args, "max_prefill_tokens", None
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None, "Engine not initialized"

        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            rid=context.trace_id,
        )

        async for res in stream:
            out: Dict[str, Any] = {}
            meta_info = res["meta_info"]
            finish_reason = meta_info["finish_reason"]

            output_ids = res.get("output_ids", [])
            if not output_ids and not finish_reason:
                if context.is_stopped():
                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                    completion_tokens = meta_info.get("completion_tokens", 0)
                    yield {
                        "token_ids": [],
                        "finish_reason": "cancelled",
                        "completion_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    break
                continue

            out["token_ids"] = output_ids

            if finish_reason:
                prompt_tokens = meta_info["prompt_tokens"]
                completion_tokens = meta_info["completion_tokens"]
                out["finish_reason"] = finish_reason["type"]
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            if context.is_stopped():
                prompt_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                yield {
                    "token_ids": output_ids,
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                break

            yield out

    async def abort(self, context: Context) -> None:
        rid = context.trace_id
        if self.engine is not None and rid is not None:
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                self.engine.tokenizer_manager.abort_request(rid=rid, abort_all=False)
                logger.debug("Aborted request %s", rid)

    async def cleanup(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            logger.info("SGLang engine shutdown")

    def _build_sampling_params(self, request: dict) -> dict:
        if self._skip_tokenizer_init:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}

    def _get_input_param(self, request: dict) -> dict:
        assert self._input_param_manager is not None, "Engine not initialized"
        request_input = self._input_param_manager.get_input_param(
            request, use_tokenizer=not self._skip_tokenizer_init
        )
        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }
