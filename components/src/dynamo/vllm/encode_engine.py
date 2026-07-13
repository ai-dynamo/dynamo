# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated multimodal Encode engine for the unified vLLM backend."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

from vllm.sampling_params import SamplingParams

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.multimodal import encoder_terminal_chunk
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.llm import ModelInput
from dynamo.vllm.args import Config, parse_args

from .multimodal_handlers import EncodeWorkerHandler
from .multimodal_utils.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)
from .multimodal_utils.request_processor import get_mm_processor_kwargs

_IMAGE_URL_KEY = "image_url"
_URL_VARIANT_KEY = "Url"
_ENCODER_RESULT_SCHEMA_VERSION = 1


def _image_urls(request: GenerateRequest) -> list[str]:
    media = request.get("multi_modal_data") or {}
    urls: list[str] = []
    for item in media.get(_IMAGE_URL_KEY, []):
        if isinstance(item, dict) and isinstance(item.get(_URL_VARIANT_KEY), str):
            urls.append(item[_URL_VARIANT_KEY])
            continue
        raise ValueError(
            "The separate vLLM encode worker supports URL-based images only; "
            "decoded image payloads must be processed by an aggregated/prefill worker"
        )
    return urls


class VllmEncodeEngine(LLMEngine):
    """Image encoder that emits a versioned Local/NIXL transfer handoff."""

    def __init__(self, config: Config):
        self._config = config
        self._handler: EncodeWorkerHandler | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None, config: Config | None = None
    ) -> tuple["VllmEncodeEngine", WorkerConfig]:
        if config is None:
            config = parse_args(argv, fpm_trace_relay_supported=False)
        if config.disaggregation_mode != DisaggregationMode.ENCODE:
            raise ValueError("VllmEncodeEngine requires --disaggregation-mode=encode")
        if not config.enable_multimodal:
            raise ValueError("The vLLM encode worker requires --enable-multimodal")
        if config.route_to_encoder:
            raise ValueError("--route-to-encoder is invalid on an encode worker")
        if not config.served_model_name:
            engine_served_names = config.engine_args.served_model_name
            config.served_model_name = (
                engine_served_names[0] if engine_served_names else config.model
            )
        if not config.engine_args.served_model_name:
            config.engine_args.served_model_name = [config.served_model_name]

        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            enable_kv_routing=False,
            enable_local_indexer=False,
        )
        return cls(config), worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        # Config validation resolves the input union to the enum; narrow the
        # declared field type for mypy at the handler boundary.
        transfer_mode = cast(
            EmbeddingTransferMode, self._config.embedding_transfer_mode
        )
        self._handler = EncodeWorkerHandler(
            self._config.engine_args,
            transfer_mode,
        )
        await self._handler.async_init_unified()
        return EngineConfig(
            model=self._config.model,
            served_model_name=self._config.served_model_name,
            # Encode cards do not use KV metadata, but LLMEngine registration
            # remains token-shaped and expects an llm record.
            llm=LlmRegistration(),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        handler = self._handler
        if handler is None:
            raise RuntimeError(
                "VllmEncodeEngine.start() must complete before generate()"
            )

        groups = [
            MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
            for url in _image_urls(request)
        ]
        encode_request = vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(prompt_token_ids=[]),
            sampling_params=SamplingParams(),
            request_id=context.id(),
            multimodal_inputs=groups,
            mm_processor_kwargs=get_mm_processor_kwargs(request),
        )

        payload: dict[str, Any] | None = None
        async for serialized in handler.generate(encode_request, context):
            payload = json.loads(serialized)
        if payload is None:
            raise RuntimeError("vLLM encode handler returned no result")

        multimodal_inputs = payload.get("multimodal_inputs")
        if not isinstance(multimodal_inputs, list):
            raise RuntimeError(
                "vLLM encode handler returned malformed multimodal_inputs"
            )
        yield encoder_terminal_chunk(
            {
                "schema_version": _ENCODER_RESULT_SCHEMA_VERSION,
                "multimodal_inputs": multimodal_inputs,
            }
        )

    async def cleanup(self) -> None:
        handler, self._handler = self._handler, None
        if handler is None:
            return
        handler.cleanup()
        await asyncio.gather(
            handler.send_complete_checker_task,
            return_exceptions=True,
        )
