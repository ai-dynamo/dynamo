# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base handler for vLLM-Omni multi-stage pipelines."""

import asyncio
import dataclasses
import logging
import time
from typing import Any, AsyncGenerator, Dict

from vllm import SamplingParams
from vllm_omni.entrypoints import AsyncOmni

try:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
except ImportError:
    DiffusionParallelConfig = None  # type: ignore[assignment, misc]

from dynamo._core import Context
from dynamo.common.protocols.audio_protocol import NvAudioSpeechResponse
from dynamo.common.utils.output_modalities import RequestType
from dynamo.vllm.args import get_served_model_name
from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

logger = logging.getLogger(__name__)


class BaseOmniHandler(BaseWorkerHandler[Dict[str, Any], Dict[str, Any]]):
    """Base handler for multi-stage pipelines using vLLM-Omni's AsyncOmni orchestrator."""

    def __init__(
        self,
        runtime,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
    ):
        """Initialize handler with AsyncOmni orchestrator.

        Args:
            runtime: Dynamo distributed runtime.
            config: Parsed Config object from args.py.
            default_sampling_params: Default sampling parameters dict.
            shutdown_event: Optional asyncio event for graceful shutdown.
        """
        logger.info(
            f"Initializing {self.__class__.__name__} for multi-stage pipelines "
            f"with model: {config.model}"
        )

        omni_kwargs = self._build_omni_kwargs(config)
        self.engine_client = AsyncOmni(**omni_kwargs)

        # Initialize attributes needed from BaseWorkerHandler
        # We don't call super().__init__() because VllmEngineMonitor expects AsyncLLM,
        # but AsyncOmni manages its own engines internally

        # TODO: Kv publishers not supported yet
        # TODO: Adopt to baseworker initialization pattern
        self.runtime = runtime
        self.default_sampling_params = default_sampling_params
        self.config = config
        self.model_max_len = config.engine_args.max_model_len
        self.shutdown_event = shutdown_event

        logger.info(f"{self.__class__.__name__} initialized successfully")

    def _build_omni_kwargs(self, config) -> Dict[str, Any]:
        """Build keyword arguments for AsyncOmni constructor."""
        omni_kwargs: Dict[str, Any] = {
            "model": config.model,
            "trust_remote_code": config.engine_args.trust_remote_code,
        }

        if config.stage_configs_path:
            omni_kwargs["stage_configs_path"] = config.stage_configs_path

        for field, value in dataclasses.asdict(config.diffusion).items():
            if value is not None:
                omni_kwargs[field] = value

        # tensor_parallel_size comes from engine_args (vLLM's --tensor-parallel-size)
        if DiffusionParallelConfig is not None:
            parallel_config = DiffusionParallelConfig(
                tensor_parallel_size=getattr(
                    config.engine_args, "tensor_parallel_size", 1
                ),
                **dataclasses.asdict(config.parallel),
            )
            omni_kwargs["parallel_config"] = parallel_config
        else:
            logger.warning(
                "DiffusionParallelConfig not available; "
                "skipping parallel config for AsyncOmni"
            )

        return omni_kwargs

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate outputs using AsyncOmni orchestrator with OpenAI-compatible format.

        Subclasses should override ``_generate_openai_mode`` for custom output handling.
        """
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        async for chunk in self._generate_openai_mode(request, context, request_id):
            yield chunk

    async def _generate_openai_mode(
        self, request, context, request_id
    ) -> AsyncGenerator[Dict, None]:
        """Generate OpenAI-compatible streaming chunks.

        Subclasses should override this to handle their specific output types.
        The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_openai_mode"
        )
        # Make this a proper async generator so the return type is correct.
        yield  # pragma: no cover

    def _extract_text_prompt(self, request: Dict[str, Any]) -> str | None:
        """Extract text prompt from OpenAI messages format.

        Looks for the last user message and returns its text content.
        """
        messages = request.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content")
        return None

    def _extract_extra_body(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract extra_body parameters from the request.

        The extra_body is passed through by the OpenAI client and contains
        model-specific parameters (e.g. diffusion sampling params).
        """
        return request.get("extra_body", {}) or {}

    def _build_sampling_params(self, request: Dict[str, Any]) -> SamplingParams:
        """Build sampling params using shared handler utility."""
        return build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

    def _error_chunk(
        self,
        request_id: str,
        error_message: str,
        request_type=None,
    ) -> Dict[str, Any]:
        """Create an error response matching the expected protocol for the request type.

        For AUDIO_GENERATION returns NvAudioSpeechResponse format.
        For all other types returns OpenAI chat.completion.chunk format.
        """
        if request_type == RequestType.AUDIO_GENERATION:
            return NvAudioSpeechResponse(
                id=request_id,
                model=get_served_model_name(
                    self.config.model, self.config.served_model_name
                ),
                status="failed",
                created=int(time.time()),
                error=error_message,
            ).model_dump()

        return {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self.config.served_model_name or self.config.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"Error: {error_message}",
                    },
                    "finish_reason": "error",
                }
            ],
        }

    def cleanup(self):
        """Cleanup AsyncOmni orchestrator resources."""
        try:
            if hasattr(self, "engine_client"):
                self.engine_client.close()
                logger.info("AsyncOmni orchestrator closed")
        except Exception as e:
            logger.error(f"Error closing AsyncOmni orchestrator: {e}")
