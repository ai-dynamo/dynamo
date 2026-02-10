# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base handler for vLLM-Omni multi-stage pipelines."""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict

from vllm import SamplingParams
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.inputs.data import OmniTextPrompt, OmniTokensPrompt
from vllm_omni.diffusion.data import DiffusionParallelConfig

from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

logger = logging.getLogger(__name__)


class BaseOmniHandler(BaseWorkerHandler):
    """Base handler for multi-stage pipelines using vLLM-Omni's AsyncOmni orchestrator.
    """

    def __init__(
        self,
        runtime,
        component,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
    ):
        """Initialize handler with AsyncOmni orchestrator.

        Args:
            runtime: Dynamo distributed runtime.
            component: Dynamo component handle.
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
        self.default_sampling_params = default_sampling_params
        self.config = config
        self.model_max_len = config.engine_args.max_model_len
        self.shutdown_event = shutdown_event
        self.use_vllm_tokenizer = config.use_vllm_tokenizer

        logger.info(
            f"{self.__class__.__name__} initialized successfully"
        )

    def _build_omni_kwargs(self, config) -> Dict[str, Any]:
        """Build keyword arguments for AsyncOmni constructor.

        Constructs the full kwargs dict including engine-level diffusion
        parameters and parallel configuration when available.

        Args:
            config: Parsed Config object.

        Returns:
            Dictionary of keyword arguments for AsyncOmni.
        """
        omni_kwargs: Dict[str, Any] = {
            "model": config.model,
            "trust_remote_code": config.engine_args.trust_remote_code,
        }

        if config.stage_configs_path:
            omni_kwargs["stage_configs_path"] = config.stage_configs_path

        # Add diffusion engine-level params if present on config
        diffusion_params = [
            "enable_layerwise_offload",
            "layerwise_num_gpu_layers",
            "vae_use_slicing",
            "vae_use_tiling",
            "boundary_ratio",
            "flow_shift",
            "diffusion_cache_backend",
            "diffusion_cache_config",
            "enable_cache_dit_summary",
            "enable_cpu_offload",
            "enforce_eager",
        ]
        for param in diffusion_params:
            if hasattr(config, param):
                value = getattr(config, param)
                if value is not None:
                    # Map config attribute names to AsyncOmni kwarg names
                    kwarg_name = param
                    if param == "diffusion_cache_backend":
                        kwarg_name = "cache_backend"
                    elif param == "diffusion_cache_config":
                        kwarg_name = "cache_config"
                    omni_kwargs[kwarg_name] = value

        # Build DiffusionParallelConfig if parallel params are present
        if hasattr(config, "ulysses_degree"):
            try:
                parallel_config = DiffusionParallelConfig(
                    ulysses_degree=getattr(config, "ulysses_degree", 1),
                    ring_degree=getattr(config, "ring_degree", 1),
                    cfg_parallel_size=getattr(config, "cfg_parallel_size", 1),
                )
                omni_kwargs["parallel_config"] = parallel_config
            except ImportError:
                logger.warning(
                    "DiffusionParallelConfig not available; "
                    "skipping parallel config for AsyncOmni"
                )

        return omni_kwargs

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict, None]:
        """Generate outputs using AsyncOmni orchestrator with OpenAI-compatible format.

        Routes to OpenAI mode (detokenized text) or token mode based on config.
        Subclasses should override ``_generate_openai_mode`` for custom output handling.
        """
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        if self.use_vllm_tokenizer:
            async for chunk in self._generate_openai_mode(request, context, request_id):
                yield chunk
        else:
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_openai_mode(self, request, context, request_id) -> AsyncGenerator[Dict, None]:
        """Generate OpenAI-compatible streaming chunks.

        Subclasses should override this to handle their specific output types.
        The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_openai_mode"
        )

    # Not used right now
    async def _generate_token_mode(self, request, context, request_id):
        """Return token-ids as output. Text input -> Token-ids output."""
        token_ids = request.get("token_ids")
        prompt = OmniTokensPrompt(token_ids=token_ids)
        num_output_tokens_so_far = 0
        try:
            async for stage_output in self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
            ):
                vllm_output = stage_output.request_output

                if not vllm_output.outputs:
                    logger.warning(f"Request {request_id} returned no outputs")
                    yield {
                        "finish_reason": "error: No outputs from vLLM engine",
                        "token_ids": [],
                    }
                    break

                output = vllm_output.outputs[0]
                next_total_toks = len(output.token_ids)

                out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                if output.finish_reason:
                    out["finish_reason"] = self._normalize_finish_reason(
                        output.finish_reason
                    )
                    out["completion_usage"] = self._build_completion_usage(vllm_output)
                    logger.debug(
                        f"Completed generation for request {request_id}: "
                        f"{next_total_toks} output tokens, finish_reason={output.finish_reason}"
                    )

                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason

                yield out
                num_output_tokens_so_far = next_total_toks

        except GeneratorExit:
            logger.info(f"Request {request_id} aborted due to shutdown")
            raise
        except Exception as e:
            logger.error(f"Error during generation for request {request_id}: {e}")
            yield {
                "finish_reason": f"error: {str(e)}",
                "token_ids": [],
            }

    def _format_text_chunk(
        self,
        request_output,
        request_id: str,
        previous_text: str,
    ) -> Dict[str, Any] | None:
        """Format text output as OpenAI chat completion chunk."""
        if not request_output.outputs:
            return self._error_chunk(request_id, "No outputs from engine")

        output = request_output.outputs[0]

        # Calculate delta text (new text since last chunk)
        delta_text = output.text[len(previous_text):]

        chunk = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self.config.served_model_name or self.config.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": self._normalize_finish_reason(output.finish_reason)
                    if output.finish_reason
                    else None,
                }
            ],
        }

        # Add usage on final chunk
        if output.finish_reason:
            chunk["usage"] = self._build_completion_usage(request_output)

        return chunk

    def _extract_text_prompt(self, request: Dict[str, Any]) -> str | None:
        """Extract text prompt from OpenAI messages format.

        Looks for the last user message and returns its text content.
        """
        messages = request.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                return message.get("content")
        return ""

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

    def _error_chunk(self, request_id: str, error_message: str) -> Dict[str, Any]:
        """Create an error chunk in OpenAI format."""
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
