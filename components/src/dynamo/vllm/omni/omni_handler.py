# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict

from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.inputs.data import OmniTextPrompt, OmniTokensPrompt

from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params
from vllm import SamplingParams

logger = logging.getLogger(__name__)


class OmniHandler(BaseWorkerHandler):
    """Handler for multi-stage pipelines using vLLM-Omni's AsyncOmni orchestrator."""

    def __init__(
        self,
        runtime,
        component,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
    ):
        """Initialize handler with AsyncOmni orchestrator."""
        logger.info(
            f"Initializing OmniHandler for multi-stage pipelines with model: {config.model}"
        )

        omni_kwargs = {
            "model": config.model,
            "trust_remote_code": config.engine_args.trust_remote_code,
            "stage_configs_path": config.stage_configs_path,
        }

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
        logger.info("OmniHandler initialized successfully for text-to-text generation")

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict, None]:
        """Generate outputs using AsyncOmni orchestrator with OpenAI-compatible format.

        Supports text-to-text and text-to-image generation based on stage configuration.
        Returns OpenAI-compatible streaming chunks with detokenized text.
        """
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        if self.use_vllm_tokenizer:
            async for chunk in self._generate_openai_mode(request, context, request_id):
                yield chunk
        else:
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    # Not used right now
    async def _generate_token_mode(self, request, context, request_id):
        """
        This mode returns token-ids as output
        Text input -> Token-ids output
        """
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
            # Shutdown was triggered during generation
            logger.info(f"Request {request_id} aborted due to shutdown")
            raise
        except Exception as e:
            logger.error(f"Error during generation for request {request_id}: {e}")
            yield {
                "finish_reason": f"error: {str(e)}",
                "token_ids": [],
            }

    async def _generate_openai_mode(self, request, context, request_id):
        """
        This mode returns OpenAI-compatible streaming chunks
        Text input -> Text output / Image output
        """

        # (ayushag) TODO: Support all type of OmniPrompt. Right now it works for only text prompts
        # (ayushag) TODO: Document all I/O formats from vllm omni
        # OmniText prompt support additional negative prompts as well. need to support that as well.
        # Support multimodal content as well. That will involve  applying tokenizer to the prompt and loading images. Follow general multimodal support pattern.
        prompt = self._extract_text_prompt(request)
        prompt = OmniTextPrompt(prompt=prompt)

        # Build sampling parameters from request
        # (ayushag) TODO: Need to add proper multi-stage sampling param support
        # sampling_params = self._build_sampling_params(request)
        # sampling_params_list = [sampling_params]

        previous_text = ""

        async with self._abort_monitor(context, request_id):
            try:
                async for stage_output in self.engine_client.generate(
                    prompt=prompt,
                    request_id=request_id,
                    # sampling_params_list=sampling_params_list,
                ):
                    if (
                        stage_output.final_output_type == "text"
                        and stage_output.request_output
                    ):
                        # Text generation (LLM stage)
                        chunk = self._format_text_chunk(
                            stage_output.request_output,
                            request_id,
                            previous_text,
                        )
                        if chunk:
                            # Update previous_text for delta calculation
                            output = stage_output.request_output.outputs[0]
                            previous_text = output.text
                            yield chunk

                    elif (
                        stage_output.final_output_type == "image"
                        and stage_output.images
                    ):
                        # Image generation (diffusion stage)
                        chunk = self._format_image_chunk(
                            stage_output.images,
                            request_id,
                        )
                        if chunk:
                            yield chunk

            except GeneratorExit:
                logger.info(f"Request {request_id} aborted due to shutdown")
                raise
            except Exception as e:
                logger.error(f"Error during generation for request {request_id}: {e}")
                yield self._error_chunk(request_id, str(e))

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
        delta_text = output.text[len(previous_text) :]

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

    def _format_image_chunk(
        self,
        images: list,
        request_id: str,
    ) -> Dict[str, Any] | None:
        """Format image output as OpenAI chat completion chunk with base64 data URLs."""
        import base64
        from io import BytesIO

        if not images:
            return self._error_chunk(request_id, "No images generated")

        # Convert images to base64 data URLs
        data_urls = []
        for idx, img in enumerate(images):
            # Convert PIL image to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Create data URL (can be opened directly in browser)
            data_url = f"data:image/png;base64,{img_base64}"
            data_urls.append(data_url)
            logger.info(f"Generated image {idx} for request {request_id}")

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
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}}
                            for data_url in data_urls
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        return chunk

    def _extract_text_prompt(self, request: Dict[str, Any]) -> str | None:
        """Extract text prompt from request."""

        # OpenAI messages format - extract text content only
        messages = request.get("messages", [])
        # Assumes single user message
        for message in messages:
            if message.get("role") == "user":
                return message.get("content")
        return ""

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
