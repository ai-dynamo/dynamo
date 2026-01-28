# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Omni handler for text-to-image POC."""

import base64
import io
import json
import logging
from typing import Any, AsyncGenerator, Dict

from vllm_omni.entrypoints import AsyncOmni

logger = logging.getLogger(__name__)


class OmniHandler:
    """
    Omni handler for text-to-image generation.
    """

    def __init__(self, config):
        """Initialize AsyncOmni engine.

        Args:
            config: Configuration object with model path and settings
        """
        logger.info(f"Initializing OmniHandler for model: {config.model}")

        self.omni = AsyncOmni(
            model=config.model,
            worker_backend="multi_process",
            trust_remote_code=True,
        )

        logger.info("OmniHandler initialized successfully")

    async def generate(self, request: Dict[str, Any], context) -> AsyncGenerator[Dict, None]:
        """Generate images from text prompt.

        Args:
            request: Chat completion request with messages
            context: Request context

        Yields:
            Dict: Response with base64-encoded images
        """
        request_id = context.id()

        # Extract prompt from messages
        messages = request.get("messages", [])
        prompt = self._get_prompt_from_messages(messages)

        logger.info(f"Request {request_id}: Generating image for prompt: {prompt[:100]}")

        sampling_params = {}
        if "size" in request:
            try:
                w, h = request["size"].split("x")
                sampling_params["width"] = int(w)
                sampling_params["height"] = int(h)
            except:
                pass

        extra = request.get("extra_body", {})
        if "num_inference_steps" in extra:
            sampling_params["num_inference_steps"] = extra["num_inference_steps"]

        async for output in self.omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=[sampling_params],
        ):
            images = output.request_output.images

            image_urls = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{b64}")

            # Build chat completion response
            chat_response = {
                "id": request_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}}
                                for url in image_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

            # Return in LLMEngineOutput format for Rust
            # - output_type: "image" signals Rust to skip detokenization
            # - text: contains the complete JSON response
            # - token_ids: empty (no tokens for image generation)
            yield {
                "token_ids": [],
                "tokens": None,
                "text": json.dumps(chat_response),
                "cum_log_probs": None,
                "log_probs": None,
                "top_logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None,
                "index": 0,
                "output_type": "image",
            }

    def _get_prompt_from_messages(self, messages: list) -> str:
        """Extract text prompt from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""
