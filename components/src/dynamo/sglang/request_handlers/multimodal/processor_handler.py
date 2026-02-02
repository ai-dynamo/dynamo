# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
import uuid
from typing import Any, Dict, List

from transformers import AutoTokenizer

from dynamo._core import Client, Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils import (
    multimodal_request_to_sglang,
    process_sglang_stream_response,
)
from dynamo.sglang.protocol import (
    MultiModalInput,
    MultiModalInputGroup,
    MultiModalRequest,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

logger = logging.getLogger(__name__)


class MultimodalProcessorHandler(BaseWorkerHandler):
    """
    Handler for multimodal processor component that processes multimodal requests
    and forwards them to the encode worker.
    """

    def __init__(
        self,
        component: Component,
        config: Config,
        encode_worker_client: Client,
    ):
        super().__init__(component, engine=None, config=config)
        self.encode_worker_client = encode_worker_client
        self.chat_template = getattr(config.server_args, "chat_template", "qwen2-vl")
        self.model = config.server_args.model_path

        # Initialize tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )

    def cleanup(self):
        pass

    async def generate(self, raw_request: MultiModalRequest, context: Context):
        """
        Process multimodal request and forward to encode worker.

        Supports multiple images in a single request. Each image is collected
        into a MultiModalInputGroup for individual processing by the encode worker.

        Args:
            raw_request: Raw multimodal request to process.
            context: Context object for cancellation handling.
        """
        if not isinstance(raw_request, MultiModalRequest):
            # If the request is not MultiModalRequest, convert it to MultiModalRequest
            raw_request = MultiModalRequest.model_validate(raw_request)

        # Collect all multimodal inputs from the request
        # Each image/video gets its own MultiModalInputGroup for individual encoding
        multimodal_inputs: List[MultiModalInputGroup] = []
        has_video = False

        for message in raw_request.messages:
            for item in message.content:
                if item.type == "image_url":
                    if has_video:
                        raise ValueError(
                            "Cannot mix images and videos in the same request"
                        )
                    multimodal_inputs.append(
                        MultiModalInputGroup(
                            multimodal_input=MultiModalInput(
                                image_url=item.image_url.url
                            )
                        )
                    )
                elif item.type == "video_url":
                    if multimodal_inputs:
                        raise ValueError(
                            "Cannot mix images and videos in the same request"
                        )
                    has_video = True
                    multimodal_inputs.append(
                        MultiModalInputGroup(
                            multimodal_input=MultiModalInput(
                                video_url=item.video_url.url
                            )
                        )
                    )

        if not multimodal_inputs:
            raise ValueError("At least one image URL or video URL is required")

        logger.debug(f"Collected {len(multimodal_inputs)} multimodal input(s)")

        async for response in self._generate(raw_request, multimodal_inputs):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            yield response

    async def _generate(
        self,
        raw_request: MultiModalRequest,
        multimodal_inputs: List[MultiModalInputGroup],
    ):
        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"Got raw request: {raw_request}")

        # Create SGLang conversation prompt
        sglang_request = multimodal_request_to_sglang(
            raw_request, self.tokenizer, self.chat_template
        )

        worker_request = SglangMultimodalRequest(
            request=sglang_request,
            multimodal_inputs=multimodal_inputs,
        )

        # Send to encoder worker
        response_generator = await self.encode_worker_client.round_robin(
            worker_request.model_dump_json()
        )

        # Process and yield SGLang responses
        finished_sent = False
        accumulated_text = ""

        async for resp in response_generator:
            try:
                # Handle Annotated response objects from Dynamo (like vLLM pattern but for SGLang)
                if hasattr(resp, "data"):
                    # Extract data from Dynamo Annotated response
                    raw_data = resp.data
                    if callable(raw_data):
                        raw_data = raw_data()

                    if isinstance(raw_data, str):
                        try:
                            response_data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            response_data = {"text": raw_data, "finished": False}
                    else:
                        response_data = raw_data
                elif isinstance(resp, str):
                    try:
                        response_data = json.loads(resp)
                    except json.JSONDecodeError:
                        response_data = {"text": resp, "finished": False}
                else:
                    response_data = resp

                # Use SGLang chat_processor for detokenization
                (
                    text_content,
                    accumulated_text,
                    is_finished,
                ) = process_sglang_stream_response(
                    response_data, self.tokenizer, accumulated_text
                )

                # Create OpenAI-compatible response (following vLLM-like pattern but for SGLang)
                if text_content or is_finished:
                    choice: Dict[str, Any] = {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                    delta: Dict[str, str] = choice["delta"]  # Type-safe access

                    # Add role for first message or when there's content
                    if text_content and not finished_sent:
                        delta["role"] = "assistant"

                    # Add content if available
                    if text_content:
                        delta["content"] = text_content

                    # Set finish reason if completed
                    if is_finished:
                        choice["finish_reason"] = response_data.get(
                            "finish_reason", "stop"
                        )
                        if not finished_sent and not text_content:
                            # Final chunk needs role if it's the first chunk
                            delta["role"] = "assistant"

                    response_json = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model,
                        "choices": [choice],
                    }

                    # Add usage only for final response
                    if is_finished:
                        response_json["usage"] = {
                            "prompt_tokens": 0,
                            "completion_tokens": len(accumulated_text.split())
                            if accumulated_text
                            else 0,
                            "total_tokens": len(accumulated_text.split())
                            if accumulated_text
                            else 0,
                        }

                    yield response_json

                    if is_finished:
                        finished_sent = True
                        break

            except Exception as e:
                logger.error(f"Error processing SGLang response: {e}")
                error_response = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"Error: {str(e)}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield error_response
                break
