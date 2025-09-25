# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import signal
import time
import uuid
from enum import Enum

import uvloop
from transformers import AutoTokenizer
from utils.args import Config, parse_args, parse_endpoint
from utils.chat_processor import (
    multimodal_request_to_sglang,
    process_sglang_stream_response,
)

# To import example local module
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils.protocol import MultiModalInput, MultiModalRequest, SglangMultimodalRequest

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class SglangProcessor:
    """
    SGLang pre and post processing for multimodal requests
    """

    @classmethod
    def parse_args(cls) -> Config:
        # Use our unified parser following SGLang backend pattern
        return parse_args(component="processor")

    def __init__(
        self,
        config: Config,
        encode_worker_client: Client,
    ):
        self.encode_worker_client = encode_worker_client
        self.chat_template = getattr(config.server_args, "chat_template", "qwen2-vl")
        self.model = config.model

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

    # Main method to parse the request and send the request to the encode worker.
    async def _generate(
        self,
        raw_request: MultiModalRequest,
        multimodal_input: MultiModalInput,
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
            multimodal_input=multimodal_input,
        )

        # Send to encoder worker
        response_generator = await self.encode_worker_client.round_robin(
            worker_request.model_dump_json()
        )

        # yield "Hello, world!"

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
                    choice = {"index": 0, "delta": {}, "finish_reason": None}

                    # Add role for first message or when there's content
                    if text_content and not finished_sent:
                        choice["delta"]["role"] = "assistant"

                    # Add content if available
                    if text_content:
                        choice["delta"]["content"] = text_content

                    # Set finish reason if completed
                    if is_finished:
                        choice["finish_reason"] = response_data.get(
                            "finish_reason", "stop"
                        )
                        if not finished_sent and not text_content:
                            # Final chunk needs role if it's the first chunk
                            choice["delta"]["role"] = "assistant"

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

    # The generate endpoint will be used by the frontend to handle incoming requests.
    async def generate(self, raw_request: MultiModalRequest):
        if not isinstance(raw_request, MultiModalRequest):
            # If the request is not MultiModalRequest, convert it to MultiModalRequest
            raw_request = MultiModalRequest.model_validate(raw_request)

        multimodal_input = MultiModalInput()

        for message in raw_request.messages:
            for item in message.content:
                if item.type == "image_url":
                    multimodal_input.image_url = item.image_url.url
                elif item.type == "video_url":
                    if multimodal_input.image_url is not None:
                        raise ValueError("Cannot provide both image and video URLs")
                    multimodal_input.video_url = item.video_url.url

        if multimodal_input.image_url is None and multimodal_input.video_url is None:
            raise ValueError("Either image URL or video URL is required")

        async for response in self._generate(raw_request, multimodal_input):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            yield response


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    config = SglangProcessor.parse_args()
    await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        config.downstream_endpoint
    )

    encode_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = SglangProcessor(config, encode_worker_client)

    logger.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model
    await register_llm(
        ModelInput.Text,  # Custom processor is used and this type bypasses SDK processor
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=getattr(config, "block_size", None),
    )

    logger.info(f"Starting to serve the {config.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
