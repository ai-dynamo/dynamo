# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from router import RouterAPI, RouterRequest, RouterResponse  # Add this import
from transformers import PreTrainedTokenizerBase
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.transformers_utils.tokenizer import get_tokenizer
from worker import VllmWorkers

from dynamo._core import compute_block_hash_for_seq_py

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingParams:
    model: str
    block_size: int
    num_workers: int
    base_kv_events_port: int
    base_metrics_port: int
    router_port: int
    http_port: int


class ServiceAPI:
    def __init__(self, init_params: ServingParams):
        self.init_params = init_params
        self.app = FastAPI(title="Router API", version="0.0.1")

        # These will be initialized in start()
        self.workers: Optional[VllmWorkers] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.openai_serving_chat: Optional[OpenAIServingChat] = None
        self.model_config: Optional[ModelConfig] = None
        self.http_client: Optional[httpx.AsyncClient] = None

        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if (
                self.workers is None
                or self.tokenizer is None
                or self.openai_serving_chat is None
                or self.http_client is None
            ):
                return ErrorResponse(
                    error={
                        "message": "Service not ready",
                        "type": "service_unavailable",
                        "code": 503,
                    },
                )

            try:
                # Determine max_tokens: use max_completion_tokens first, then max_tokens, or error
                max_tokens_value = None
                if (
                    hasattr(request, "max_completion_tokens")
                    and request.max_completion_tokens is not None
                ):
                    max_tokens_value = request.max_completion_tokens
                elif hasattr(request, "max_tokens") and request.max_tokens is not None:
                    max_tokens_value = request.max_tokens
                else:
                    return ErrorResponse(
                        error={
                            "message": "Either max_tokens or max_completion_tokens must be specified",
                            "type": "invalid_request_error",
                            "code": 400,
                        },
                    )

                # Use vLLM's preprocessing to convert chat to prompt
                # In newer vLLM, _preprocess_chat returns (conversation, engine_prompts) - 2 values
                (
                    conversation,
                    engine_prompts,
                ) = await self.openai_serving_chat._preprocess_chat(
                    request,
                    self.tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.tokenizer.chat_template,
                    chat_template_content_format=self.openai_serving_chat.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    add_special_tokens=False,
                )

                engine_prompt = engine_prompts[0]

                # Convert request to sampling parameters with our determined max_tokens
                sampling_params = request.to_sampling_params(
                    max_tokens=max_tokens_value,
                    logits_processor_pattern=None,
                    default_sampling_params={},
                )

                # Get best worker using HTTP request to router
                tokens: list[int] = engine_prompt["prompt_token_ids"]
                num_tokens = len(tokens)
                if num_tokens == 0:
                    return ErrorResponse(
                        error={
                            "message": "Input prompt is empty",
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    )

                # It is much preferred to communicate block hashes to the router instead of
                # raw text prompts or tokens, especially when over network using pydantic validation,
                # as block hashes can be orders of magnitude smaller.
                # Note that the hashing function needs to be deterministic (across processes),
                # and has to be consistent with the hashing function used to send KV Events to the Router.
                local_hashes = compute_block_hash_for_seq_py(
                    tokens, self.init_params.block_size
                )

                # Call router via HTTP
                try:
                    router_request = RouterRequest(
                        local_hashes=local_hashes, num_tokens=num_tokens
                    )
                    router_response = await self.http_client.post(
                        f"http://localhost:{self.init_params.router_port}/find_best_worker",
                        json=router_request.model_dump(),
                        timeout=1,
                    )

                    router_response.raise_for_status()
                    router_data = RouterResponse.model_validate(router_response.json())
                    best_worker_id = router_data.worker_id

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    logger.error(f"Router request failed: {e}")
                    return ErrorResponse(
                        error={
                            "message": "Router service unavailable",
                            "type": "service_unavailable",
                            "code": 503,
                        }
                    )

                logger.info(f"Selected worker {best_worker_id} for request")

                # Generate request ID
                request_id = f"chatcmpl-{uuid.uuid4()}"
                request_metadata = RequestResponseMetadata(request_id=request_id)

                # Convert engine_prompt dict to TokensPrompt object
                tokens_prompt = TokensPrompt(prompt_token_ids=tokens)
                logger.info(f"Created TokensPrompt with {len(tokens)} tokens")

                # Get the generator from the selected worker with sampling params
                result_generator = self.workers.direct(
                    tokens_prompt, best_worker_id, sampling_params
                )
                assert request.stream

                # Use vLLM's streaming response generator
                return StreamingResponse(
                    self.openai_serving_chat.chat_completion_stream_generator(
                        request,
                        result_generator,
                        request_id,
                        self.init_params.model,
                        conversation,
                        self.tokenizer,
                        request_metadata,
                        enable_force_include_usage=False,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return ErrorResponse(
                    error={"message": str(e), "type": "internal_error", "code": 500}
                )

    async def initialize_services(self):
        """Initialize workers, HTTP client, and OpenAI serving components"""
        logger.info("Initializing VllmWorkers...")
        self.workers = VllmWorkers(
            model=self.init_params.model,
            block_size=self.init_params.block_size,
            base_kv_events_port=self.init_params.base_kv_events_port,
            base_metrics_port=self.init_params.base_metrics_port,
            num_workers=self.init_params.num_workers,
        )

        # Initialize HTTP client for router communication
        self.http_client = httpx.AsyncClient()

        logger.info("Initializing OpenAI serving components...")
        # Initialize tokenizer and model config
        self.tokenizer = get_tokenizer(self.init_params.model)

        # Create a mock model config
        self.model_config = ModelConfig(
            model=self.init_params.model,
            enforce_eager=True,
        )

        # Initialize OpenAI serving models
        base_model_paths = [
            BaseModelPath(
                name=self.init_params.model, model_path=self.init_params.model
            )
        ]
        openai_serving_models = OpenAIServingModels(
            engine_client=None,
            model_config=self.model_config,
            base_model_paths=base_model_paths,
        )

        # Initialize OpenAI serving chat
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=None,
            model_config=self.model_config,
            models=openai_serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        logger.info("Waiting 2 seconds for services to initialize...")
        await asyncio.sleep(2)
        logger.info("Services initialized successfully!")

    async def start(self):
        """Start the API server"""
        # Initialize services first
        await self.initialize_services()

        # Start the API server
        logger.info(f"Starting API server on port {self.init_params.http_port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.init_params.http_port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self):
        """Proper shutdown handler"""
        logger.info("Shutting down API...")

        if self.http_client:
            await self.http_client.aclose()

        logger.info("API shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="Router API Server")

    # Arguments from worker.py
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to use",
    )

    # Common arguments
    parser.add_argument(
        "--block-size", type=int, default=64, help="Block size for caching"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--base-kv-events-port", type=int, default=5557, help="Base port for KV events"
    )
    parser.add_argument(
        "--base-metrics-port", type=int, default=5657, help="Base port for metrics"
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=7000,
        help="Port for router service",
    )
    parser.add_argument(
        "--http-port", type=int, default=8000, help="Port to serve the API on"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    init_params = ServingParams(
        model=args.model,
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        router_port=args.router_port,
        http_port=args.http_port,
    )

    # Create both services
    api = ServiceAPI(init_params=init_params)
    router_api = RouterAPI(
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        port=args.router_port,
    )

    async def run_with_shutdown():
        try:
            # Start both services concurrently
            await asyncio.gather(
                api.start(), router_api.start(), return_exceptions=True
            )
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down services...")
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
        finally:
            await api.shutdown()

    try:
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        # Just in case KeyboardInterrupt happens outside of the event loop
        logger.info("Force shutdown via KeyboardInterrupt.")


if __name__ == "__main__":
    main()
