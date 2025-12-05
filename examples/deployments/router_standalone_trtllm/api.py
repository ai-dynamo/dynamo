# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Fix protobuf version conflict with etcd3
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from router import RouterAPI, RouterRequest, RouterResponse
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from worker import TrtllmWorkers

from dynamo._core import compute_block_hash_for_seq_py

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: bool = True


class ErrorResponse(BaseModel):
    error: dict


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
        self.app = FastAPI(title="TensorRT-LLM Router API", version="0.0.1")

        self.workers: Optional[TrtllmWorkers] = None
        self.tokenizer = None
        self.http_client: Optional[httpx.AsyncClient] = None

        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if (
                self.workers is None
                or self.tokenizer is None
                or self.http_client is None
            ):
                return ErrorResponse(
                    error={
                        "message": "Service not ready",
                        "type": "service_unavailable",
                        "code": 503,
                    }
                )

            try:
                max_tokens_value = None
                if request.max_completion_tokens is not None:
                    max_tokens_value = request.max_completion_tokens
                elif request.max_tokens is not None:
                    max_tokens_value = request.max_tokens
                else:
                    return ErrorResponse(
                        error={
                            "message": "Either max_tokens or max_completion_tokens must be specified",
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    )

                messages_dict = [
                    {"role": msg.role, "content": msg.content} for msg in request.messages
                ]

                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages_dict, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply chat template: {e}, falling back to simple formatting"
                    )
                    prompt = self._format_messages_simple(messages_dict)

                tokens = self.tokenizer.encode(prompt)
                num_tokens = len(tokens)

                if num_tokens == 0:
                    return ErrorResponse(
                        error={
                            "message": "Input prompt is empty",
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    )

                logger.info(
                    f"API: Tokenized {num_tokens} tokens, first 10: {tokens[:10]}, "
                    f"block_size={self.init_params.block_size}"
                )

                local_hashes = compute_block_hash_for_seq_py(
                    tokens, self.init_params.block_size
                )
                logger.info(f"API: Computed {len(local_hashes)} local_hashes: {local_hashes}")

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

                request_id = f"chatcmpl-{uuid.uuid4()}"

                sampling_params = {
                    "max_tokens": max_tokens_value,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                }

                result_generator = self.workers.direct(
                    tokens, best_worker_id, sampling_params
                )

                return StreamingResponse(
                    self._chat_completion_stream_generator(
                        request, result_generator, request_id
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return ErrorResponse(
                    error={"message": str(e), "type": "internal_error", "code": 500}
                )

    def _format_messages_simple(self, messages: list[dict]) -> str:
        """Simple fallback formatting when chat template is not available."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        return formatted

    async def _chat_completion_stream_generator(
        self, request: ChatCompletionRequest, result_generator, request_id: str
    ):
        """Generate SSE formatted streaming responses."""
        created = int(time.time())
        first_chunk = True

        try:
            async for output in result_generator:
                if hasattr(output, "text_diff"):
                    text = output.text_diff
                elif hasattr(output, "text"):
                    text = output.text
                else:
                    text = str(output)

                if not text and not first_chunk:
                    continue

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text} if not first_chunk else {"role": "assistant", "content": text},
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk)}\n\n"
                first_chunk = False

            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500,
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    async def initialize_services(self):
        """Initialize workers, HTTP client, and tokenizer."""
        logger.info("=" * 60)
        logger.info("STARTING INITIALIZATION - This may take 5-15 minutes on first run")
        logger.info("=" * 60)
        
        logger.info("Step 1/4: Initializing TrtllmWorkers...")
        logger.info(f"  Model: {self.init_params.model}")
        logger.info(f"  Workers: {self.init_params.num_workers}")
        logger.info(f"  Block size: {self.init_params.block_size}")
        logger.info("  (TensorRT-LLM will compile the model - please be patient)")
        
        self.workers = TrtllmWorkers(
            model=self.init_params.model,
            block_size=self.init_params.block_size,
            base_kv_events_port=self.init_params.base_kv_events_port,
            base_metrics_port=self.init_params.base_metrics_port,
            num_workers=self.init_params.num_workers,
        )

        logger.info("Step 2/4: Starting worker background tasks...")
        await self.workers.start_all()

        logger.info("Step 3/4: Initializing HTTP client...")
        self.http_client = httpx.AsyncClient()

        logger.info("Step 4/4: Initializing tokenizer...")
        self.tokenizer = tokenizer_factory(self.init_params.model)

        logger.info("Waiting 2 seconds for services to stabilize...")
        await asyncio.sleep(2)
        
        logger.info("=" * 60)
        logger.info("ALL SERVICES INITIALIZED SUCCESSFULLY!")
        logger.info("=" * 60)

    async def start(self):
        """Start the API server."""
        await self.initialize_services()

        logger.info(f"Starting API server on port {self.init_params.http_port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.init_params.http_port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self):
        """Proper shutdown handler."""
        logger.info("Shutting down API...")

        if self.http_client:
            await self.http_client.aclose()

        if self.workers:
            self.workers.shutdown_all()

        logger.info("API shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Router API Server")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--block-size", type=int, default=32, help="Block size for caching (TensorRT-LLM uses 32)"
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
            # Router is lightweight, start it first
            router_task = asyncio.create_task(router_api.start())
            await asyncio.sleep(0.5)  # Let router bind ports

            # API initialization is heavy (TensorRT-LLM), start after router
            api_task = asyncio.create_task(api.start())

            await asyncio.gather(router_task, api_task)
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

