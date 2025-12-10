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
from PIL import Image
from pydantic import BaseModel
from router import RouterAPI, RouterRequest, RouterResponse
from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.inputs.utils import default_multimodal_input_loader, load_image
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from transformers import AutoProcessor
from worker import TrtllmWorkers

from dynamo._core import compute_block_hash_for_seq_py

logger = logging.getLogger(__name__)

# Debug flag: set DYNAMO_DEBUG=1 to enable debug file dumps
DEBUG_ENABLED = os.environ.get("DYNAMO_DEBUG", "0") == "1"
DEBUG_API_FILE = "/tmp/debug_api_hashes.txt"


def dump_api_debug(
    tokens: list[int],
    block_size: int,
    local_hashes: list[int],
    mm_hash: int | None,
    block_mm_infos: list | None,
    image_urls: list[str] | None,
):
    """Dump API-side hash computation to file for debugging."""
    if not DEBUG_ENABLED:
        return
    import datetime

    with open(DEBUG_API_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Image URLs: {image_urls}\n")
        f.write(f"mm_hash: {mm_hash}\n")
        f.write(f"block_size: {block_size}\n")
        f.write(f"num_tokens: {len(tokens)}\n")
        f.write(f"tokens (first 50): {tokens[:50]}\n")
        f.write(f"tokens (last 50): {tokens[-50:]}\n")
        f.write(f"block_mm_infos: {block_mm_infos}\n")
        f.write(f"local_hashes ({len(local_hashes)}): {local_hashes}\n")
        f.write(f"{'='*60}\n")

# Multimodal content types (OpenAI format)
class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]  # str for text-only, list for multimodal


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
    model_type: str  # e.g., "qwen2_vl", "llava"
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
        self.processor = None  # HuggingFace processor for token expansion
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

                # Extract text and multimodal content from messages
                messages_dict = []
                image_urls = []

                for msg in request.messages:
                    if isinstance(msg.content, str):
                        # Text-only message
                        messages_dict.append({"role": msg.role, "content": msg.content})
                    else:
                        # Multimodal message
                        text_parts = []
                        for part in msg.content:
                            if part.type == "text" and part.text:
                                text_parts.append(part.text)
                            elif part.type == "image_url" and part.image_url:
                                image_urls.append(part.image_url.url)
                        messages_dict.append({"role": msg.role, "content": " ".join(text_parts)})

                # Build prompt text
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages_dict, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply chat template: {e}, falling back to simple formatting"
                    )
                    prompt = self._format_messages_simple(messages_dict)

                # Process multimodal or text-only
                block_mm_infos = None
                mm_hash = None
                mm_input = None  # The input to pass to TRTLLM generate_async
                image_offsets = None  # [start, end] of image tokens in the sequence

                if image_urls:
                    # Use default_multimodal_input_loader for multimodal requests
                    try:
                        # 1. Get processed input for TRTLLM generation
                        inputs = default_multimodal_input_loader(
                            tokenizer=self.tokenizer,
                            model_dir=self.init_params.model,
                            model_type=self.init_params.model_type,
                            modality="image",
                            prompts=[prompt],
                            media=[image_urls],
                            image_data_format="pt",
                            device="cuda",
                        )
                        mm_input = inputs[0]  # Dict to pass to generate_async
                        processed_prompt = mm_input.get("prompt", prompt)
                        multi_modal_data = mm_input.get("multi_modal_data")

                        logger.info(f"API: Processed MM input for TRTLLM, prompt length: {len(processed_prompt)}")

                        # 2. Use HF processor to get expanded tokens for routing hash
                        if self.processor is not None:
                            pil_images = [load_image(url, format="pil") for url in image_urls]

                            processor_output = self.processor(
                                text=[processed_prompt],
                                images=pil_images,
                                return_tensors="pt",
                                padding=True,
                            )
                            tokens = processor_output["input_ids"][0].tolist()
                            logger.info(f"API: Processor returned {len(tokens)} tokens (with visual token expansion)")

                            # 3. Replace image_token with vocab_size + 1 (TRTLLM convention)
                            #    and find the image token positions (offsets)
                            image_token_id = getattr(self.processor, "image_token_id", None)
                            if image_token_id is None:
                                # Fallback for Qwen2-VL: image_token_id = 151655
                                image_token_id = 151655
                            replacement_id = 151937

                            # Find image token positions and replace
                            image_start = None
                            image_end = None
                            num_replaced = 0
                            for i, t in enumerate(tokens):
                                if t == image_token_id:
                                    if image_start is None:
                                        image_start = i
                                    image_end = i + 1  # exclusive end
                                    tokens[i] = replacement_id
                                    num_replaced += 1

                            # Store image offsets for mm_hash routing
                            image_offsets = None
                            if image_start is not None and image_end is not None:
                                image_offsets = [image_start, image_end]
                                logger.info(
                                    f"API: Image tokens at positions [{image_start}, {image_end}), "
                                    f"replaced {num_replaced} tokens"
                                )
                            else:
                                logger.info(f"API: Replaced {num_replaced} image tokens")
                        else:
                            tokens = self.tokenizer.encode(processed_prompt)
                            logger.warning(f"API: No processor, using tokenizer only: {len(tokens)} tokens")

                        # 4. Compute mm_hash from multi_modal_data
                        if multi_modal_data:
                            mm_hashes_dict = apply_mm_hashes(multi_modal_data)
                            if "image" in mm_hashes_dict and mm_hashes_dict["image"]:
                                first_hex = mm_hashes_dict["image"][0][:16]
                                mm_hash = int(first_hex, 16)
                                logger.info(f"API: Computed mm_hash={mm_hash}")

                    except Exception as e:
                        logger.warning(f"Failed to process MM input: {e}, falling back to text-only")
                        tokens = self.tokenizer.encode(prompt)
                        mm_input = None
                else:
                    # Text-only request
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

                # Build block_mm_infos if we have mm_hash
                # Use actual image token positions (image_offsets) not [0, num_tokens]
                if mm_hash is not None and image_offsets is not None:
                    num_blocks = (num_tokens + self.init_params.block_size - 1) // self.init_params.block_size
                    block_mm_infos = [
                        {
                            "mm_objects": [{
                                "mm_hash": mm_hash,
                                "offsets": [image_offsets]  # [[start, end]] - request level mask
                            }]
                        }
                        for _ in range(num_blocks)
                    ]
                    logger.info(f"API: block_mm_infos with offsets {image_offsets}")

                # Compute block hashes (with MM info if available)
                local_hashes = compute_block_hash_for_seq_py(
                    tokens, self.init_params.block_size, block_mm_infos
                )
                logger.info(f"API: Computed {len(local_hashes)} local_hashes: {local_hashes}")

                # Dump debug info to file
                dump_api_debug(
                    tokens=tokens,
                    block_size=self.init_params.block_size,
                    local_hashes=local_hashes,
                    mm_hash=mm_hash,
                    block_mm_infos=block_mm_infos,
                    image_urls=image_urls,
                )

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

                # For MM requests, pass mm_input dict; for text, pass tokens
                prompt_input = mm_input if mm_input else tokens
                result_generator = self.workers.direct(
                    prompt_input, best_worker_id, sampling_params
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

        logger.info("Step 4/5: Initializing tokenizer...")
        self.tokenizer = tokenizer_factory(self.init_params.model)

        logger.info("Step 5/5: Initializing HuggingFace processor for MM token expansion...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.init_params.model, trust_remote_code=True
            )
            logger.info("Processor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize processor: {e}")
            self.processor = None

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
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model name to use (VLM for multimodal support)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen2_vl",
        help="Model type for TRTLLM (e.g., qwen2_vl, llava, phi3_v)",
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
        model_type=args.model_type,
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

