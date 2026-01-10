# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from fastapi.responses import JSONResponse, StreamingResponse
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

# Qwen2-VL specific token IDs
QWEN2_VL_IMAGE_TOKEN_ID = 151655
QWEN2_VL_REPLACEMENT_ID = 151937


def dump_api_debug(
    tokens: list[int],
    block_size: int,
    local_hashes: list[int],
    mm_hashes: list[int] | None,
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
        f.write(f"mm_hashes: {mm_hashes}\n")
        f.write(f"block_size: {block_size}\n")
        f.write(f"num_tokens: {len(tokens)}\n")
        f.write(f"tokens (first 50): {tokens[:50]}\n")
        f.write(f"tokens (last 50): {tokens[-50:]}\n")
        f.write(f"block_mm_infos: {block_mm_infos}\n")
        f.write(f"local_hashes ({len(local_hashes)}): {local_hashes}\n")
        f.write(f"{'='*60}\n")


def make_error(message: str, error_type: str, code: int) -> dict:
    """Create a standardized error response dict."""
    return {"message": message, "type": error_type, "code": code}


# Pydantic models for OpenAI-compatible API
class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


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
    """Configuration parameters for the serving API."""

    model: str
    model_type: str  # e.g., "qwen2_vl", "llava"
    block_size: int
    num_workers: int
    base_kv_events_port: int
    base_metrics_port: int
    router_port: int
    http_port: int


@dataclass
class ParsedRequest:
    """Parsed and preprocessed request data."""

    messages_dict: list[dict]
    image_urls: list[str]
    max_tokens: int
    temperature: float
    top_p: float
    model: str


@dataclass
class ProcessedInput:
    """Processed input ready for routing and generation."""

    tokens: list[int]
    mm_input: dict | None  # For multimodal requests
    mm_hashes: list[int] | None  # List of mm_hash for each image
    image_offsets_list: list[list[int]] | None  # List of [start, end] for each image


class ServiceAPI:
    """Main API service handling chat completion requests with KV cache routing."""

    def __init__(self, init_params: ServingParams):
        self.init_params = init_params
        self.app = FastAPI(title="TensorRT-LLM Router API", version="0.0.1")

        self.workers: Optional[TrtllmWorkers] = None
        self.tokenizer = None
        self.processor = None
        self.http_client: Optional[httpx.AsyncClient] = None

        self._setup_routes()

    # -------------------------------------------------------------------------
    # Request Parsing Helpers
    # -------------------------------------------------------------------------

    def _parse_request(
        self, request: ChatCompletionRequest
    ) -> ParsedRequest | ErrorResponse:
        """Parse and validate the incoming request."""
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens is None:
            return ErrorResponse(
                error=make_error(
                    "Either max_tokens or max_completion_tokens must be specified",
                    "invalid_request_error",
                    400,
                )
            )

        messages_dict, image_urls = self._extract_messages(request.messages)

        return ParsedRequest(
            messages_dict=messages_dict,
            image_urls=image_urls,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            model=request.model,
        )

    def _extract_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict], list[str]]:
        """Extract text messages and image URLs from request messages."""
        messages_dict = []
        image_urls = []

        for msg in messages:
            if isinstance(msg.content, str):
                messages_dict.append({"role": msg.role, "content": msg.content})
            else:
                text_parts = []
                for part in msg.content:
                    if part.type == "text" and part.text:
                        text_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        image_urls.append(part.image_url.url)
                messages_dict.append(
                    {"role": msg.role, "content": " ".join(text_parts)}
                )

        return messages_dict, image_urls

    def _build_prompt(self, messages_dict: list[dict]) -> str:
        """Build prompt text from messages using chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using simple format")
            return self._format_messages_simple(messages_dict)

    def _format_messages_simple(self, messages: list[dict]) -> str:
        """Simple fallback formatting when chat template is unavailable."""
        parts = []
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        for msg in messages:
            prefix = role_map.get(msg["role"], msg["role"].capitalize())
            parts.append(f"{prefix}: {msg['content']}\n")
        parts.append("Assistant: ")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Multimodal Processing Helpers
    # -------------------------------------------------------------------------

    def _process_multimodal(self, prompt: str, image_urls: list[str]) -> ProcessedInput:
        """Process multimodal request: load images, compute tokens and mm_hashes."""
        try:
            # Use "multiple_image" modality when there are multiple images
            modality = "multiple_image" if len(image_urls) > 1 else "image"
            inputs = default_multimodal_input_loader(
                tokenizer=self.tokenizer,
                model_dir=self.init_params.model,
                model_type=self.init_params.model_type,
                modality=modality,
                prompts=[prompt],
                media=[image_urls],
                image_data_format="pt",
                device="cuda",
            )
            mm_input = inputs[0]
            processed_prompt = mm_input.get("prompt", prompt)
            multi_modal_data = mm_input.get("multi_modal_data")

            tokens, image_offsets_list = self._get_mm_tokens(
                processed_prompt, image_urls
            )
            mm_hashes = self._compute_mm_hashes(multi_modal_data)

            return ProcessedInput(
                tokens=tokens,
                mm_input=mm_input,
                mm_hashes=mm_hashes,
                image_offsets_list=image_offsets_list,
            )
        except Exception as e:
            logger.warning(f"MM processing failed: {e}, falling back to text-only")
            return ProcessedInput(
                tokens=self.tokenizer.encode(prompt),
                mm_input=None,
                mm_hashes=None,
                image_offsets_list=None,
            )

    def _get_mm_tokens(
        self, prompt: str, image_urls: list[str]
    ) -> tuple[list[int], list[list[int]] | None]:
        """Get tokens with visual expansion and find image token positions."""
        if self.processor is None:
            return self.tokenizer.encode(prompt), None

        pil_images = [load_image(url, format="pil") for url in image_urls]
        processor_output = self.processor(
            text=[prompt], images=pil_images, return_tensors="pt", padding=True
        )
        tokens = processor_output["input_ids"][0].tolist()

        image_token_id = getattr(
            self.processor, "image_token_id", QWEN2_VL_IMAGE_TOKEN_ID
        )
        return self._replace_image_tokens(
            tokens, image_token_id, QWEN2_VL_REPLACEMENT_ID
        )

    def _replace_image_tokens(
        self, tokens: list[int], image_token_id: int, replacement_id: int
    ) -> tuple[list[int], list[list[int]] | None]:
        """Replace image tokens and return their positions as list of [start, end] per image.

        Finds contiguous regions of image tokens. Each contiguous region is assumed
        to be one image.
        """
        image_offsets_list: list[list[int]] = []
        current_start: int | None = None

        for i, t in enumerate(tokens):
            if t == image_token_id:
                if current_start is None:
                    current_start = i
                tokens[i] = replacement_id
            else:
                # End of a contiguous image token region
                if current_start is not None:
                    image_offsets_list.append([current_start, i])
                    current_start = None

        # Handle case where image tokens go to the end
        if current_start is not None:
            image_offsets_list.append([current_start, len(tokens)])

        if image_offsets_list:
            logger.debug(f"Image token regions: {image_offsets_list}")
            return tokens, image_offsets_list
        return tokens, None

    def _compute_mm_hashes(self, multi_modal_data: dict | None) -> list[int] | None:
        """Compute mm_hash for each image in multimodal data.

        Returns:
            List of mm_hash (one per image), or None if no images.
        """
        if not multi_modal_data:
            return None

        mm_hashes_dict = apply_mm_hashes(multi_modal_data)
        if "image" in mm_hashes_dict and mm_hashes_dict["image"]:
            # Convert each 256-bit hex digest to 64-bit int
            mm_hashes = [
                int(hex_digest[:16], 16) for hex_digest in mm_hashes_dict["image"]
            ]
            logger.debug(f"Computed mm_hashes for {len(mm_hashes)} images: {mm_hashes}")
            return mm_hashes
        return None

    # -------------------------------------------------------------------------
    # Routing Helpers
    # -------------------------------------------------------------------------

    def _build_block_mm_infos(
        self,
        num_tokens: int,
        mm_hashes: list[int] | None,
        image_offsets_list: list[list[int]] | None,
    ) -> list[dict | None] | None:
        """Build block_mm_infos for routing hash computation.

        For each block, includes mm_objects for all images that overlap with that block.

        Args:
            num_tokens: Total number of tokens
            mm_hashes: List of mm_hash, one per image
            image_offsets_list: List of [start, end] offsets, one per image

        Returns:
            List of mm_info dicts (one per block), with None for blocks without images.
        """
        if mm_hashes is None or image_offsets_list is None:
            return None

        if len(mm_hashes) != len(image_offsets_list):
            logger.warning(
                f"mm_hashes ({len(mm_hashes)}) and image_offsets_list "
                f"({len(image_offsets_list)}) length mismatch"
            )
            return None

        block_size = self.init_params.block_size
        num_blocks = (num_tokens + block_size - 1) // block_size

        result: list[dict | None] = []
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            # Find all images that overlap with this block
            mm_objects = []
            for mm_hash, offsets in zip(mm_hashes, image_offsets_list):
                img_start, img_end = offsets
                if block_end > img_start and block_start < img_end:
                    mm_objects.append({"mm_hash": mm_hash, "offsets": [offsets]})

            if mm_objects:
                result.append({"mm_objects": mm_objects})
            else:
                result.append(None)

        return result

    async def _route_request(
        self, local_hashes: list[int], num_tokens: int
    ) -> int | ErrorResponse:
        """Query router for best worker ID."""
        try:
            router_request = RouterRequest(
                local_hashes=local_hashes, num_tokens=num_tokens
            )
            response = await self.http_client.post(
                f"http://localhost:{self.init_params.router_port}/find_best_worker",
                json=router_request.model_dump(),
                timeout=1,
            )
            response.raise_for_status()
            return RouterResponse.model_validate(response.json()).worker_id
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Router request failed: {e}")
            return ErrorResponse(
                error=make_error(
                    "Router service unavailable", "service_unavailable", 503
                )
            )

    # -------------------------------------------------------------------------
    # Response Streaming
    # -------------------------------------------------------------------------

    async def _stream_response(
        self, request: ChatCompletionRequest, result_generator, request_id: str
    ):
        """Generate SSE formatted streaming responses."""
        created = int(time.time())
        first_chunk = True
        try:
            async for output in result_generator:
                # Handle both dict (from worker) and object responses
                if isinstance(output, dict):
                    text = output.get("text_diff") or output.get("text", "")
                else:
                    text = getattr(output, "text_diff", None) or getattr(
                        output, "text", ""
                    )

                if not text and not first_chunk:
                    continue

                delta = (
                    {"role": "assistant", "content": text}
                    if first_chunk
                    else {"content": text}
                )
                yield self._format_chunk(
                    request_id, created, request.model, delta, None
                )
                first_chunk = False

            # Final chunk
            yield self._format_chunk(request_id, created, request.model, {}, "stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': make_error(str(e), 'internal_error', 500)})}\n\n"

    def _format_chunk(
        self,
        request_id: str,
        created: int,
        model: str,
        delta: dict,
        finish_reason: str | None,
    ) -> str:
        """Format a single SSE chunk."""
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def _generate_full_response(
        self, request: ChatCompletionRequest, result_generator, request_id: str
    ) -> dict:
        """Collect all outputs and generate a complete (non-streaming) response."""
        created = int(time.time())
        full_text = ""

        try:
            async for output in result_generator:
                if isinstance(output, dict):
                    text = output.get("text_diff") or output.get("text", "")
                else:
                    text = getattr(output, "text_diff", None) or getattr(
                        output, "text", ""
                    )
                full_text += text

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Not tracked in this implementation
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"error": make_error(str(e), "internal_error", 500)}

    # -------------------------------------------------------------------------
    # Main Request Handler
    # -------------------------------------------------------------------------

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            # Check service readiness
            if (
                self.workers is None
                or self.tokenizer is None
                or self.http_client is None
            ):
                return ErrorResponse(
                    error=make_error("Service not ready", "service_unavailable", 503)
                )

            try:
                # Parse request
                parsed = self._parse_request(request)
                if isinstance(parsed, ErrorResponse):
                    return parsed

                # Process input (multimodal or text-only)
                if parsed.image_urls:
                    # For multimodal: pass raw text, let default_multimodal_input_loader apply chat template
                    raw_text = " ".join(
                        msg["content"]
                        for msg in parsed.messages_dict
                        if msg.get("content")
                    )
                    processed = self._process_multimodal(raw_text, parsed.image_urls)
                else:
                    # For text-only: apply chat template ourselves
                    prompt = self._build_prompt(parsed.messages_dict)
                    processed = ProcessedInput(
                        tokens=self.tokenizer.encode(prompt),
                        mm_input=None,
                        mm_hashes=None,
                        image_offsets_list=None,
                    )

                # Validate tokens
                if not processed.tokens:
                    return ErrorResponse(
                        error=make_error(
                            "Input prompt is empty", "invalid_request_error", 400
                        )
                    )

                # Compute block hashes for routing
                block_mm_infos = self._build_block_mm_infos(
                    len(processed.tokens),
                    processed.mm_hashes,
                    processed.image_offsets_list,
                )
                logger.debug(f"block_mm_infos: {block_mm_infos}")
                local_hashes = compute_block_hash_for_seq_py(
                    processed.tokens, self.init_params.block_size, block_mm_infos
                )

                # Debug dump
                dump_api_debug(
                    tokens=processed.tokens,
                    block_size=self.init_params.block_size,
                    local_hashes=local_hashes,
                    mm_hashes=processed.mm_hashes,
                    block_mm_infos=block_mm_infos,
                    image_urls=parsed.image_urls,
                )

                # Route to best worker
                worker_id = await self._route_request(
                    local_hashes, len(processed.tokens)
                )
                if isinstance(worker_id, ErrorResponse):
                    return worker_id

                # Generate response
                request_id = f"chatcmpl-{uuid.uuid4()}"
                sampling_params = {
                    "max_tokens": parsed.max_tokens,
                    "temperature": parsed.temperature,
                    "top_p": parsed.top_p,
                }
                prompt_input = processed.mm_input or processed.tokens
                logger.debug(f"Sending to worker {worker_id}")
                result_generator = self.workers.direct(
                    prompt_input, worker_id, sampling_params
                )

                if request.stream:
                    return StreamingResponse(
                        self._stream_response(request, result_generator, request_id),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    # Non-streaming: collect all outputs and return complete response
                    response_data = await self._generate_full_response(
                        request, result_generator, request_id
                    )
                    return JSONResponse(content=response_data)

            except Exception as e:
                logger.error(f"Request processing error: {e}")
                return ErrorResponse(error=make_error(str(e), "internal_error", 500))

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def initialize_services(self):
        """Initialize workers, HTTP client, and tokenizer."""
        logger.info(
            f"Initializing services: model={self.init_params.model}, "
            f"workers={self.init_params.num_workers}, block_size={self.init_params.block_size}"
        )

        self.workers = TrtllmWorkers(
            model=self.init_params.model,
            block_size=self.init_params.block_size,
            base_kv_events_port=self.init_params.base_kv_events_port,
            base_metrics_port=self.init_params.base_metrics_port,
            num_workers=self.init_params.num_workers,
        )
        await self.workers.start_all()

        self.http_client = httpx.AsyncClient()
        self.tokenizer = tokenizer_factory(self.init_params.model)

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.init_params.model, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize HF processor: {e}")
            self.processor = None

        await asyncio.sleep(2)
        logger.info("All services initialized")

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
        "--block-size",
        type=int,
        default=32,
        help="Block size for caching (TensorRT-LLM uses 32)",
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
        "--router-port", type=int, default=7000, help="Port for router service"
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
            router_task = asyncio.create_task(router_api.start())
            await asyncio.sleep(0.5)
            api_task = asyncio.create_task(api.start())
            await asyncio.gather(router_task, api_task)
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
        finally:
            await api.shutdown()

    try:
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        logger.info("Force shutdown via KeyboardInterrupt.")


if __name__ == "__main__":
    main()
