# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom API Frontend for Multimodal KV Router Example.

This example demonstrates how to use Dynamo's standard TRT-LLM workers
with a custom frontend that computes mm_hash for MM-aware KV routing.

Architecture:
- Custom api.py: HTTP server + mm_hash computation + routing logic
- Standard workers: python -m dynamo.trtllm (no custom worker code)
- Dynamo Client.direct(): Route to specific worker based on KV cache overlap
- KvIndexer: Subscribe to KV events from workers via NATS
"""

import os

# Fix protobuf version conflict
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.inputs.utils import default_multimodal_input_loader, load_image
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from transformers import AutoProcessor

from dynamo._core import KvIndexer, RadixTree, compute_block_hash_for_seq_py
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

# Qwen2-VL specific token IDs
QWEN2_VL_IMAGE_TOKEN_ID = 151655
QWEN2_VL_REPLACEMENT_ID = 151937


def make_error(message: str, error_type: str, code: int) -> dict:
    """Create a standardized error response dict."""
    return {"message": message, "type": error_type, "code": code}


# Pydantic models for OpenAI-compatible API
class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
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


@dataclass
class ProcessedInput:
    """Processed input ready for routing and generation."""

    tokens: list[int]
    mm_input: dict | None
    mm_hashes: list[int] | None
    image_offsets_list: list[list[int]] | None


# Global state (set by dynamo_worker)
app_state = {
    "client": None,
    "indexer": None,
    "instance_ids": [],
    "tokenizer": None,
    "processor": None,
    "block_size": 32,
    "model": None,
    "model_type": None,
}

app = FastAPI(title="Dynamo TRT-LLM Multimodal KV Router API", version="0.1.0")


# =============================================================================
# Multimodal Processing Helpers (reused from standalone)
# =============================================================================


def process_multimodal(
    prompt: str, image_urls: list[str], model: str, model_type: str, tokenizer
) -> ProcessedInput:
    """Process multimodal request: load images, compute tokens and mm_hashes."""
    try:
        modality = "multiple_image" if len(image_urls) > 1 else "image"
        inputs = default_multimodal_input_loader(
            tokenizer=tokenizer,
            model_dir=model,
            model_type=model_type,
            modality=modality,
            prompts=[prompt],
            media=[image_urls],
            image_data_format="pt",
            device="cuda",
        )
        mm_input = inputs[0]
        processed_prompt = mm_input.get("prompt", prompt)
        multi_modal_data = mm_input.get("multi_modal_data")

        processor = app_state.get("processor")
        tokens, image_offsets_list = get_mm_tokens(
            processed_prompt, image_urls, tokenizer, processor
        )
        mm_hashes = compute_mm_hashes(multi_modal_data)

        return ProcessedInput(
            tokens=tokens,
            mm_input=mm_input,
            mm_hashes=mm_hashes,
            image_offsets_list=image_offsets_list,
        )
    except Exception as e:
        logger.warning(f"MM processing failed: {e}, falling back to text-only")
        return ProcessedInput(
            tokens=tokenizer.encode(prompt),
            mm_input=None,
            mm_hashes=None,
            image_offsets_list=None,
        )


def get_mm_tokens(
    prompt: str, image_urls: list[str], tokenizer, processor
) -> tuple[list[int], list[list[int]] | None]:
    """Get tokens with visual expansion and find image token positions."""
    if processor is None:
        return tokenizer.encode(prompt), None

    pil_images = [load_image(url, format="pil") for url in image_urls]
    processor_output = processor(
        text=[prompt], images=pil_images, return_tensors="pt", padding=True
    )
    tokens = processor_output["input_ids"][0].tolist()

    image_token_id = getattr(processor, "image_token_id", QWEN2_VL_IMAGE_TOKEN_ID)
    return replace_image_tokens(tokens, image_token_id, QWEN2_VL_REPLACEMENT_ID)


def replace_image_tokens(
    tokens: list[int], image_token_id: int, replacement_id: int
) -> tuple[list[int], list[list[int]] | None]:
    """Replace image tokens and return their positions."""
    image_offsets_list: list[list[int]] = []
    current_start: int | None = None

    for i, t in enumerate(tokens):
        if t == image_token_id:
            if current_start is None:
                current_start = i
            tokens[i] = replacement_id
        else:
            if current_start is not None:
                image_offsets_list.append([current_start, i])
                current_start = None

    if current_start is not None:
        image_offsets_list.append([current_start, len(tokens)])

    return tokens, image_offsets_list if image_offsets_list else None


def compute_mm_hashes(multi_modal_data: dict | None) -> list[int] | None:
    """Compute mm_hash for each image in multimodal data."""
    if not multi_modal_data:
        return None

    mm_hashes_dict = apply_mm_hashes(multi_modal_data)
    if "image" in mm_hashes_dict and mm_hashes_dict["image"]:
        mm_hashes = [int(hex_digest[:16], 16) for hex_digest in mm_hashes_dict["image"]]
        logger.debug(f"Computed mm_hashes for {len(mm_hashes)} images: {mm_hashes}")
        return mm_hashes
    return None


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_offsets_list: list[list[int]] | None,
) -> list[dict | None] | None:
    """Build block_mm_infos for routing hash computation."""
    if mm_hashes is None or image_offsets_list is None:
        return None

    if len(mm_hashes) != len(image_offsets_list):
        logger.warning(
            f"mm_hashes ({len(mm_hashes)}) and image_offsets_list "
            f"({len(image_offsets_list)}) length mismatch"
        )
        return None

    num_blocks = (num_tokens + block_size - 1) // block_size

    result: list[dict | None] = []
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        mm_objects = []
        for mm_hash, offsets in zip(mm_hashes, image_offsets_list):
            img_start, img_end = offsets
            if block_end > img_start and block_start < img_end:
                mm_objects.append({"mm_hash": mm_hash, "offsets": [offsets]})

        result.append({"mm_objects": mm_objects} if mm_objects else None)

    return result


# =============================================================================
# Routing Logic
# =============================================================================


async def find_best_worker(
    local_hashes: list[int], instance_ids: list[int], indexer
) -> int:
    """Find the best worker based on KV cache overlap."""
    if not instance_ids:
        raise ValueError("No workers available")

    if indexer is None:
        # Fallback to round-robin if no indexer
        return instance_ids[0]

    try:
        # Query indexer for overlap scores
        overlap_scores = await indexer.find_matches(local_hashes)

        # Find worker with highest overlap
        best_worker_id = instance_ids[0]
        best_score = 0

        for worker_id in instance_ids:
            score = overlap_scores.get(worker_id, 0)
            if score > best_score:
                best_score = score
                best_worker_id = worker_id

        logger.debug(
            f"Routing decision: worker={best_worker_id}, "
            f"overlap={best_score}, scores={overlap_scores}"
        )
        return best_worker_id

    except Exception as e:
        logger.warning(f"Indexer query failed: {e}, falling back to first worker")
        return instance_ids[0]


# =============================================================================
# Response Streaming
# =============================================================================


async def stream_response(
    request: ChatCompletionRequest,
    result_generator: AsyncGenerator,
    request_id: str,
):
    """Generate SSE formatted streaming responses."""
    created = int(time.time())
    first_chunk = True

    try:
        async for output in result_generator:
            if isinstance(output, dict):
                text = output.get("text_diff") or output.get("text", "")
            else:
                text = output.data().get("text_diff", "") if hasattr(output, "data") else ""

            if not text and not first_chunk:
                continue

            delta = (
                {"role": "assistant", "content": text}
                if first_chunk
                else {"content": text}
            )
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            first_chunk = False

        # Final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': make_error(str(e), 'internal_error', 500)})}\n\n"


async def generate_full_response(
    request: ChatCompletionRequest,
    result_generator: AsyncGenerator,
    request_id: str,
) -> dict:
    """Collect all outputs and generate a complete response."""
    created = int(time.time())
    full_text = ""

    try:
        async for output in result_generator:
            if isinstance(output, dict):
                text = output.get("text_diff") or output.get("text", "")
            else:
                text = output.data().get("text_diff", "") if hasattr(output, "data") else ""
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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"error": make_error(str(e), "internal_error", 500)}


# =============================================================================
# API Routes
# =============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests with MM-aware KV routing."""
    client = app_state.get("client")
    indexer = app_state.get("indexer")
    tokenizer = app_state.get("tokenizer")
    instance_ids = app_state.get("instance_ids", [])
    block_size = app_state.get("block_size", 32)
    model = app_state.get("model")
    model_type = app_state.get("model_type")

    if client is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": make_error("Service not ready", "service_unavailable", 503)},
        )

    try:
        # Parse request
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": make_error(
                        "max_tokens or max_completion_tokens required",
                        "invalid_request_error",
                        400,
                    )
                },
            )

        # Extract messages and images
        messages_dict = []
        image_urls = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages_dict.append({"role": msg.role, "content": msg.content})
            else:
                text_parts = []
                for part in msg.content:
                    if part.type == "text" and part.text:
                        text_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        image_urls.append(part.image_url.url)
                messages_dict.append({"role": msg.role, "content": " ".join(text_parts)})

        # Process input
        if image_urls:
            raw_text = " ".join(
                msg["content"] for msg in messages_dict if msg.get("content")
            )
            processed = process_multimodal(raw_text, image_urls, model, model_type, tokenizer)
        else:
            prompt = tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
            processed = ProcessedInput(
                tokens=tokenizer.encode(prompt),
                mm_input=None,
                mm_hashes=None,
                image_offsets_list=None,
            )

        if not processed.tokens:
            return JSONResponse(
                status_code=400,
                content={
                    "error": make_error("Empty input", "invalid_request_error", 400)
                },
            )

        # Compute block hashes WITH mm_info for MM-aware routing
        block_mm_infos = build_block_mm_infos(
            len(processed.tokens),
            block_size,
            processed.mm_hashes,
            processed.image_offsets_list,
        )
        local_hashes = compute_block_hash_for_seq_py(
            processed.tokens, block_size, block_mm_infos
        )

        logger.debug(
            f"Computed {len(local_hashes)} block hashes, "
            f"mm_hashes={processed.mm_hashes}"
        )

        # Find best worker based on KV cache overlap
        best_worker_id = await find_best_worker(local_hashes, instance_ids, indexer)

        # Build request for worker
        request_dict = {
            "prompt_token_ids": processed.tokens,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if processed.mm_input:
            request_dict["multi_modal_data"] = processed.mm_input.get("multi_modal_data")

        logger.info(f"Routing to worker {best_worker_id}")

        # Route to specific worker using Client.direct()
        request_id = f"chatcmpl-{uuid.uuid4()}"
        result_generator = client.direct(request_dict, str(best_worker_id))

        if request.stream:
            return StreamingResponse(
                stream_response(request, result_generator, request_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response_data = await generate_full_response(
                request, result_generator, request_id
            )
            return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception(f"Request error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": make_error(str(e), "internal_error", 500)},
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "workers": len(app_state.get("instance_ids", []))}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    model = app_state.get("model", "unknown")
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": "dynamo"}],
    }


# =============================================================================
# Main Entry Point
# =============================================================================


@dynamo_worker()
async def main(runtime: DistributedRuntime):
    """Main worker function that connects to Dynamo runtime."""
    global app_state

    parser = argparse.ArgumentParser(description="Dynamo TRT-LLM Multimodal KV Router API")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model-type", type=str, default="qwen2_vl")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--component", type=str, default="trtllm")
    parser.add_argument("--endpoint", type=str, default="generate")
    parser.add_argument("--http-port", type=int, default=8000)
    args, _ = parser.parse_known_args()

    logger.info(f"Connecting to Dynamo runtime...")

    # Connect to workers' endpoint
    endpoint = (
        runtime.namespace(args.namespace)
        .component(args.component)
        .endpoint(args.endpoint)
    )
    client = await endpoint.client()

    # Wait for workers to be available
    logger.info("Waiting for workers...")
    instance_ids = await client.wait_for_instances()
    logger.info(f"Found {len(instance_ids)} workers: {instance_ids}")

    # Create KV indexer to track worker cache states
    try:
        component = runtime.namespace(args.namespace).component(args.component)
        indexer = await KvIndexer.create(
            component=component,
            block_size=args.block_size,
        )
        logger.info("KvIndexer created successfully")
    except Exception as e:
        logger.warning(f"Failed to create KvIndexer: {e}, routing will be random")
        indexer = None

    # Initialize tokenizer and processor
    tokenizer = tokenizer_factory(args.model)
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load processor: {e}")
        processor = None

    # Store in global state
    app_state.update({
        "client": client,
        "indexer": indexer,
        "instance_ids": list(instance_ids),
        "tokenizer": tokenizer,
        "processor": processor,
        "block_size": args.block_size,
        "model": args.model,
        "model_type": args.model_type,
    })

    # Start FastAPI server
    logger.info(f"Starting API server on port {args.http_port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=args.http_port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvloop

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvloop.install()
    asyncio.run(main())
