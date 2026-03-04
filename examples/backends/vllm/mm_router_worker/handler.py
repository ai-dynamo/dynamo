# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

import logging
import time
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging

from .image_cache_server import ImageCacheServer
from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Skip data URI replacement for images larger than 4MB base64-encoded
MAX_DATA_URI_SIZE = 4 * 1024 * 1024


def _strip_image_content(extra_args: dict | None) -> dict | None:
    """Remove inline image data from extra_args['messages'] to shrink payload.

    The MM router already extracted and processed images; the downstream backend
    receives them via multi_modal_data.  Keeping the original base64 data URIs
    inside messages doubles the serialized size for no benefit.
    """
    if not extra_args:
        return extra_args
    messages = extra_args.get("messages")
    if not messages:
        return extra_args

    stripped = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(msg)
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                # Replace heavy image data with a lightweight placeholder
                new_content.append({"type": "image_url", "image_url": {"url": "stripped"}})
            else:
                new_content.append(part)
        stripped.append({**msg, "content": new_content})

    return {**extra_args, "messages": stripped}


class MMRouterHandler:
    """
    Handler that computes mm_hash for multimodal requests and routes
    to the best vLLM worker based on KV cache overlap.
    """

    def __init__(
        self,
        kv_router: KvRouter,
        tokenizer: Any,
        processor: Any,
        model: str,
        block_size: int,
        image_cache: ImageCacheServer | None = None,
    ):
        """
        Initialize the MM Router Handler.

        Args:
            kv_router: KvRouter for KV-aware worker selection and routing
            tokenizer: HuggingFace AutoTokenizer
            processor: HuggingFace AutoProcessor (optional)
            model: Model path/name
            block_size: KV cache block size
            image_cache: Optional image cache server for payload optimization
        """
        self.kv_router = kv_router
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.block_size = block_size
        self.image_loader = ImageLoader()
        self.image_cache = image_cache

    async def generate(self, request: dict) -> AsyncGenerator[dict, None]:
        """
        Main entry point - receives request, computes routing, forwards to best worker.

        The request format (after Frontend preprocessing with ModelInput.Tokens):
        {
            "token_ids": [...],
            "sampling_options": {...},
            "stop_conditions": {...},
            "extra_args": {"messages": [...]}
        }

        Args:
            request: Preprocessed request from Frontend

        Yields:
            Response chunks from the downstream vLLM worker
        """
        handler_start = time.perf_counter()

        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
            # Process multimodal: download images, compute mm_hash
            # Do not reuse request["token_ids"] for MM routing: those are placeholder-level
            # tokens from frontend. We need processor-expanded tokens to build block_mm_infos.
            # Request payload does not include a rendered template string; extra_args carries
            # original messages, so mm_processor reapplies chat template locally.
            mm_start = time.perf_counter()
            processed = await process_multimodal(
                messages=messages,
                image_urls=image_urls,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=self.model,
                image_loader=self.image_loader,
            )
            mm_ms = (time.perf_counter() - mm_start) * 1000

            # Build block_mm_infos for MM-aware hash computation
            block_start = time.perf_counter()
            block_mm_infos = build_block_mm_infos(
                num_tokens=len(processed.tokens),
                block_size=self.block_size,
                mm_hashes=processed.mm_hashes,
                image_ranges=processed.image_ranges,
            )
            block_mm_ms = (time.perf_counter() - block_start) * 1000
            if block_mm_infos is None:
                raise ValueError(
                    "Failed to build block_mm_infos for multimodal request"
                )

            routing_tokens = processed.tokens
            routing_blocks = (
                len(routing_tokens) + self.block_size - 1
            ) // self.block_size
            logger.debug(
                f"MM request: {len(routing_tokens)} routing tokens, "
                f"{len(image_urls)} images, {routing_blocks} routing blocks"
            )
        else:
            mm_ms = 0.0
            block_mm_ms = 0.0
            # Text-only: rely on frontend-preprocessed token_ids (ModelInput.Tokens contract)
            tokens = request.get("token_ids")
            if not tokens:
                raise ValueError(
                    "Missing or empty token_ids in preprocessed request for text-only routing"
                )

            routing_tokens = tokens
            routing_blocks = (
                len(routing_tokens) + self.block_size - 1
            ) // self.block_size
            logger.debug(
                f"Text request: {len(routing_tokens)} routing tokens, {routing_blocks} routing blocks"
            )
            # Text-only routing has no multimodal objects; provide per-block None entries.
            block_mm_infos = [None] * routing_blocks

        # Route and generate through KvRouter with explicit fields.
        # We pass:
        # - execution payload (token_ids + multi_modal_data)
        # - routing payload (mm_routing_info: routing_token_ids + block_mm_infos)
        # so generate() can select worker internally while preserving MM correctness.
        token_ids = request.get("token_ids")
        if not token_ids:
            raise ValueError("Missing or empty token_ids in preprocessed request")

        # Rewrite image URLs in multi_modal_data to reduce payload size.
        # If image_cache is available, replace data URIs with lightweight
        # HTTP URLs pointing to the local cache (~1MB → ~60 bytes per image).
        # Otherwise, fall back to the original data URI forwarding.
        rewrite_start = time.perf_counter()
        multi_modal_data = request.get("multi_modal_data")
        if image_urls and multi_modal_data:
            image_url_items = multi_modal_data.get("image_url", [])
            if self.image_cache and self.image_cache.port:
                # Cache raw bytes and replace with local HTTP URL
                for i, raw_bytes in enumerate(processed.raw_bytes_list):
                    if i < len(image_url_items):
                        item = image_url_items[i]
                        if isinstance(item, dict) and "Url" in item:
                            key = self.image_cache.put(raw_bytes)
                            item["Url"] = self.image_cache.url_for(key)
            elif processed.data_uris:
                # Fallback: replace HTTP URLs with data URIs
                for i, data_uri in enumerate(processed.data_uris):
                    if i < len(image_url_items):
                        item = image_url_items[i]
                        if isinstance(item, dict) and "Url" in item:
                            if urlparse(item["Url"]).scheme in ("http", "https"):
                                if len(data_uri) <= MAX_DATA_URI_SIZE:
                                    item["Url"] = data_uri
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000

        mm_routing_info: dict[str, Any] = {
            "routing_token_ids": routing_tokens,
            "block_mm_infos": block_mm_infos,
        }

        # Strip large image content from extra_args before forwarding.
        # The backend only needs extra_args for kv_transfer_params, not the
        # original messages with inline base64 images. Keeping them bloats
        # the serialized payload (e.g. 2MB for a single 512x512 image).
        forwarded_extra_args = _strip_image_content(request.get("extra_args"))

        pre_route_ms = (time.perf_counter() - handler_start) * 1000

        route_start = time.perf_counter()
        stream = await self.kv_router.generate(
            token_ids=token_ids,
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=forwarded_extra_args,
            multi_modal_data=multi_modal_data,
            mm_routing_info=mm_routing_info,
        )
        # Time spent before a stream object is returned from KvRouter.generate().
        # This may include worker selection and request-plane setup.
        kv_open_stream_ms = (time.perf_counter() - route_start) * 1000

        first_chunk = True
        async for response in stream:
            if first_chunk:
                first_chunk_ms = (time.perf_counter() - handler_start) * 1000
                # Time spent waiting for the first downstream chunk once the stream exists.
                kv_wait_first_chunk_ms = max(
                    0.0, first_chunk_ms - pre_route_ms - kv_open_stream_ms
                )
                kv_total_first_chunk_ms = kv_open_stream_ms + kv_wait_first_chunk_ms
                logger.debug(
                    "[perf][mm_handler] generate: "
                    "first_chunk=%.3fms pre_route=%.3fms "
                    "process_mm=%.3fms build_block=%.3fms rewrite_uri=%.3fms "
                    "kv_router_generate=%.3fms "
                    "kv_open_stream=%.3fms kv_wait_first_chunk=%.3fms "
                    "kv_to_first_chunk=%.3fms "
                    "images=%d tokens=%d",
                    first_chunk_ms,
                    pre_route_ms,
                    mm_ms,
                    block_mm_ms,
                    rewrite_ms,
                    kv_open_stream_ms,
                    kv_open_stream_ms,
                    kv_wait_first_chunk_ms,
                    kv_total_first_chunk_ms,
                    len(image_urls),
                    len(token_ids),
                )
                first_chunk = False
            yield response
