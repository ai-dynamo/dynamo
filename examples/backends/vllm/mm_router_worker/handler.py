# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

import base64
import json
import logging
import time
from typing import Any, AsyncGenerator

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging

from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _detect_mime_type(raw_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


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
    ):
        self.kv_router = kv_router
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.block_size = block_size
        self.image_loader = ImageLoader()

    async def generate(self, request: dict) -> AsyncGenerator[dict, None]:
        handler_start = time.perf_counter()

        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
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
            block_mm_infos = [None] * routing_blocks

        token_ids = request.get("token_ids")
        if not token_ids:
            raise ValueError("Missing or empty token_ids in preprocessed request")

        mm_routing_info: dict[str, Any] = {
            "routing_token_ids": routing_tokens,
            "block_mm_infos": block_mm_infos,
        }

        pre_route_ms = (time.perf_counter() - handler_start) * 1000

        # Rewrite multi_modal_data URLs:
        # 1. Convert Url → RawUrl to skip url::Url::parse in Rust (~5ms savings in release)
        # 2. For HTTP URLs, replace with data URI so backend doesn't re-download the image
        rewrite_start = time.perf_counter()
        multi_modal_data = request.get("multi_modal_data")
        if multi_modal_data and image_urls:
            raw_bytes_list = processed.raw_bytes_list if processed.raw_bytes_list else []
            img_idx = 0
            for mm_type_items in multi_modal_data.values():
                if not isinstance(mm_type_items, list):
                    continue
                for item in mm_type_items:
                    if not isinstance(item, dict) or "Url" not in item:
                        continue
                    url_val = item["Url"]
                    # For HTTP URLs, replace with data URI from already-downloaded bytes
                    if url_val.startswith(("http://", "https://")) and img_idx < len(raw_bytes_list):
                        raw = raw_bytes_list[img_idx]
                        mime = _detect_mime_type(raw)
                        b64 = base64.b64encode(raw).decode("ascii")
                        item["RawUrl"] = f"data:{mime};base64,{b64}"
                    else:
                        item["RawUrl"] = url_val
                    del item["Url"]
                    img_idx += 1
        elif multi_modal_data:
            for mm_type_items in multi_modal_data.values():
                if isinstance(mm_type_items, list):
                    for item in mm_type_items:
                        if isinstance(item, dict) and "Url" in item:
                            item["RawUrl"] = item.pop("Url")

        # Strip image content from extra_args to reduce serialization payload.
        forwarded_extra_args = _strip_image_content(request.get("extra_args"))
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000

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
        kv_open_stream_ms = (time.perf_counter() - route_start) * 1000

        first_chunk = True
        async for response in stream:
            if first_chunk:
                first_chunk_ms = (time.perf_counter() - handler_start) * 1000
                kv_wait_first_chunk_ms = max(
                    0.0, first_chunk_ms - pre_route_ms - rewrite_ms - kv_open_stream_ms
                )
                logger.debug(
                    "[perf][mm_handler] generate: "
                    "first_chunk=%.3fms pre_route=%.3fms "
                    "process_mm=%.3fms build_block=%.3fms "
                    "rewrite_url=%.3fms kv_open_stream=%.3fms "
                    "kv_wait_first_chunk=%.3fms "
                    "images=%d tokens=%d",
                    first_chunk_ms,
                    pre_route_ms,
                    mm_ms,
                    block_mm_ms,
                    rewrite_ms,
                    kv_open_stream_ms,
                    kv_wait_first_chunk_ms,
                    len(image_urls),
                    len(token_ids),
                )
                first_chunk = False
            yield response
