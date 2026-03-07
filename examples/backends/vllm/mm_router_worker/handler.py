# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

import base64
import logging
import time
from typing import Any, AsyncGenerator

from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging

from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)


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
        """
        Initialize the MM Router Handler.

        Args:
            kv_router: KvRouter for KV-aware worker selection and routing
            tokenizer: HuggingFace AutoTokenizer
            processor: HuggingFace AutoProcessor (optional)
            model: Model path/name
            block_size: KV cache block size
        """
        self.kv_router = kv_router
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.block_size = block_size

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
        t_start = time.perf_counter()

        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
            # Process multimodal: download images, compute mm_hash
            # Do not reuse request["token_ids"] for MM routing: those are placeholder-level
            # tokens from frontend. We need processor-expanded tokens to build block_mm_infos.
            # Request payload does not include a rendered template string; extra_args carries
            # original messages, so mm_processor reapplies chat template locally.
            t_pre = time.perf_counter()
            processed, raw_bytes_list = process_multimodal(
                messages=messages,
                image_urls=image_urls,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=self.model,
            )
            t_process = time.perf_counter()

            # (#3) Strip image content from messages to reduce serialization payload
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            part["image_url"]["url"] = "<stripped>"

            # (#5) Convert HTTP URLs to data URIs + (#2) Use RawUrl key
            mm_data = request.get("multi_modal_data", {})
            if "image_url" in mm_data:
                for i, item in enumerate(mm_data["image_url"]):
                    if "Url" in item:
                        url_val = item["Url"]
                        if isinstance(url_val, str) and url_val.startswith(
                            ("http://", "https://")
                        ):
                            raw = raw_bytes_list[i]
                            mime = _detect_mime(raw)
                            b64 = base64.b64encode(raw).decode("ascii")
                            data_uri = f"data:{mime};base64,{b64}"
                        else:
                            data_uri = url_val if isinstance(url_val, str) else str(url_val)
                        # Use RawUrl to skip url::Url::parse() in Rust depythonize
                        del item["Url"]
                        item["RawUrl"] = data_uri

            # Build block_mm_infos for MM-aware hash computation
            block_mm_infos = build_block_mm_infos(
                num_tokens=len(processed.tokens),
                block_size=self.block_size,
                mm_hashes=processed.mm_hashes,
                image_ranges=processed.image_ranges,
            )
            if block_mm_infos is None:
                raise ValueError(
                    "Failed to build block_mm_infos for multimodal request"
                )

            t_rewrite = time.perf_counter()

            routing_tokens = processed.tokens
            routing_blocks = (
                len(routing_tokens) + self.block_size - 1
            ) // self.block_size
            logger.debug(
                f"MM request: {len(routing_tokens)} routing tokens, "
                f"{len(image_urls)} images, {routing_blocks} routing blocks"
            )
        else:
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

        mm_routing_info: dict[str, Any] = {
            "routing_token_ids": routing_tokens,
            "block_mm_infos": block_mm_infos,
        }

        t_pre_route = time.perf_counter()
        stream = await self.kv_router.generate(
            token_ids=token_ids,
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=request.get("extra_args"),
            multi_modal_data=request.get("multi_modal_data"),
            mm_routing_info=mm_routing_info,
        )
        t_stream_open = time.perf_counter()

        first_chunk = True
        async for response in stream:
            if first_chunk:
                t_first_chunk = time.perf_counter()
                is_mm = bool(image_urls) if 'image_urls' in dir() else False
                logger.info(
                    f"[perf][mm_handler] "
                    f"pre_route={1000*(t_pre_route - t_start):.2f}ms "
                    f"kv_open_stream={1000*(t_stream_open - t_pre_route):.2f}ms "
                    f"first_chunk_wait={1000*(t_first_chunk - t_stream_open):.2f}ms "
                    f"total_to_first_chunk={1000*(t_first_chunk - t_start):.2f}ms"
                    + (
                        f" | process_mm={1000*(t_process - t_pre):.2f}ms"
                        f" rewrite={1000*(t_rewrite - t_process):.2f}ms"
                        if is_mm and 't_process' in dir() else ""
                    )
                )
                first_chunk = False
            yield response


def _detect_mime(raw_bytes: bytes) -> str:
    """Detect MIME type from raw image bytes using magic bytes."""
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    if raw_bytes[:4] == b"GIF8":
        return "image/gif"
    return "application/octet-stream"
