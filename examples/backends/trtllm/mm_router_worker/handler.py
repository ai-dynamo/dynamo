# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best TRT-LLM worker based on KV cache overlap.
"""

import logging
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging
from examples.backends.mm_router_worker_utils import rewrite_multimodal_data_urls

from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Skip data URI replacement for images larger than 5MB base64-encoded
MAX_DATA_URI_SIZE = 5 * 1024 * 1024


class MMRouterHandler:
    """
    Handler that computes mm_hash for multimodal requests and routes
    to the best worker based on KV cache overlap.
    """

    def __init__(
        self,
        kv_router: KvRouter,
        tokenizer: Any,
        processor: Any,
        model: str,
        model_type: str,
        block_size: int,
    ):
        """
        Initialize the MM Router Handler.

        Args:
            kv_router: KvRouter for KV-aware worker selection and routing
            tokenizer: TRT-LLM tokenizer
            processor: HuggingFace AutoProcessor (optional)
            model: Model path/name
            model_type: Model type (e.g., "qwen2_vl")
            block_size: KV cache block size
        """
        self.kv_router = kv_router
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.model_type = model_type
        self.block_size = block_size
        self.image_loader = ImageLoader()

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
            Response chunks from the downstream TRT-LLM worker via KvRouter
        """
        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)
        multi_modal_data = request.get("multi_modal_data")

        if image_urls:
            # Process multimodal: download images, compute mm_hash
            # Do not reuse request["token_ids"] for MM routing: those are placeholder-level
            # tokens from frontend. We need processor-expanded tokens to build block_mm_infos.
            processed = await process_multimodal(
                messages=messages,
                image_urls=image_urls,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=self.model,
                model_type=self.model_type,
                image_loader=self.image_loader,
            )
            if processed.data_uris:
                multi_modal_data = rewrite_multimodal_data_urls(
                    multi_modal_data=multi_modal_data,
                    rewritten_image_urls=processed.data_uris,
                    logger=logger,
                )

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
            token_ids = request.get("token_ids")
            if not token_ids:
                raise ValueError(
                    "Missing or empty token_ids in preprocessed request for text-only routing"
                )

            routing_tokens = token_ids
            routing_blocks = (
                len(routing_tokens) + self.block_size - 1
            ) // self.block_size
            logger.debug(
                f"Text request: {len(routing_tokens)} routing tokens, "
                f"{routing_blocks} routing blocks"
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

        # Replace HTTP URLs with base64 data URIs so backend doesn't re-download
        multi_modal_data = request.get("multi_modal_data")
        if image_urls and multi_modal_data and processed.data_uris:
            image_url_items = multi_modal_data.get("image_url", [])
            for i, data_uri in enumerate(processed.data_uris):
                if i < len(image_url_items):
                    item = image_url_items[i]
                    if isinstance(item, dict) and "Url" in item:
                        if urlparse(item["Url"]).scheme in ("http", "https"):
                            if len(data_uri) <= MAX_DATA_URI_SIZE:
                                item["Url"] = data_uri

        mm_routing_info: dict[str, Any] = {
            "routing_token_ids": routing_tokens,
            "block_mm_infos": block_mm_infos,
        }

        stream = await self.kv_router.generate(
            token_ids=token_ids,
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=request.get("extra_args"),
            multi_modal_data=multi_modal_data,
            mm_routing_info=mm_routing_info,
        )

        async for response in stream:
            yield response
