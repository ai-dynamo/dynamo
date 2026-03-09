# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MM Router Handler — routes multimodal requests via KV-cache-aware worker selection."""

import logging
from typing import Any, AsyncGenerator

from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging

from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class MMRouterHandler:
    """Routes requests to the vLLM worker with the best KV cache overlap."""

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

    async def generate(self, request: dict) -> AsyncGenerator[dict, None]:
        """Main entry point: process request, compute routing, forward to best worker."""
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
            routing_tokens, block_mm_infos = self._process_mm_request(
                request, messages, image_urls
            )
        else:
            routing_tokens = request.get("token_ids")
            if not routing_tokens:
                raise ValueError("Missing token_ids in preprocessed request")
            n_blocks = (len(routing_tokens) + self.block_size - 1) // self.block_size
            block_mm_infos = [None] * n_blocks

        stream = await self.kv_router.generate(
            token_ids=request.get("token_ids"),
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=request.get("extra_args"),
            multi_modal_data=request.get("multi_modal_data"),
            mm_routing_info={
                "routing_token_ids": routing_tokens,
                "block_mm_infos": block_mm_infos,
            },
        )
        async for response in stream:
            yield response

    def _process_mm_request(
        self,
        request: dict,
        messages: list[dict],
        image_urls: list[str],
    ) -> tuple[list[int], list[dict | None]]:
        """Process multimodal: load images, expand tokens, build routing info."""
        processed = process_multimodal(
            messages=messages,
            image_urls=image_urls,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model=self.model,
        )

        # Strip image content from messages to reduce serialization payload
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = "<stripped>"

        # Rewrite Url → RawUrl to skip url::Url::parse in Rust depythonize
        mm_data = request.get("multi_modal_data", {})
        if isinstance(mm_data, dict):
            for item in mm_data.get("image_url", []):
                if isinstance(item, dict) and "Url" in item:
                    item["RawUrl"] = item.pop("Url")

        block_mm_infos = build_block_mm_infos(
            num_tokens=len(processed.tokens),
            block_size=self.block_size,
            mm_hashes=processed.mm_hashes,
            image_ranges=processed.image_ranges,
        )
        if block_mm_infos is None:
            raise ValueError("Failed to build block_mm_infos")

        return processed.tokens, block_mm_infos
