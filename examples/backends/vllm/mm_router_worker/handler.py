# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

import logging
from typing import Any, AsyncGenerator

from dynamo.llm import KvPushRouter
from dynamo.runtime import Client
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
        client: Client,
        kv_push_router: KvPushRouter,
        instance_ids: list[int],
        tokenizer: Any,
        processor: Any,
        model: str,
        block_size: int,
    ):
        """
        Initialize the MM Router Handler.

        Args:
            client: Dynamo client for downstream vLLM workers
            kv_push_router: KvPushRouter for KV-aware worker selection and routing
            instance_ids: List of available worker instance IDs
            tokenizer: HuggingFace AutoTokenizer
            processor: HuggingFace AutoProcessor (optional)
            model: Model path/name
            block_size: KV cache block size
        """
        self.client = client
        self.kv_push_router = kv_push_router
        self.instance_ids = instance_ids
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
        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
            # Process multimodal: download images, compute mm_hash
            # Do not reuse request["token_ids"] for MM routing: those are placeholder-level
            # tokens from frontend. We need processor-expanded tokens to build block_mm_infos.
            # Request payload does not include a rendered template string; extra_args carries
            # original messages, so mm_processor reapplies chat template locally.
            processed = process_multimodal(
                messages=messages,
                image_urls=image_urls,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=self.model,
            )

            # Build block_mm_infos for MM-aware hash computation
            block_mm_infos = build_block_mm_infos(
                num_tokens=len(processed.tokens),
                block_size=self.block_size,
                mm_hashes=processed.mm_hashes,
                image_ranges=processed.image_ranges,
            )

            routing_tokens = processed.tokens
            routing_blocks = (len(routing_tokens) + self.block_size - 1) // self.block_size
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
            routing_blocks = (len(routing_tokens) + self.block_size - 1) // self.block_size
            logger.debug(
                f"Text request: {len(routing_tokens)} routing tokens, {routing_blocks} routing blocks"
            )
            block_mm_infos = None

        # Route and generate through KvPushRouter with explicit fields.
        # We pass:
        # - execution payload (token_ids + multi_modal_data)
        # - routing payload (routing_token_ids + block_mm_infos)
        # so generate() can select worker internally while preserving MM correctness.
        token_ids = request.get("token_ids")
        if not token_ids:
            raise ValueError("Missing or empty token_ids in preprocessed request")

        raw_extra_args = request.get("extra_args")
        if raw_extra_args is not None and not isinstance(raw_extra_args, dict):
            logger.warning(
                "request.extra_args is not a dict; replacing with routing-only extra_args"
            )
            extra_args: dict[str, Any] = {}
        else:
            extra_args = dict(raw_extra_args or {})
        extra_args["routing_token_ids"] = routing_tokens

        stream = await self.kv_push_router.generate(
            token_ids=token_ids,
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=extra_args,
            block_mm_infos=block_mm_infos,
            multi_modal_data=request.get("multi_modal_data"),
        )

        async for response in stream:
            yield response

    def update_instance_ids(self, instance_ids: list[int]) -> None:
        """Update the list of available worker instance IDs."""
        self.instance_ids = instance_ids
        logger.info(f"Updated instance IDs: {instance_ids}")
