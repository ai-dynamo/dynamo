# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

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
        t0 = time.perf_counter()
        t0_ns = time.time_ns()

        # Hop 1: frontend → MM router TCP transfer
        t_frontend_send_ns = request.get("extra_args", {}).get("_t_frontend_send_ns")
        if t_frontend_send_ns:
            hop1_ms = (t0_ns - t_frontend_send_ns) / 1e6
            logger.info(f"[timing] hop1_frontend_to_router={hop1_ms:.1f}ms")

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
            t1 = time.perf_counter()

            # Build block_mm_infos for MM-aware hash computation
            block_mm_infos = build_block_mm_infos(
                num_tokens=len(processed.tokens),
                block_size=self.block_size,
                mm_hashes=processed.mm_hashes,
                image_ranges=processed.image_ranges,
            )
            t2 = time.perf_counter()

            if block_mm_infos is None:
                raise ValueError(
                    "Failed to build block_mm_infos for multimodal request"
                )

            routing_tokens = processed.tokens
            routing_blocks = (
                len(routing_tokens) + self.block_size - 1
            ) // self.block_size
            logger.info(
                f"[timing] n_images={len(image_urls)} n_urls_bytes={sum(len(u) for u in image_urls)} "
                f"process_multimodal={1000*(t1-t0):.1f}ms "
                f"build_block_mm_infos={1000*(t2-t1):.1f}ms "
                f"routing_tokens={len(routing_tokens)} routing_blocks={routing_blocks}"
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
            t1 = t2 = time.perf_counter()
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
        t3 = time.perf_counter()

        first_chunk = True
        async for response in stream:
            if first_chunk:
                t4 = time.perf_counter()
                logger.info(
                    f"[timing] kv_router.generate={1000*(t3-t2):.1f}ms "
                    f"first_chunk={1000*(t4-t3):.1f}ms "
                    f"total_handler={1000*(t4-t0):.1f}ms"
                )
                first_chunk = False
            yield response
