# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best vLLM worker based on KV cache overlap.
"""

import logging
from copy import deepcopy
from typing import Any, AsyncGenerator

from dynamo._core import compute_block_hash_for_seq_py
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
        kv_push_router: KvPushRouter | None,
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

            # Compute block hashes WITH mm_info
            local_hashes = compute_block_hash_for_seq_py(
                processed.tokens, self.block_size, block_mm_infos
            )

            logger.debug(
                f"MM request: {len(processed.tokens)} tokens, "
                f"{len(image_urls)} images, {len(local_hashes)} blocks"
            )
            expanded_tokens = processed.tokens
        else:
            # Text-only: rely on frontend-preprocessed token_ids (ModelInput.Tokens contract)
            tokens = request.get("token_ids")
            if not tokens:
                raise ValueError(
                    "Missing or empty token_ids in preprocessed request for text-only routing"
                )

            local_hashes = compute_block_hash_for_seq_py(tokens, self.block_size, None)

            logger.debug(
                f"Text request: {len(tokens)} tokens, {len(local_hashes)} blocks"
            )
            expanded_tokens = tokens
            block_mm_infos = None

        total_blocks = len(local_hashes)

        # Find best worker based on KV cache overlap
        best_worker_id, dp_rank = await self._find_best_worker(
            expanded_tokens,
            block_mm_infos,
            total_blocks=total_blocks,
        )

        logger.info(
            f"Routing to worker {best_worker_id} (dp_rank={dp_rank}, "
            f"mm={'yes' if image_urls else 'no'})"
        )

        # If router is unavailable, fallback to direct forwarding.
        if self.kv_push_router is None:
            async for response in await self.client.direct(request, best_worker_id):
                yield response.data()
            return

        # Route and generate through KvPushRouter while preserving the full
        # PreprocessedRequest payload (especially multi_modal_data for Qwen mRoPE).
        routed_request = deepcopy(request)
        routing = routed_request.get("routing") or {}
        routing["backend_instance_id"] = best_worker_id
        routing["dp_rank"] = dp_rank
        routed_request["routing"] = routing

        stream = await self.kv_push_router.generate_from_request(routed_request)

        async for response in stream:
            yield response

    async def _find_best_worker(
        self,
        routing_tokens: list[int],
        block_mm_infos: list[dict | None] | None = None,
        total_blocks: int = 0,
    ) -> tuple[int, int]:
        """
        Find the worker with the highest KV cache overlap.

        Args:
            routing_tokens: Token IDs used for routing
            block_mm_infos: Optional block-level MM metadata aligned with routing_tokens

        Returns:
            Tuple of (worker_id, dp_rank)
        """
        if not self.instance_ids:
            raise ValueError("No workers available")

        if self.kv_push_router is None:
            logger.warning("No KvPushRouter available, using first worker")
            return self.instance_ids[0], 0

        try:
            best_worker_id, dp_rank, overlap_blocks = await self.kv_push_router.best_worker(
                token_ids=routing_tokens,
                block_mm_infos=block_mm_infos,
            )

            logger.info(
                "[ROUTING] "
                f"Best: worker_{best_worker_id} dp_rank={dp_rank} "
                f"with {overlap_blocks}/{total_blocks} blocks overlap"
            )

            return best_worker_id, dp_rank

        except Exception as e:
            logger.warning(f"KvPushRouter query failed: {e}, using first worker")
            return self.instance_ids[0], 0

    def update_instance_ids(self, instance_ids: list[int]) -> None:
        """Update the list of available worker instance IDs."""
        self.instance_ids = instance_ids
        logger.info(f"Updated instance IDs: {instance_ids}")
