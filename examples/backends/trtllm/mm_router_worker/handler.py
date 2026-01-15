# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Handler - Routes requests to best worker based on KV cache overlap.
"""

import logging
from typing import Any, AsyncGenerator

from dynamo._core import KvIndexer, compute_block_hash_for_seq_py
from dynamo.runtime import Client

from .mm_processor import (
    ProcessedInput,
    build_block_mm_infos,
    build_prompt_from_messages,
    compute_mm_hashes,
    extract_image_urls,
    get_mm_tokens,
    process_multimodal,
)

logger = logging.getLogger(__name__)


class MMRouterHandler:
    """
    Handler that computes mm_hash for multimodal requests and routes
    to the best worker based on KV cache overlap.
    """

    def __init__(
        self,
        client: Client,
        indexer: KvIndexer,
        instance_ids: list[int],
        tokenizer: Any,
        processor: Any,
        model: str,
        model_type: str,
        block_size: int,
    ):
        """
        Initialize the MM Router Handler.

        Args:
            client: Dynamo client for downstream TRT-LLM workers
            indexer: KvIndexer for querying worker cache states
            instance_ids: List of available worker instance IDs
            tokenizer: TRT-LLM tokenizer
            processor: HuggingFace AutoProcessor (optional)
            model: Model path/name
            model_type: Model type (e.g., "qwen2_vl")
            block_size: KV cache block size
        """
        self.client = client
        self.indexer = indexer
        self.instance_ids = instance_ids
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.model_type = model_type
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
            Response chunks from the downstream TRT-LLM worker
        """
        # Extract messages from extra_args (set by Frontend preprocessor)
        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)

        if image_urls:
            # Process multimodal: download images, compute mm_hash
            processed = process_multimodal(
                messages=messages,
                image_urls=image_urls,
                tokenizer=self.tokenizer,
                processor=self.processor,
                model=self.model,
                model_type=self.model_type,
            )

            # Build block_mm_infos for MM-aware hash computation
            block_mm_infos = build_block_mm_infos(
                num_tokens=len(processed.tokens),
                block_size=self.block_size,
                mm_hashes=processed.mm_hashes,
                image_offsets_list=processed.image_offsets_list,
            )

            # Compute block hashes WITH mm_info
            local_hashes = compute_block_hash_for_seq_py(
                processed.tokens, self.block_size, block_mm_infos
            )

            logger.debug(
                f"MM request: {len(processed.tokens)} tokens, "
                f"{len(image_urls)} images, {len(local_hashes)} blocks"
            )
        else:
            # Text-only: tokenize messages for routing
            tokens = request.get("token_ids", [])
            if not tokens:
                # Tokenize from messages if token_ids not provided
                prompt = self._apply_chat_template(messages)
                tokens = self.tokenizer.encode(prompt)
                logger.debug(f"Tokenized text-only prompt: {len(tokens)} tokens")

            local_hashes = compute_block_hash_for_seq_py(
                tokens, self.block_size, None
            )

            logger.debug(
                f"Text request: {len(tokens)} tokens, {len(local_hashes)} blocks"
            )

        # Find best worker based on KV cache overlap
        best_worker_id = await self._find_best_worker(local_hashes)

        logger.info(
            f"Routing to worker {best_worker_id} "
            f"(mm={'yes' if image_urls else 'no'})"
        )

        # Forward ORIGINAL request to the selected worker
        # TRT-LLM worker will process images itself
        async for response in await self.client.direct(request, best_worker_id):
            yield response.data()

    async def _find_best_worker(self, local_hashes: list[int]) -> int:
        """
        Find the worker with the highest KV cache overlap.

        Args:
            local_hashes: Block hashes for the current request

        Returns:
            Instance ID of the best worker
        """
        if not self.instance_ids:
            raise ValueError("No workers available")

        if self.indexer is None:
            logger.warning("No indexer available, using first worker")
            return self.instance_ids[0]

        try:
            # Query indexer for overlap scores
            logger.info(f"Querying indexer with {len(local_hashes)} block hashes")
            logger.info(f"First 5 hashes: {local_hashes[:5]}")
            overlap_scores = await self.indexer.find_matches(local_hashes)
            scores_dict = overlap_scores.scores
            # Check tree_sizes to see if indexer has any blocks stored
            tree_sizes = getattr(overlap_scores, 'tree_sizes', {})
            logger.info(f"Indexer returned scores_dict: {scores_dict}")
            logger.info(f"Indexer tree_sizes (total blocks per worker): {tree_sizes}")

            # Find worker with highest overlap
            best_worker_id = self.instance_ids[0]
            best_score = 0

            # Build a map from worker_id to score
            # scores_dict keys are (worker_id, dp_rank) tuples, not just worker_id
            worker_id_to_score = {}
            for key, score in scores_dict.items():
                if isinstance(key, tuple):
                    wid = key[0]  # Extract worker_id from (worker_id, dp_rank)
                else:
                    wid = key  # Backwards compatibility with int keys
                # Sum scores across dp_ranks for same worker
                worker_id_to_score[wid] = worker_id_to_score.get(wid, 0) + score

            # Log all worker scores for debugging
            worker_scores = []
            for worker_id in self.instance_ids:
                score = worker_id_to_score.get(worker_id, 0)
                worker_scores.append(f"worker_{worker_id}={score}")
                if score > best_score:
                    best_score = score
                    best_worker_id = worker_id

            # Always log routing decision at INFO level for visibility
            logger.info(
                f"[ROUTING] Scores: [{', '.join(worker_scores)}] | "
                f"Best: worker_{best_worker_id} with {best_score}/{len(local_hashes)} blocks overlap"
            )

            return best_worker_id

        except Exception as e:
            logger.warning(f"Indexer query failed: {e}, using first worker")
            return self.instance_ids[0]

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template to messages for tokenization."""
        try:
            # Try using tokenizer's chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception as e:
            logger.debug(f"Chat template failed: {e}")

        # Fallback to simple prompt building
        return build_prompt_from_messages(messages)

    def update_instance_ids(self, instance_ids: list[int]) -> None:
        """Update the list of available worker instance IDs."""
        self.instance_ids = instance_ids
        logger.info(f"Updated instance IDs: {instance_ids}")
