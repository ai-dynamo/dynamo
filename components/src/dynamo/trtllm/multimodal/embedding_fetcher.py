# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Embedding fetcher utilities for multimodal processing with caching.

Provides utility functions for fetching image embeddings from remote encoder
with per-URL caching support.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.common.multimodal.async_encoder_cache import EncoderCacheManager
from dynamo.trtllm.multimodal.cuda_ipc import extract_embeddings_from_handles
from dynamo.trtllm.multimodal.hasher import MultimodalHasher

logger = logging.getLogger(__name__)


async def fetch_embeddings_from_encoder(
    image_urls: List[str],
    request: Dict[str, Any],
    encode_client: Any,
    encoder_cache: Optional[EncoderCacheManager] = None,
) -> Union[List[torch.Tensor], DisaggregatedParams]:
    """
    Fetch embeddings from remote encode worker.

    Args:
        image_urls: List of image URLs to encode (must not be empty)
        request: Request dict (used for creating modified requests for caching)
        encode_client: Client to call remote encode worker
        encoder_cache: Optional cache for embeddings

    Returns:
        - List[torch.Tensor]: When using cache (CPU tensors from cache)
        - DisaggregatedParams: When not using cache (contains CUDA IPC handles)

    Raises:
        ValueError: If image_urls is empty
    """
    if not image_urls:
        raise ValueError("image_urls must not be empty")

    logger.info(f"fetch_embeddings_from_encoder: image_urls={image_urls}")

    if encoder_cache:
        # Cache path: extract embeddings to CPU tensors
        return await _fetch_embeddings_with_cache(
            image_urls,
            request,
            encoder_cache,
            lambda req: _remote_encode_full_epd(
                req, encode_client, update_request_for_decode=False
            ),
        )
    else:
        # No cache: return DisaggregatedParams directly (no GPU→CPU extraction)
        return await _remote_encode_full_epd(
            request, encode_client, update_request_for_decode=True
        )


async def _remote_encode_full_epd(
    request: Dict[str, Any],
    encode_client: Any,
    update_request_for_decode: bool = True,
) -> DisaggregatedParams:
    """
    Call encode worker for full EPD flow.

    Args:
        request: Request dict
        encode_client: Client to call remote encode worker
        update_request_for_decode: If True, store EPD metadata in request

    Returns:
        DisaggregatedParams with multimodal_embedding_handles

    Raises:
        RuntimeError: If encode worker returns invalid response
    """
    encode_response = None
    async for res in await encode_client.round_robin(request):
        encode_response = res.data()
        break

    if not encode_response:
        raise RuntimeError("Did not receive a response from the encode worker.")

    if "ep_disaggregated_params" not in encode_response:
        raise RuntimeError("Encode response missing ep_disaggregated_params.")

    params_dict = encode_response["ep_disaggregated_params"]
    if params_dict is None:
        raise RuntimeError("ep_disaggregated_params is None.")

    # Store EPD metadata in request for decode worker (only when not using cache)
    if update_request_for_decode:
        if "processed_prompt" in encode_response:
            request["_epd_processed_prompt"] = encode_response["processed_prompt"]
        if "prompt_token_ids" in encode_response:
            request["_epd_prompt_token_ids"] = encode_response["prompt_token_ids"]

    return DisaggregatedParams(**params_dict)


async def _fetch_embeddings_with_cache(
    image_urls: List[str],
    request: Dict[str, Any],
    cache: EncoderCacheManager,
    encode_fn: Callable[[Dict[str, Any]], DisaggregatedParams],
) -> List[torch.Tensor]:
    """
    Encode image URLs with per-URL caching and partial cache usage.

    Checks cache for each URL. Cached embeddings are reused directly.
    For uncached URLs, sends a single encode request for only those URLs,
    then caches the results.

    Args:
        image_urls: List of image URLs to encode
        request: Original request dict containing the images
        cache: AsyncEncoderCache instance for caching embeddings
        encode_fn: Async function that encodes a request and returns ep_disaggregated_params
                   Should accept a modified request dict with subset of URLs

    Returns:
        List of embedding tensors for all images in original order
    """
    if not image_urls:
        raise ValueError("image_urls list is empty")

    # Check cache for each URL
    embeddings_with_index = []  # List of (original_index, tensor)
    uncached_urls = []
    uncached_indices = []
    uncached_hashes = []

    for i, url in enumerate(image_urls):
        url_hash = MultimodalHasher.hash_bytes(url.encode())
        cached = cache.get(url_hash)
        if cached is not None:
            logger.info(f"fetch_embeddings_with_cache: cache hit for URL: {url}")
            embeddings_with_index.append((i, cached))
        else:
            logger.info(f"fetch_embeddings_with_cache: cache miss for URL: {url}")
            uncached_urls.append(url)
            uncached_indices.append(i)
            uncached_hashes.append(url_hash)

    # If all cached, return immediately
    if not uncached_urls:
        logger.info(f"fetch_embeddings_with_cache: all {len(image_urls)} URLs cached")
        embeddings_with_index.sort(key=lambda x: x[0])
        tensors = [t for _, t in embeddings_with_index]
        return tensors

    # Encode uncached URLs
    logger.info(
        f"fetch_embeddings_with_cache: encoding {len(uncached_urls)} uncached URLs"
    )

    # Create modified request with only uncached URLs
    modified_request = _create_request_with_urls(request, uncached_urls)

    # Call encode function
    ep_disaggregated_params = await encode_fn(modified_request)
    if not ep_disaggregated_params:
        raise RuntimeError(
            "fetch_embeddings_with_cache: Failed to get ep_disaggregated_params"
        )

    # Extract handles from disaggregated params
    handles = getattr(ep_disaggregated_params, "multimodal_embedding_handles", None)
    if not handles:
        raise RuntimeError(
            "fetch_embeddings_with_cache: No multimodal_embedding_handles in ep_disaggregated_params"
        )

    # Extract tensors from CUDA IPC handles
    new_tensors = await extract_embeddings_from_handles(handles)

    # Cache new tensors (reuse hashes computed during cache lookup)
    for url, url_hash, tensor in zip(uncached_urls, uncached_hashes, new_tensors):
        cache.set(url_hash, tensor)
        logger.info(
            f"fetch_embeddings_with_cache: cached embedding for URL: {url}, shape: {tensor.shape}"
        )

    # Add new tensors to our list with their original indices
    for idx, tensor in zip(uncached_indices, new_tensors):
        embeddings_with_index.append((idx, tensor))

    # Sort by original order and return list
    embeddings_with_index.sort(key=lambda x: x[0])
    tensors = [t for _, t in embeddings_with_index]
    return tensors


def _create_request_with_urls(
    original_request: Dict[str, Any], image_urls: List[str]
) -> Dict[str, Any]:
    """
    Create a modified request containing only specified image URLs.

    Args:
        original_request: Original request dict
        image_urls: URLs to include in the modified request

    Returns:
        Modified request dict with filtered image URLs
    """
    # Deep copy to avoid modifying original
    import copy

    modified_request = copy.deepcopy(original_request)

    # Extract messages
    messages = modified_request.get("extra_args", {}).get(
        "messages", modified_request.get("messages", [])
    )

    # Filter messages to only include specified URLs
    filtered_messages = []
    for message in messages:
        new_message = {"role": message.get("role", "user"), "content": []}

        for content in message.get("content", []):
            if isinstance(content, dict):
                if content.get("type") == "image_url":
                    # Only include if URL is in our list
                    url = content.get("image_url", {}).get("url")
                    if url in image_urls:
                        new_message["content"].append(content)
                elif content.get("type") == "text":
                    # Keep text content
                    new_message["content"].append(content)
            elif isinstance(content, str):
                new_message["content"].append(content)

        if new_message["content"]:
            filtered_messages.append(new_message)

    # Update the request with filtered messages
    if "extra_args" in modified_request:
        modified_request["extra_args"]["messages"] = filtered_messages
    else:
        modified_request["messages"] = filtered_messages

    return modified_request
