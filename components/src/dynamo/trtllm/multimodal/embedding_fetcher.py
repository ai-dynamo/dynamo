# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Embedding fetcher utilities for multimodal processing with caching.

Provides utility functions for fetching image embeddings from remote encoder
with per-URL caching support, and media items (images and videos) with a
per-item cache when TRT-LLM can encode items without the prompt.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.common.http import HttpStatusError
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.image_loader import (
    IMAGE_CACHE_SCOPE_KEY,
    image_cache_scope_from_request,
    image_cache_session_scoped_from_env,
    scope_image_cache_key,
)
from dynamo.common.utils.token_ids import normalize_request_token_ids
from dynamo.trtllm.multimodal.cuda_ipc import extract_embeddings_from_handles
from dynamo.trtllm.multimodal.hasher import MultimodalHasher
from dynamo.trtllm.multimodal_processor import resolve_mm_processor_kwargs

logger = logging.getLogger(__name__)


async def fetch_embeddings_from_encoder(
    image_urls: List[str],
    request: Dict[str, Any],
    encode_client: Any,
    encoder_cache: Optional[MultimodalEmbeddingCacheManager] = None,
    trace_context=None,
) -> Union[List[Union[torch.Tensor, Dict[str, Any]]], DisaggregatedParams]:
    """
    Fetch embeddings from remote encode worker.

    Args:
        image_urls: List of image URLs to encode (must not be empty)
        request: Request dict (used for creating modified requests for caching)
        encode_client: Client to call remote encode worker
        encoder_cache: Optional cache for embeddings
        trace_context: Optional Dynamo context for OTel trace propagation

    Returns:
        - List: When using cache, one CPU embedding per image (see
          _embedding_inputs)
        - DisaggregatedParams: When not using cache (contains CUDA IPC handles)

    Raises:
        ValueError: If image_urls is empty
    """
    if not image_urls:
        raise ValueError("image_urls must not be empty")

    if encoder_cache:
        # Cache path: extract embeddings to CPU tensors
        return await _fetch_embeddings_with_cache(
            image_urls,
            request,
            encoder_cache,
            lambda req: _remote_encode_full_epd(
                req,
                encode_client,
                update_request_for_decode=False,
                trace_context=trace_context,
            ),
        )
    else:
        # No cache: return DisaggregatedParams directly (no GPU→CPU extraction)
        return await _remote_encode_full_epd(
            request,
            encode_client,
            update_request_for_decode=True,
            trace_context=trace_context,
        )


async def _remote_encode_full_epd(
    request: Dict[str, Any],
    encode_client: Any,
    update_request_for_decode: bool = True,
    trace_context=None,
) -> DisaggregatedParams:
    """
    Call encode worker for full EPD flow.

    Args:
        request: Request dict
        encode_client: Client to call remote encode worker
        update_request_for_decode: If True, store EPD metadata in request
        trace_context: Optional Dynamo context for OTel trace propagation

    Returns:
        DisaggregatedParams with multimodal_embedding_handles

    Raises:
        RuntimeError: If encode worker returns invalid response
    """
    encode_response = None
    # The encode worker may sit behind a JSON codec, which would spell a packed
    # buffer out one byte per element; forward a list.
    forwarded = normalize_request_token_ids(dict(request))
    async for res in await encode_client.round_robin(forwarded, context=trace_context):
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


async def fetch_mm_items_from_encoder(
    media_urls: Dict[str, List[str]],
    request: Dict[str, Any],
    encode_client: Any,
    encoder_cache: Optional[MultimodalEmbeddingCacheManager] = None,
    trace_context=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Encode media items on their own, without the prompt, with a per-item cache.

    For TRT-LLM builds whose input processor sets ``supports_mm_item_handoff``.
    Each item is looked up in the cache by its modality, URL and processor
    overrides. The encode worker gets only the missed items and no prompt, in
    one call per modality (Qwen2.5-VL cannot encode images and videos in one
    call). Each new item is cached with its metadata from the encoder: grid,
    video timing and ``multimodal_hash``. Without a cache, every item is a miss.

    Args:
        media_urls: Media URLs by modality ("image", "video"), in request order
        request: Request dict (processor overrides and cache scope)
        encode_client: Client to call remote encode worker
        encoder_cache: Optional cache for embeddings
        trace_context: Optional Dynamo context for OTel trace propagation

    Returns:
        Items by modality for TRT-LLM's ``multi_modal_embeddings``, each
        ``{"mm_embeddings": tensor, **metadata}``, in the order of
        ``media_urls[modality]``: the order of that modality's placeholders in
        the prompt. TRT-LLM builds the prompt and its positions from them.
    """
    mm_kwargs = _cache_mm_kwargs(request)
    cache_scope = image_cache_scope_from_request(request)
    session_scoped_cache = image_cache_session_scoped_from_env()

    def _cache_key(modality: str, url: str) -> str | None:
        # The same URL as an image and as a video gives different items.
        cache_key = MultimodalHasher.hash_bytes(
            json.dumps([modality, url, mm_kwargs], sort_keys=True, default=str).encode()
        )
        return scope_image_cache_key(
            cache_key,
            cache_scope,
            session_scoped_cache=session_scoped_cache,
        )

    items: Dict[str, List[Any]] = {}
    misses: Dict[str, List[Tuple[int, str, str | None]]] = {}
    for modality, urls in media_urls.items():
        items[modality] = [None] * len(urls)
        for index, url in enumerate(urls):
            key = _cache_key(modality, url) if encoder_cache is not None else None
            cached = (
                encoder_cache.get(key)
                if encoder_cache is not None and key is not None
                else None
            )
            if cached is not None and cached.metadata is not None:
                items[modality][index] = cached
            else:
                misses.setdefault(modality, []).append((index, url, key))

    for modality, missed in misses.items():
        encode_request: Dict[str, Any] = {
            "multi_modal_data": {
                f"{modality}_url": [{"Url": url} for _, url, _ in missed]
            }
        }
        if mm_kwargs:
            encode_request["mm_processor_kwargs"] = mm_kwargs
        if cache_scope is not None:
            encode_request[IMAGE_CACHE_SCOPE_KEY] = cache_scope
        params = await _remote_encode_full_epd(
            encode_request,
            encode_client,
            update_request_for_decode=False,
            trace_context=trace_context,
        )
        handles = params.multimodal_embedding_handles or []
        if len(handles) != len(missed):
            raise RuntimeError(
                f"The encode worker returned {len(handles)} embeddings for "
                f"{len(missed)} {modality} items"
            )
        metadata = [dict(handle.get("metadata") or {}) for handle in handles]
        tensors = await extract_embeddings_from_handles(handles)
        for (index, _, key), tensor, item_metadata in zip(missed, tensors, metadata):
            entry = CachedEmbedding(tensor=tensor, metadata=item_metadata)
            if encoder_cache is not None and key is not None:
                encoder_cache.set(key, entry)
            items[modality][index] = entry

    return {
        modality: [{"mm_embeddings": e.tensor, **(e.metadata or {})} for e in entries]
        for modality, entries in items.items()
    }


async def _fetch_embeddings_with_cache(
    image_urls: List[str],
    request: Dict[str, Any],
    cache: MultimodalEmbeddingCacheManager,
    encode_fn: Callable[[Dict[str, Any]], DisaggregatedParams],
) -> List[Union[torch.Tensor, Dict[str, Any]]]:
    """
    Encode image URLs with per-URL caching and partial cache usage.

    Checks cache for each URL. Cached embeddings are reused directly.
    For uncached URLs, sends a single encode request for only those URLs,
    then caches the results. Session-scoped mode partitions keys by the
    request's image-cache scope and bypasses the cache when the scope is absent.

    Args:
        image_urls: List of image URLs to encode
        request: Original request dict containing the images
        cache: AsyncEncoderCache instance for caching embeddings
        encode_fn: Async function that encodes a request and returns ep_disaggregated_params
                   Should accept a modified request dict with subset of URLs

    Returns:
        One embedding per image in original order (see _embedding_inputs)
    """
    if not image_urls:
        raise ValueError("image_urls list is empty")

    # Check cache for each URL
    embeddings_with_index = []  # List of (original_index, CachedEmbedding)
    uncached_urls = []
    uncached_indices = []
    uncached_hashes = []

    # Overrides change the embeddings for a URL, so they are part of cache
    # identity; without them the hash is unchanged, so existing entries stay valid.
    mm_kwargs = _cache_mm_kwargs(request)

    cache_scope = image_cache_scope_from_request(request)
    session_scoped_cache = image_cache_session_scoped_from_env()

    def _cache_key(url: str) -> str | None:
        # JSON-encode the pair rather than concatenating: `url + salt` is ambiguous,
        # so a URL ending in another request's serialized overrides would collide
        # with it and be served the wrong embeddings.
        if mm_kwargs is None:
            cache_key = MultimodalHasher.hash_bytes(url.encode())
        else:
            cache_key = MultimodalHasher.hash_bytes(
                json.dumps([url, mm_kwargs], sort_keys=True, default=str).encode()
            )
        return scope_image_cache_key(
            cache_key,
            cache_scope,
            session_scoped_cache=session_scoped_cache,
        )

    for i, url in enumerate(image_urls):
        url_hash = _cache_key(url)
        cached = cache.get(url_hash) if url_hash is not None else None
        if cached is not None:
            embeddings_with_index.append((i, cached))
        else:
            uncached_urls.append(url)
            uncached_indices.append(i)
            uncached_hashes.append(url_hash)

    # If all cached, return immediately
    if not uncached_urls:
        return _embedding_inputs(embeddings_with_index)

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

    # The encoder may put each image's grid on its handle; read it before the
    # handles are used up
    grids = [(handle.get("metadata") or {}).get("image_grid_thw") for handle in handles]

    # Extract tensors from CUDA IPC handles
    new_tensors = await extract_embeddings_from_handles(handles)

    # Cache new embeddings with their grids (reuse hashes computed during
    # cache lookup) and add them with their original indices
    for idx, url_hash, tensor, grid in zip(
        uncached_indices, uncached_hashes, new_tensors, grids
    ):
        entry = CachedEmbedding(tensor=tensor, image_grid_thw=grid)
        if url_hash is not None:
            cache.set(url_hash, entry)
        embeddings_with_index.append((idx, entry))

    return _embedding_inputs(embeddings_with_index)


def _cache_mm_kwargs(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The request's processor overrides, or None when it has none.

    Reject before any cache lookup, not after: normalizing a malformed value to
    None would hash the plain URL and serve a cache hit with 200, while the
    aggregated path 400s on the same input.
    """
    mm_kwargs = (
        resolve_mm_processor_kwargs(request) if isinstance(request, dict) else None
    )
    if mm_kwargs is not None and not isinstance(mm_kwargs, dict):
        raise HttpStatusError(
            400,
            "Malformed mm_processor_kwargs field: expected an object",
            str(mm_kwargs),
        )
    return mm_kwargs or None


def _embedding_inputs(
    embeddings_with_index: List[Tuple[int, CachedEmbedding]],
) -> List[Union[torch.Tensor, Dict[str, Any]]]:
    """Return each image's embedding in request order, with its grid if known.

    TRT-LLM needs each image's grid to set its mRoPE positions. When the
    encoder gave no grid, the embedding goes as a bare tensor, as before.
    """
    embeddings_with_index.sort(key=lambda x: x[0])
    return [
        entry.tensor
        if entry.image_grid_thw is None
        else {"mm_embeddings": entry.tensor, "image_grid_thw": entry.image_grid_thw}
        for _, entry in embeddings_with_index
    ]


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

    mm_data = modified_request.get("multi_modal_data")
    if isinstance(mm_data, dict) and isinstance(mm_data.get("image_url"), list):
        filtered_items: List[Any] = []
        for item in mm_data["image_url"]:
            if isinstance(item, dict) and item.get("Url") in image_urls:
                filtered_items.append(item)
            elif isinstance(item, str) and item in image_urls:
                filtered_items.append(item)
        mm_data["image_url"] = filtered_items

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
