# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import torch
from vllm.sampling_params import SamplingParams as VllmSamplingParams

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    LocalEmbeddingReceiver,
)
from dynamo.runtime import Client

from .encode_utils import get_embedding_hash
from .model import construct_mm_data
from .protocol import (
    MultiModalGroup,
    MultiModalInput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

SPLIT_ENCODE = int(os.getenv("DYN_SPLIT_ENCODE", 1))


# ── Internal helpers (all underscore-prefixed) ───────────────────────


class _PendingRelease:
    """Tracks NIXL tensor buffers that should be released after consumption.

    For NIXL receivers, embeddings are views into pre-allocated reusable
    buffers.  Instead of cloning each embedding eagerly, we defer the
    release until the caller has consumed the tensors (e.g. via
    ``_accumulate_embeddings`` which copies data through ``torch.cat``).
    """

    __slots__ = ("_receiver", "_tensor_ids")

    def __init__(self, receiver: AbstractEmbeddingReceiver):
        self._receiver = receiver
        self._tensor_ids: List[int] = []

    def track(self, tensor_id: int) -> None:
        self._tensor_ids.append(tensor_id)

    def release_all(self) -> None:
        for tid in self._tensor_ids:
            self._receiver.release_tensor(tid)
        self._tensor_ids.clear()


def _accumulate_embeddings(
    multi_modal_data: Dict[str, Any],
    model: str,
    embeddings_dtype: torch.dtype,
    embeddings: torch.Tensor,
    image_grid_thw,
) -> None:
    """Construct model-specific mm_data from embeddings and merge into the
    accumulated ``multi_modal_data`` dict (mutated in-place).

    Handles both video (numpy conversion) and image modalities, including
    the Qwen-VL dict-style embeddings with ``image_embeds`` + ``image_grid_thw``.
    """
    if "video" in model.lower():
        video_numpy = embeddings.numpy()
        mm_data = construct_mm_data(
            model,
            embeddings_dtype,
            video_numpy=video_numpy,
        )
        multi_modal_data["video"].append(mm_data["video"])
        return

    mm_data = construct_mm_data(
        model,
        embeddings_dtype,
        image_embeds=embeddings,
        image_grid_thw=image_grid_thw,
    )

    if isinstance(mm_data["image"], dict):
        # Qwen-VL style: dict with image_embeds + image_grid_thw tensors
        if multi_modal_data["image"] == []:
            multi_modal_data["image"] = mm_data["image"]
        else:
            # [gluo FIXME] need to understand how Qwen consumes multi-image embeddings
            multi_modal_data["image"]["image_embeds"] = torch.cat(
                (
                    multi_modal_data["image"]["image_embeds"],
                    mm_data["image"]["image_embeds"],
                )
            )
            multi_modal_data["image"]["image_grid_thw"] = torch.cat(
                (
                    multi_modal_data["image"]["image_grid_thw"],
                    mm_data["image"]["image_grid_thw"],
                )
            )
    else:
        # Plain tensor embeddings
        logger.info(f"Get embedding of shape {mm_data['image'].shape}")
        # [gluo FIXME] embedding with multiple images?
        if multi_modal_data["image"] == []:
            multi_modal_data["image"] = mm_data["image"]
        else:
            multi_modal_data["image"] = torch.cat(
                (multi_modal_data["image"], mm_data["image"])
            )


def _ensure_owned_tensors(multi_modal_data: Dict[str, Any]) -> None:
    """Clone tensor views so NIXL buffers can be safely released.

    Only needed for single-image; multi-image goes through torch.cat
    which already produces owned tensors.
    """
    img = multi_modal_data.get("image")
    if isinstance(img, dict):
        for k, v in img.items():
            if isinstance(v, torch.Tensor):
                img[k] = v.clone()
    elif isinstance(img, torch.Tensor):
        multi_modal_data["image"] = img.clone()


async def _fetch_from_encode_workers(
    encode_worker_client: Client,
    image_urls: List[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
) -> tuple[List[MultiModalGroup], _PendingRelease | None]:
    """Fan out image URLs to encode workers, load embeddings, and return ready groups.

    For NIXL receivers the returned embeddings are zero-copy views into
    pre-allocated buffers.  The returned ``_PendingRelease`` must be
    released after the tensors have been consumed.
    """
    encode_worker_count = len(encode_worker_client.instance_ids())
    if encode_worker_count == 0:
        raise RuntimeError("No encode workers available to process multimodal input")

    encode_batch_size = (
        max(1, len(image_urls) // encode_worker_count)
        if SPLIT_ENCODE
        else len(image_urls)
    )

    encode_request = vLLMMultimodalRequest(
        engine_prompt=PatchedTokensPrompt(prompt_token_ids=[]),
        sampling_params=VllmSamplingParams(),
        request_id=request_id,
        multimodal_inputs=[],
    )

    batch: List[MultiModalGroup] = []
    encode_response_streams = []
    for url in image_urls:
        multimodal_input = MultiModalInput()
        multimodal_input.image_url = url
        batch.append(MultiModalGroup(multimodal_input=multimodal_input))

        if len(batch) >= encode_batch_size:
            encode_request.multimodal_inputs = batch
            payload = encode_request.model_dump_json()
            encode_response_streams.append(
                await encode_worker_client.round_robin(payload)  # type: ignore[arg-type]
            )
            batch = []

    if batch:
        encode_request.multimodal_inputs = batch
        payload = encode_request.model_dump_json()
        encode_response_streams.append(
            await encode_worker_client.round_robin(payload)  # type: ignore[arg-type]
        )

    multimodal_groups: List[MultiModalGroup] = []
    for stream in encode_response_streams:
        async for response in stream:
            logger.debug(f"Received response from encode worker: {response}")
            output = vLLMMultimodalRequest.model_validate_json(response.data())  # type: ignore[attr-defined]
            if output.multimodal_inputs:
                multimodal_groups.extend(output.multimodal_inputs)

    tasks = [
        asyncio.create_task(receiver.receive_embeddings(group.serialized_request))
        for group in multimodal_groups
    ]
    loaded = await asyncio.gather(*tasks)

    is_local = isinstance(receiver, LocalEmbeddingReceiver)
    pending: _PendingRelease | None = None if is_local else _PendingRelease(receiver)
    for group, (tensor_id, embedding) in zip(multimodal_groups, loaded, strict=True):
        group.loaded_embedding = embedding
        if pending is not None:
            pending.track(tensor_id)

    return multimodal_groups, pending


async def _fetch_embeddings(
    encode_worker_client: Client,
    image_urls: list[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    cache: MultimodalEmbeddingCacheManager | None = None,
) -> tuple[list[MultiModalGroup], _PendingRelease | None]:
    """Fetch multimodal embeddings with transparent cache-through.

    Pipeline: check_cache → fetch misses from encode workers → update_cache.
    When *cache* is ``None`` the cache steps are no-ops and all URLs go
    straight to the encode workers.

    For NIXL receivers the returned embeddings are zero-copy views.  The
    returned ``_PendingRelease`` must be released after consuming the
    tensors.
    """
    results: list[MultiModalGroup | None] = [None] * len(image_urls)
    to_fetch: list[tuple[int, str, str | None]] = []

    # ── 1. Check cache (no-op when cache is None) ────────────────────
    for idx, url in enumerate(image_urls):
        if cache is not None:
            key = get_embedding_hash(url)
            cached = cache.get(key)
            if cached is not None:
                logger.debug(f"[{request_id}] Cache hit for URL index {idx}")
                results[idx] = MultiModalGroup(
                    loaded_embedding=cached.tensor,
                    image_grid_thw=cached.image_grid_thw,
                )
                continue
        else:
            key = None
        to_fetch.append((idx, url, key))

    # ── 2. Fetch uncached from encode workers ────────────────────────
    pending: _PendingRelease | None = None
    if to_fetch:
        if cache is not None:
            logger.info(
                f"[{request_id}] Cache miss for {len(to_fetch)}/{len(image_urls)} URLs, "
                "fetching from encode workers"
            )
        miss_urls = [url for _, url, _ in to_fetch]
        groups, pending = await _fetch_from_encode_workers(
            encode_worker_client,
            miss_urls,
            request_id,
            receiver,
        )

        # ── 3. Update cache (no-op when cache is None) ──────────────
        for (idx, _url, key), group in zip(to_fetch, groups, strict=True):
            if cache is not None and key is not None:
                cache.set(
                    key,
                    CachedEmbedding(
                        tensor=group.loaded_embedding.clone(),
                        image_grid_thw=group.image_grid_thw,
                    ),
                )
            results[idx] = group
    else:
        logger.info(f"[{request_id}] All {len(image_urls)} URLs served from cache")

    return [r for r in results if r is not None], pending


# ── Public API (single entry point) ─────────────────────────────────


async def load_multimodal_embeddings(
    encode_worker_client: Client,
    image_urls: list[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    *,
    model: str,
    embeddings_dtype: torch.dtype,
    cache: MultimodalEmbeddingCacheManager | None = None,
) -> Dict[str, Any]:
    """Fetch embeddings and build engine-ready ``multi_modal_data``.

    Full pipeline:
      cache check → remote fetch → cache update → accumulate → release NIXL buffers.

    Returns a dict suitable for passing to ``TokensPrompt(multi_modal_data=...)``.
    """
    groups, pending = await _fetch_embeddings(
        encode_worker_client,
        image_urls,
        request_id,
        receiver,
        cache=cache,
    )

    multi_modal_data: Dict[str, Any] = defaultdict(list)
    for group in groups:
        _accumulate_embeddings(
            multi_modal_data,
            model,
            embeddings_dtype,
            group.loaded_embedding,
            group.image_grid_thw,
        )

    if pending is not None:
        # Multi-image: torch.cat in _accumulate_embeddings already created
        # owned tensors.  Single-image: the data is still a view into the
        # NIXL buffer, so we must clone before releasing.
        if len(groups) == 1:
            _ensure_owned_tensors(multi_modal_data)
        pending.release_all()

    return multi_modal_data
