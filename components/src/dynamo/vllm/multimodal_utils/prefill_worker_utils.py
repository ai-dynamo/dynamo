# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
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
from dynamo.common.multimodal.image_loader import (
    DECODED_VARIANT_KEY,
    URL_VARIANT_KEY,
    decoded_content_hash_key,
)
from dynamo.common.utils.time_section import time_and_log_code_section
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


def parse_image_item(item: Any) -> tuple[str | None, Dict[str, Any] | None]:
    """Split one image wire item into ``(url, decoded_metadata)``.

    Accepts a bare URL string, ``{"Url": ...}``, or ``{"Decoded": {...}}``.
    Exactly one element of the returned tuple is non-``None``.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        variants = [
            key for key in (URL_VARIANT_KEY, DECODED_VARIANT_KEY) if key in item
        ]
        if len(variants) == 1:
            if variants[0] == URL_VARIANT_KEY and isinstance(
                item[URL_VARIANT_KEY], str
            ):
                return item[URL_VARIANT_KEY], None
            if variants[0] == DECODED_VARIANT_KEY and isinstance(
                item[DECODED_VARIANT_KEY], dict
            ):
                return None, item[DECODED_VARIANT_KEY]
    raise ValueError(f"Unsupported image item: {item!r}")


def _image_item_cache_key(item: Any) -> str | None:
    """Embedding-cache key for one image wire item, or ``None`` if unkeyed.

    URL items hash the URL; frontend-decoded items reuse the canonical
    content hash serialized by the Rust media decoder. Decoded items with a
    missing/malformed hash return ``None`` and simply bypass the cache.
    """
    url, decoded = parse_image_item(item)
    if url is not None:
        return get_embedding_hash(url)
    return decoded_content_hash_key(decoded)


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

    if "image" not in multi_modal_data:
        multi_modal_data["image"] = mm_data["image"]
        return

    if isinstance(mm_data["image"], dict):
        # Qwen-VL style: dict with image_embeds + image_grid_thw tensors
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
    elif isinstance(mm_data["image"], torch.Tensor):
        multi_modal_data["image"] = torch.cat(
            (multi_modal_data["image"], mm_data["image"])
        )
    else:
        raise ValueError(
            f"Unexpected image data format from construct_mm_data: {type(mm_data['image'])}"
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
    image_items: List[Any],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    context=None,
) -> tuple[List[MultiModalGroup], _PendingRelease | None]:
    """Fan out image items (URL or frontend-decoded) to encode workers, load
    embeddings, and return ready groups.

    For NIXL receivers the returned embeddings are zero-copy views into
    pre-allocated buffers.  The returned ``_PendingRelease`` must be
    released after the tensors have been consumed.
    """
    encode_worker_count = len(encode_worker_client.instance_ids())
    if encode_worker_count == 0:
        raise RuntimeError("No encode workers available to process multimodal input")

    encode_batch_size = (
        max(1, len(image_items) // encode_worker_count)
        if SPLIT_ENCODE
        else len(image_items)
    )

    encode_request = vLLMMultimodalRequest(
        engine_prompt=PatchedTokensPrompt(prompt_token_ids=[]),
        sampling_params=VllmSamplingParams(),
        request_id=request_id,
        multimodal_inputs=[],
    )

    with time_and_log_code_section(f"[PREFILL] request: {request_id} dispatch encode"):
        batch: List[MultiModalGroup] = []
        encode_response_streams = []
        for item in image_items:
            url, decoded = parse_image_item(item)
            multimodal_input = MultiModalInput()
            multimodal_input.image_url = url
            multimodal_input.image_decoded = decoded
            batch.append(MultiModalGroup(multimodal_input=multimodal_input))

            if len(batch) >= encode_batch_size:
                encode_request.multimodal_inputs = batch
                payload = encode_request.model_dump_json()
                encode_response_streams.append(
                    await encode_worker_client.round_robin(payload, context=context)  # type: ignore[arg-type]
                )
                batch = []

        if batch:
            encode_request.multimodal_inputs = batch
            payload = encode_request.model_dump_json()
            encode_response_streams.append(
                await encode_worker_client.round_robin(payload, context=context)  # type: ignore[arg-type]
            )

    with time_and_log_code_section(
        f"[PREFILL] request: {request_id} receive encode responses"
    ):
        multimodal_groups: List[MultiModalGroup] = []
        for stream in encode_response_streams:
            async for response in stream:
                output = vLLMMultimodalRequest.model_validate_json(response.data())  # type: ignore[attr-defined]
                if output.multimodal_inputs:
                    multimodal_groups.extend(output.multimodal_inputs)

    with time_and_log_code_section(
        f"[PREFILL] request: {request_id} receive embeddings"
    ):
        tasks = [
            asyncio.create_task(receiver.receive_embeddings(group.serialized_request))
            for group in multimodal_groups
            if group.serialized_request is not None
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
    image_items: list[Any],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    cache: MultimodalEmbeddingCacheManager | None = None,
    context=None,
) -> tuple[list[MultiModalGroup], _PendingRelease | None]:
    """Fetch multimodal embeddings with transparent cache-through.

    Pipeline: check_cache → fetch misses from encode workers → update_cache.
    When *cache* is ``None`` the cache steps are no-ops and all items go
    straight to the encode workers. Items without a cache key (e.g. a
    frontend-decoded descriptor missing its content hash) are fetched and
    not cached.

    For NIXL receivers the returned embeddings are zero-copy views.  The
    returned ``_PendingRelease`` must be released after consuming the
    tensors.
    """
    results: list[MultiModalGroup | None] = [None] * len(image_items)
    to_fetch: list[tuple[int, Any, str | None]] = []

    # ── 1. Check cache (no-op when cache is None) ────────────────────
    for idx, item in enumerate(image_items):
        if cache is not None:
            key = _image_item_cache_key(item)
            cached = cache.get(key) if key is not None else None
            if cached is not None:
                logger.debug(f"[{request_id}] Cache hit for image index {idx}")
                results[idx] = MultiModalGroup(
                    loaded_embedding=cached.tensor,
                    image_grid_thw=cached.image_grid_thw,
                )
                continue
        else:
            key = None
        to_fetch.append((idx, item, key))

    # ── 2. Fetch uncached from encode workers ────────────────────────
    pending: _PendingRelease | None = None
    if to_fetch:
        miss_items = [item for _, item, _ in to_fetch]
        groups, pending = await _fetch_from_encode_workers(
            encode_worker_client,
            miss_items,
            request_id,
            receiver,
            context=context,
        )

        # ── 3. Update cache (no-op when cache is None) ──────────────

        for (idx, _item, key), group in zip(to_fetch, groups, strict=True):
            if cache is not None and key is not None:
                assert group.loaded_embedding is not None
                cache.set(
                    key,
                    CachedEmbedding(
                        tensor=group.loaded_embedding.clone(),
                        image_grid_thw=group.image_grid_thw,
                    ),
                )
            results[idx] = group

    return [r for r in results if r is not None], pending


# ── Public API (single entry point) ─────────────────────────────────


class MultiModalEmbeddingLoader:
    """Helper class for requesting remote encode and receive embeddings."""

    def __init__(
        self,
        encode_worker_client: Client,
        receiver: AbstractEmbeddingReceiver,
        embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None,
    ):
        self._encode_worker_client = encode_worker_client
        self._receiver = receiver
        self._embedding_cache_manager = embedding_cache_manager

    async def load_multimodal_embeddings(
        self,
        image_items: list[Any],
        request_id: str,
        *,
        model: str,
        context=None,
    ) -> Dict[str, Any]:
        """Fetch embeddings and build engine-ready ``multi_modal_data``.

        ``image_items`` are wire items: bare URL strings, ``{"Url": ...}``, or
        frontend-decoded ``{"Decoded": {...}}`` descriptors.

        Full pipeline:
        cache check → remote fetch → cache update → accumulate → release NIXL buffers.

        Returns a dict suitable for passing to ``TokensPrompt(multi_modal_data=...)``.
        """
        if self._encode_worker_client is None or not image_items:
            return {}

        groups, pending = await _fetch_embeddings(
            self._encode_worker_client,
            image_items,
            request_id,
            self._receiver,
            cache=self._embedding_cache_manager,
            context=context,
        )

        multi_modal_data: Dict[str, Any] = {}
        with time_and_log_code_section(
            f"[PREFILL] request: {request_id} accumulate embeddings"
        ):
            for group in groups:
                assert group.loaded_embedding is not None
                _accumulate_embeddings(
                    multi_modal_data,
                    model,
                    group.loaded_embedding.dtype,
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
