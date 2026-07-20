# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal media extraction shared by the disaggregated prefill and decode
workers, so both feed identical media URLs to the engine and reproduce the same
token layout the transferred KV depends on.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"
AUDIO_URL_KEY = "audio_url"
VIDEO_URL_KEY = "video_url"
_SUPPORTED_MULTIMODAL_CONTENT_TYPES = frozenset(
    {IMAGE_URL_KEY, AUDIO_URL_KEY, VIDEO_URL_KEY}
)
# BaseMultiModalProcessorOutput.organize_results() builds SGLang's mm_items in
# this order, independent of their order in the original prompt.
_SGLANG_MM_ITEM_MODALITY_ORDER = ("image", "video", "audio")


def _multi_modal_data(request: Dict[str, Any]) -> Dict[str, Any]:
    if "multi_modal_data" not in request or request.get("multi_modal_data") is None:
        return {}

    mm_data = request["multi_modal_data"]
    if not isinstance(mm_data, dict):
        raise ValueError(
            f"multi_modal_data must be an object, got {type(mm_data).__name__}"
        )
    return mm_data


def _raw_multimodal_content_types(request: Dict[str, Any]) -> set[str]:
    extra_args = request.get("extra_args") or {}
    messages = extra_args.get("messages") or request.get("messages") or []
    content_types: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in _SUPPORTED_MULTIMODAL_CONTENT_TYPES
            ):
                content_types.add(part["type"])
    return content_types


def raise_if_unextracted_multimodal(request: Dict[str, Any]) -> None:
    """Reject raw multimodal messages that were not extracted by the frontend."""

    mm_data = _multi_modal_data(request)
    raw_types = _raw_multimodal_content_types(request)
    if not (mm_data or raw_types):
        return

    missing_mm_types = {
        content_type for content_type in raw_types if not mm_data.get(content_type)
    }
    if not missing_mm_types:
        return

    types_str = ", ".join(sorted(missing_mm_types))
    message = (
        "Multimodal input received but SGLang worker did not receive "
        f"corresponding multi_modal_data for: {types_str}. Ensure the "
        "frontend processor extracted image_url/audio_url/video_url content or "
        "remove the corresponding multimodal content."
    )
    logger.error(message)
    raise RuntimeError(message)


def extract_media_urls(
    mm_data: Optional[Dict[str, Any]], media_key: str
) -> list[str] | None:
    """Return the URLs under ``media_key`` (``{"Url": ...}`` or string items), or
    ``None`` if absent. Raises on malformed or frontend-decoded payloads rather
    than silently degrading a multimodal request to text.
    """
    if not mm_data:
        return None

    items = mm_data.get(media_key)
    if items is None:
        return None
    if not isinstance(items, list):
        raise ValueError(
            f"{media_key} must be a list of URL items, got {type(items).__name__}"
        )
    if not items:
        return None

    urls: list[str] = []
    for item in items:
        if isinstance(item, str):
            urls.append(item)
            continue
        if isinstance(item, dict):
            variants = [key for key in ("Url", "Decoded") if key in item]
            if len(variants) != 1:
                raise ValueError(f"Unsupported {media_key} item: {item!r}")
            if variants[0] == "Url" and isinstance(item["Url"], str):
                urls.append(item["Url"])
                continue
            if variants[0] == "Decoded":
                raise ValueError(
                    f"Frontend-decoded media is not supported for disaggregated "
                    f"{media_key}; use URL-based inputs."
                )
        raise ValueError(f"Unsupported {media_key} item: {item!r}")

    return urls or None


def extract_mm_hashes(request: Dict[str, Any]) -> list[str] | None:
    """Return frontend MM hashes in SGLang's modality-grouped item order."""
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None

    grouped = extra_args.get("mm_hashes_by_modality")
    if grouped is not None:
        if not isinstance(grouped, dict):
            logger.warning(
                "extra_args.mm_hashes_by_modality is not an object; "
                "ignoring routing-side hashes and letting SGLang recompute"
            )
            return None

        unknown_modalities = {
            str(modality)
            for modality, hashes in grouped.items()
            if modality not in _SGLANG_MM_ITEM_MODALITY_ORDER and hashes
        }
        if unknown_modalities:
            logger.warning(
                "extra_args.mm_hashes_by_modality contains unsupported "
                "modalities %s; ignoring routing-side hashes and letting "
                "SGLang recompute",
                sorted(unknown_modalities),
            )
            return None

        flattened: list[str] = []
        for modality in _SGLANG_MM_ITEM_MODALITY_ORDER:
            hashes = grouped.get(modality)
            if hashes is None:
                continue
            if not isinstance(hashes, list) or not all(
                isinstance(value, str) for value in hashes
            ):
                logger.warning(
                    "extra_args.mm_hashes_by_modality[%s] is not a string "
                    "list; ignoring routing-side hashes and letting SGLang recompute",
                    modality,
                )
                return None
            flattened.extend(hashes)
        return flattened or None

    mm_hashes = extra_args.get("mm_hashes")
    if not mm_hashes or not isinstance(mm_hashes, list):
        return None
    if not all(isinstance(value, str) for value in mm_hashes):
        logger.warning(
            "extra_args.mm_hashes contained non-str entries; ignoring "
            "routing-side hashes and letting SGLang recompute"
        )
        return None
    return mm_hashes


def build_disagg_mm_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
    """Build media kwargs for a disaggregated worker's ``async_generate`` call.

    All keys are always present (``None`` when absent).
    """
    mm_data = _multi_modal_data(request)
    image_data = extract_media_urls(mm_data, IMAGE_URL_KEY)
    audio_data = extract_media_urls(mm_data, AUDIO_URL_KEY)
    video_data = extract_media_urls(mm_data, VIDEO_URL_KEY)
    # TODO: Native EP/D works with this raw-media path, but both prefill and
    # decode call it, so SGLang may fetch/load/preprocess the same media twice.
    # Remove the duplicate preprocessing once native EP/D can share processed
    # media or embeddings while preserving decode-side token layout.
    if image_data or audio_data or video_data:
        logger.debug(
            "disaggregated multimodal request: images=%d, audio=%d, videos=%d",
            len(image_data or []),
            len(audio_data or []),
            len(video_data or []),
        )
    return {
        "image_data": image_data,
        "audio_data": audio_data,
        "video_data": video_data,
    }
