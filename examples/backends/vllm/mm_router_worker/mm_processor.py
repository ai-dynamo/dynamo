# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing utilities for vLLM MM Router Worker.

Key differences from TRT-LLM version:
- Image loading: PIL header-only (no full decode) + requests/base64
- mm_hash: blake3 of raw file bytes (matches vLLM multi_modal_uuids)
- Token replacement: NOT needed — vLLM keeps the original image_token_id as-is
- Fast path token expansion: compute token count from image dimensions directly
"""

import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image

from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ProcessedInput:
    """Processed multimodal input."""

    tokens: list[int]
    mm_hashes: list[int] | None
    image_ranges: list[tuple[int, int]] | None  # [(start, end), ...] per image


# =============================================================================
# Public functions
# =============================================================================


def extract_image_urls(messages: list[dict]) -> list[str]:
    """Extract image URLs from OpenAI-format messages."""
    urls = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url")
                    if url:
                        urls.append(url)
    return urls


def process_multimodal(
    messages: list[dict],
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
    model: str,
) -> tuple[ProcessedInput, list[bytes]]:
    """
    Process multimodal request: load images, get expanded tokens and mm_hashes.

    Returns (ProcessedInput, raw_bytes_list) — raw_bytes_list is used by the
    caller for HTTP→dataURI conversion (#5).

    Uses lightweight image loading (header-only PIL) and fast path token expansion.
    Unlike TRT-LLM, vLLM keeps original image_token_id (no replacement).
    """
    t0 = time.perf_counter()

    # The preprocessed request does not carry a rendered template string; it carries
    # original messages in extra_args, so we must apply chat template again here.
    prompt = _build_prompt_with_images(messages, tokenizer, processor)
    t1 = time.perf_counter()

    # Lightweight image loading — only reads header, no full pixel decode
    image_data = [load_image_for_routing(url) for url in image_urls]
    t2 = time.perf_counter()
    raw_bytes_list = [d[0] for d in image_data]
    image_dims = [(d[1], d[2]) for d in image_data]

    # Fast path token expansion from dimensions
    tokens, image_ranges = _get_expanded_tokens_fast(
        prompt, image_dims, tokenizer, processor
    )
    t3 = time.perf_counter()

    # Hash raw file bytes directly (blake3)
    mm_uuids = compute_mm_uuids_from_images(raw_bytes_list)
    t4 = time.perf_counter()
    mm_hashes = [int(uuid[:16], 16) for uuid in mm_uuids]

    logger.info(
        f"[perf][process_mm] template={1000*(t1-t0):.2f}ms "
        f"load_images={1000*(t2-t1):.2f}ms "
        f"expand_tokens={1000*(t3-t2):.2f}ms "
        f"hash={1000*(t4-t3):.2f}ms "
        f"total={1000*(t4-t0):.2f}ms "
        f"n_images={len(image_urls)}"
    )

    return ProcessedInput(tokens=tokens, mm_hashes=mm_hashes, image_ranges=image_ranges), raw_bytes_list


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_ranges: list[tuple[int, int]] | None,
) -> list[dict | None] | None:
    """
    Build per-block mm_info for routing.

    For each block, check which images overlap with it and add their mm_hash.

    Assumption: mm_hashes and image_ranges are in the same order as images appear
    in the request (which matches their order in the token sequence).
    """
    if not mm_hashes or not image_ranges or len(mm_hashes) != len(image_ranges):
        return None

    num_blocks = (num_tokens + block_size - 1) // block_size
    result = []

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Find images overlapping this block
        mm_objects = [
            {"mm_hash": mm_hash, "offsets": []}
            for mm_hash, (img_start, img_end) in zip(mm_hashes, image_ranges)
            # FIXME: Revisit the bounds checks here
            # https://github.com/ai-dynamo/dynamo/issues/6588
            if block_end > img_start and block_start <= img_end
        ]

        result.append({"mm_objects": mm_objects} if mm_objects else None)

    return result


# =============================================================================
# Internal functions
# =============================================================================


def _build_prompt_with_images(
    messages: list[dict], tokenizer: Any, processor: Any
) -> str:
    """
    Build a prompt that includes image placeholders using the tokenizer's
    chat template. This is critical for Qwen2-VL/Qwen2.5-VL models which
    need <|vision_start|><|image_pad|>...<|vision_end|> in the prompt for
    the processor to expand image tokens correctly.

    Raises if chat template cannot be applied. For MM routing correctness, we do
    not silently fall back to text-only prompts.
    """
    # Try processor first (has the best chat template for multimodal)
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fall back to tokenizer if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    raise ValueError("Neither processor nor tokenizer provides apply_chat_template")


_light_cache: dict[str, tuple[bytes, int, int]] = {}  # url -> (raw_bytes, w, h)


def load_image_for_routing(url: str) -> tuple[bytes, int, int]:
    """
    Router-only lightweight image loading: download and get (raw_bytes, width, height).

    Uses PIL lazy open (header-only, no full pixel decode) for dimensions.
    Caches HTTP URLs to avoid redundant downloads.
    """
    if url in _light_cache:
        return _light_cache[url]

    parsed = urlparse(url)

    if parsed.scheme == "data":
        # data:image/png;base64,<data>
        _, data = parsed.path.split(",", 1)
        raw_bytes = base64.b64decode(data)
    elif parsed.scheme in ("http", "https"):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw_bytes = response.content
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    # PIL lazy open — only reads header, no full pixel decode
    img = Image.open(BytesIO(raw_bytes))
    w, h = img.size

    # Cache HTTP URLs to avoid redundant downloads
    if parsed.scheme in ("http", "https"):
        _light_cache[url] = (raw_bytes, w, h)

    return raw_bytes, w, h


def _compute_tokens_per_image_fast(
    image_dims: list[tuple[int, int]], processor: Any
) -> list[int]:
    """Compute per-image token counts from image dimensions without full preprocessing."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor.image_processor not found")

    get_num_patches = getattr(image_processor, "get_number_of_image_patches", None)
    if not callable(get_num_patches):
        raise NotImplementedError(
            "image_processor.get_number_of_image_patches not available"
        )

    merge_size = getattr(image_processor, "merge_size", None)
    if merge_size is None:
        raise ValueError("image_processor.merge_size not found")

    tokens_per_image = []
    images_kwargs: dict[str, Any] = {}
    for width, height in image_dims:
        num_patches = int(get_num_patches(height, width, images_kwargs))
        tokens_per_image.append(num_patches // (merge_size**2))
    return tokens_per_image


def _get_expanded_tokens_fast(
    prompt: str,
    image_dims: list[tuple[int, int]],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """
    Fast path token expansion using image dimensions instead of full HF processor.

    Computes token counts from image dimensions directly, avoiding the expensive
    processor(..., images=pil_images) call (55ms→0.23ms).
    """
    if processor is None:
        return tokenizer.encode(prompt), None

    try:
        # 1. Text-only tokenize to get base tokens
        base_tokens = tokenizer.encode(prompt)

        # 2. Compute tokens_per_image from dimensions via processor API
        tokens_per_image = _compute_tokens_per_image_fast(image_dims, processor)

        # 3. Find placeholder token and ranges
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor.image_token_id not found")

        # Find single-token placeholders in base_tokens
        placeholder_indices = [
            i for i, t in enumerate(base_tokens) if t == image_token_id
        ]

        if len(placeholder_indices) != len(image_dims):
            logger.warning(
                f"Placeholder count ({len(placeholder_indices)}) != image count "
                f"({len(image_dims)}), falling back to contiguous range expansion"
            )
            # Fallback: try contiguous range approach
            contiguous_ranges = _find_image_token_ranges(base_tokens, image_token_id)
            return base_tokens, _compute_per_image_ranges(
                contiguous_ranges, tokens_per_image
            )

        # 4. Expand: replace each single placeholder with N tokens
        expanded_tokens: list[int] = []
        image_ranges: list[tuple[int, int]] = []
        prev_end = 0

        for img_idx, placeholder_pos in enumerate(placeholder_indices):
            # Copy text tokens before this placeholder
            expanded_tokens.extend(base_tokens[prev_end:placeholder_pos])
            # Insert expanded image tokens
            img_start = len(expanded_tokens)
            n_tokens = tokens_per_image[img_idx]
            expanded_tokens.extend([image_token_id] * n_tokens)
            image_ranges.append((img_start, img_start + n_tokens))
            prev_end = placeholder_pos + 1  # skip the single placeholder

        # Append remaining tokens after last placeholder
        expanded_tokens.extend(base_tokens[prev_end:])

        return expanded_tokens, image_ranges

    except Exception as e:
        logger.warning(f"Fast path token expansion failed: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


def _find_image_token_ranges(
    tokens: list[int], image_token_id: int
) -> list[tuple[int, int]]:
    """
    Find all contiguous ranges of image tokens.

    Unlike the TRT-LLM version, this does NOT replace tokens — vLLM keeps
    the original image_token_id as-is in KV events.

    Returns: list of (start, end) ranges for contiguous image token regions.
    """
    ranges = []
    start = None

    for i, t in enumerate(tokens):
        if t == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i))
            start = None

    if start is not None:
        ranges.append((start, len(tokens)))

    if ranges:
        logger.info(
            f"Found {sum(e - s for s, e in ranges)} image tokens "
            f"(id={image_token_id}) in {len(ranges)} range(s)"
        )

    return ranges


def _compute_per_image_ranges(
    contiguous_ranges: list[tuple[int, int]],
    tokens_per_image: list[int],
) -> list[tuple[int, int]] | None:
    """
    Split contiguous image token ranges by each image's token count.

    Example: contiguous_ranges=[(0, 100)], tokens_per_image=[60, 40]
    Returns: [(0, 60), (60, 100)]  # image 1 at 0-60, image 2 at 60-100
    """
    if not contiguous_ranges:
        if tokens_per_image:
            logger.warning(
                f"No image tokens found but {len(tokens_per_image)} images expected"
            )
        return None

    # Greedily assign images to ranges in order
    result = []
    image_idx = 0

    for range_start, range_end in contiguous_ranges:
        range_size = range_end - range_start
        pos = range_start
        consumed = 0

        # Consume images that fit entirely in this range
        # (a single image's tokens are always contiguous, cannot span ranges)
        while image_idx < len(tokens_per_image):
            needed = tokens_per_image[image_idx]
            if consumed + needed <= range_size:
                result.append((pos, pos + needed))
                pos += needed
                consumed += needed
                image_idx += 1
            else:
                break

        # Range must be exactly filled (no leftover image tokens)
        if consumed != range_size:
            logger.warning(
                f"Range size mismatch: consumed {consumed} != range {range_size}"
            )
            return None

    # All images must be consumed
    if image_idx != len(tokens_per_image):
        logger.warning(f"Not all images mapped: {image_idx} < {len(tokens_per_image)}")
        return None

    return result
