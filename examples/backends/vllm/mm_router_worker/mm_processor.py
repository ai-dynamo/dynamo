# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing utilities for vLLM MM Router Worker.

Key differences from TRT-LLM version:
- Image loading: PIL + requests/base64 (no TRT-LLM dependency)
- mm_hash: SHA256 of normalized PNG bytes (matches vLLM multi_modal_uuids)
- Token replacement: NOT needed — vLLM keeps the original image_token_id as-is
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ProcessedInput:
    """Processed multimodal input."""

    tokens: list[int]
    mm_hashes: list[int] | None
    image_ranges: list[tuple[int, int]] | None  # [(start, end), ...] per image
    raw_bytes_list: list[bytes] | None = None  # raw image bytes for data URI conversion


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


async def process_multimodal(
    messages: list[dict],
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
    model: str,
    image_loader: ImageLoader | None = None,
) -> ProcessedInput:
    """
    Process multimodal request: load images, get expanded tokens and mm_hashes.

    Uses async ImageLoader for non-blocking image download with FIFO cache.
    Unlike TRT-LLM, vLLM keeps original image_token_id (no replacement).
    """
    total_start = time.perf_counter()

    # The preprocessed request does not carry a rendered template string; it carries
    # original messages in extra_args, so we must apply chat template again here.
    prompt_start = time.perf_counter()
    prompt = _build_prompt_with_images(messages, tokenizer, processor)
    prompt_ms = _elapsed_ms(prompt_start)
    logger.info(f"Prompt (first 300 chars): {prompt[:300]}")

    # Load images: only need raw bytes + dimensions (skip convert("RGB"))
    load_start = time.perf_counter()
    if image_loader is not None:
        results = await asyncio.gather(
            *[image_loader.load_image_light(url) for url in image_urls]
        )
        raw_bytes_list = [r[0] for r in results]
        image_dims = [(r[1], r[2]) for r in results]  # (width, height)
    else:
        pil_images = [_load_image(url) for url in image_urls]
        raw_bytes_list = None
        image_dims = [(img.width, img.height) for img in pil_images]
    load_ms = _elapsed_ms(load_start)

    # Get expanded tokens and image ranges (no token replacement for vLLM)
    expand_start = time.perf_counter()
    tokens, image_ranges = _get_expanded_tokens(
        prompt, image_dims, tokenizer, processor
    )
    expand_ms = _elapsed_ms(expand_start)
    logger.info(f"Expanded: {len(tokens)} tokens, " f"image_ranges={image_ranges}")

    # Compute mm_hashes exactly like vLLM handler's multi_modal_uuids path.
    hash_start = time.perf_counter()
    mm_uuids = compute_mm_uuids_from_images(raw_bytes_list)
    mm_hashes = [int(uuid[:16], 16) for uuid in mm_uuids]
    hash_ms = _elapsed_ms(hash_start)

    logger.info(f"mm_hashes={mm_hashes}")

    total_ms = _elapsed_ms(total_start)
    logger.debug(
        "[perf][mm_router] process_multimodal: "
        "total=%.3fms prompt=%.3fms load_images=%.3fms "
        "expanded_tokens=%.3fms mm_hash=%.3fms "
        "images=%d tokens=%d",
        total_ms, prompt_ms, load_ms, expand_ms, hash_ms,
        len(image_urls), len(tokens),
    )

    return ProcessedInput(tokens=tokens, mm_hashes=mm_hashes, image_ranges=image_ranges, raw_bytes_list=raw_bytes_list)


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


def _load_image(url: str) -> Image.Image:
    """
    Load an image from URL (http/https or data URI) and return a PIL RGB image.
    """
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

    return Image.open(BytesIO(raw_bytes)).convert("RGB")


def _get_expanded_tokens(
    prompt: str,
    image_dims: list[tuple[int, int]],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """
    Get tokens with visual expansion and find each image's token range.

    Args:
        image_dims: list of (width, height) tuples per image.

    Uses fast path: compute token count from image dimensions, skip full
    HF processor pixel preprocessing.
    """
    if processor is None:
        return tokenizer.encode(prompt), None

    total_start = time.perf_counter()

    try:
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor.image_token_id not found")

        tpi_start = time.perf_counter()
        tokens_per_image = _compute_tokens_per_image_fast(image_dims, processor)
        tpi_ms = _elapsed_ms(tpi_start)

        tok_start = time.perf_counter()
        text_tokens = _tokenize_prompt_text_only(prompt, tokenizer, processor)
        tok_ms = _elapsed_ms(tok_start)

        exp_start = time.perf_counter()
        expanded_tokens, image_ranges = _expand_image_tokens(
            text_tokens, image_token_id, tokens_per_image
        )
        exp_ms = _elapsed_ms(exp_start)

        logger.debug(
            "[perf][mm_router] _get_expanded_tokens: mode=fast "
            "total=%.3fms tpi=%.3fms tokenize=%.3fms expand=%.3fms "
            "images=%d tokens=%d",
            _elapsed_ms(total_start), tpi_ms, tok_ms, exp_ms,
            len(image_dims), len(expanded_tokens),
        )
        return expanded_tokens, image_ranges

    except Exception as e:
        logger.warning(f"Token expansion failed: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


def _compute_tokens_per_image_fast(
    image_dims: list[tuple[int, int]], processor: Any
) -> list[int]:
    """Compute per-image token counts from image dimensions without full preprocessing.

    Args:
        image_dims: list of (width, height) tuples per image.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor.image_processor not found")

    get_num_patches = getattr(image_processor, "get_number_of_image_patches", None)
    if not callable(get_num_patches):
        raise NotImplementedError("image_processor.get_number_of_image_patches not available")

    merge_size = getattr(image_processor, "merge_size", None)
    if merge_size is None:
        raise ValueError("image_processor.merge_size not found")

    tokens_per_image = []
    images_kwargs: dict[str, Any] = {}
    for width, height in image_dims:
        num_patches = int(get_num_patches(height, width, images_kwargs))
        tokens_per_image.append(num_patches // (merge_size**2))
    return tokens_per_image


def _tokenize_prompt_text_only(prompt: str, tokenizer: Any, processor: Any) -> list[int]:
    """Tokenize prompt text without image expansion."""
    if processor is not None and callable(processor):
        out = processor(text=[prompt], images=None, return_tensors=None, padding=False)
        return list(out["input_ids"][0])
    if hasattr(tokenizer, "__call__"):
        out = tokenizer([prompt], return_tensors=None)
        return list(out["input_ids"][0])
    return tokenizer.encode(prompt)


def _expand_image_tokens(
    tokens: list[int], image_token_id: int, tokens_per_image: list[int]
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Expand each image placeholder token and return per-image ranges."""
    if not tokens_per_image:
        return tokens, None

    expanded: list[int] = []
    ranges: list[tuple[int, int]] = []
    img_idx = 0

    for token in tokens:
        if token != image_token_id:
            expanded.append(token)
            continue
        if img_idx >= len(tokens_per_image):
            raise ValueError(f"More placeholders than images ({img_idx + 1} > {len(tokens_per_image)})")
        repeat = tokens_per_image[img_idx]
        start = len(expanded)
        expanded.extend([image_token_id] * repeat)
        ranges.append((start, len(expanded)))
        img_idx += 1

    if img_idx != len(tokens_per_image):
        raise ValueError(f"Not all images mapped: {img_idx} < {len(tokens_per_image)}")
    return expanded, ranges


def _compute_tokens_per_image(processor_output: dict, processor: Any) -> list[int]:
    """
    Compute the number of visual tokens for each image from processor output.

    Only Qwen-style processors (Qwen2-VL, Qwen2.5-VL) are supported.
    Other model families will raise ValueError.
    """
    processor_cls = type(processor).__qualname__
    if "qwen" not in processor_cls.lower():
        raise NotImplementedError(
            f"_compute_tokens_per_image only supports Qwen-style processors "
            f"tuples. Got processor class: {processor_cls}"
        )

    grid_thw = processor_output.get("image_grid_thw")
    if grid_thw is None:
        raise ValueError("image_grid_thw not found in processor output")

    merge_size = getattr(processor.image_processor, "merge_size", 2)
    return [int(t * h * w) // (merge_size**2) for t, h, w in grid_thw]


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
