# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing utilities for vLLM MM Router Worker.

Key differences from TRT-LLM version:
- Image loading: PIL + requests/base64 (no TRT-LLM dependency)
- mm_hash: SHA256 of normalized PNG bytes (matches vLLM multi_modal_uuids)
- Token replacement: NOT needed — vLLM keeps the original image_token_id as-is
"""

import base64
import logging
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
) -> ProcessedInput:
    """
    Process multimodal request: load images, get expanded tokens and mm_hashes.

    Uses PIL for image loading and hashlib for mm_hash computation.
    Unlike TRT-LLM, vLLM keeps original image_token_id (no replacement).
    """
    # The preprocessed request does not carry a rendered template string; it carries
    # original messages in extra_args, so we must apply chat template again here.
    prompt = _build_prompt_with_images(messages, tokenizer, processor)
    logger.info(f"Prompt (first 300 chars): {prompt[:300]}")

    # Load images as PIL
    pil_images = []
    for url in image_urls:
        pil_img = _load_image(url)
        pil_images.append(pil_img)

    # Get expanded tokens and image ranges (no token replacement for vLLM)
    tokens, image_ranges = _get_expanded_tokens(
        prompt, pil_images, tokenizer, processor
    )
    logger.info(f"Expanded: {len(tokens)} tokens, " f"image_ranges={image_ranges}")

    # Compute mm_hashes exactly like vLLM handler's multi_modal_uuids path.
    mm_uuids = compute_mm_uuids_from_images(pil_images)
    mm_hashes = [int(uuid[:16], 16) for uuid in mm_uuids]

    logger.info(f"mm_hashes={mm_hashes}")

    return ProcessedInput(tokens=tokens, mm_hashes=mm_hashes, image_ranges=image_ranges)


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
            if block_end > img_start and block_start < img_end
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
    pil_images: list[Image.Image],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """
    Get tokens with visual expansion and find each image's token range.

    Unlike TRT-LLM, vLLM keeps the original image_token_id (no replacement).
    """
    if processor is None:
        return tokenizer.encode(prompt), None

    try:
        output = processor(
            text=[prompt], images=pil_images, return_tensors="pt", padding=True
        )
        tokens = output["input_ids"][0].tolist()

        # Get image_token_id from processor
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor.image_token_id not found")

        # Find contiguous image token ranges (NO replacement for vLLM)
        contiguous_ranges = _find_image_token_ranges(tokens, image_token_id)

        # Compute tokens per image from processor output
        tokens_per_image = _compute_tokens_per_image(output, processor)

        # Split ranges according to tokens_per_image
        image_ranges = _compute_per_image_ranges(contiguous_ranges, tokens_per_image)

        return tokens, image_ranges

    except Exception as e:
        logger.warning(f"HF processor failed: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


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
