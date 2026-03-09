# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing for vLLM MM Router Worker.

Handles image loading, token expansion, and mm_hash computation for
KV-cache-aware routing. Unlike TRT-LLM, vLLM keeps the original
image_token_id as-is (no token replacement needed).
"""

import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

from PIL import Image

from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

from .image_loader import RoutingImageLoader

logger = logging.getLogger(__name__)


@dataclass
class ProcessedInput:
    tokens: list[int]
    mm_hashes: list[int] | None
    image_ranges: list[tuple[int, int]] | None


# =============================================================================
# Public API
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
    image_loader: RoutingImageLoader,
) -> tuple[ProcessedInput, list[bytes]]:
    """Process multimodal request: load images, expand tokens, compute mm_hashes.

    Returns (ProcessedInput, raw_bytes_list).
    """
    t0 = time.perf_counter()

    prompt = _apply_chat_template(messages, tokenizer, processor)
    t1 = time.perf_counter()

    image_data = await image_loader.load_batch(image_urls)
    t2 = time.perf_counter()
    raw_bytes_list = [d[0] for d in image_data]
    image_dims = [(d[1], d[2]) for d in image_data]

    tokens, image_ranges = _get_expanded_tokens(
        prompt, image_dims, raw_bytes_list, tokenizer, processor
    )
    t3 = time.perf_counter()

    mm_uuids = compute_mm_uuids_from_images(raw_bytes_list)
    t4 = time.perf_counter()
    mm_hashes = [int(uuid[:16], 16) for uuid in mm_uuids]

    logger.info(
        "[perf][process_mm] template=%.2fms load_images=%.2fms "
        "expand_tokens=%.2fms hash=%.2fms total=%.2fms n_images=%d",
        1000 * (t1 - t0), 1000 * (t2 - t1),
        1000 * (t3 - t2), 1000 * (t4 - t3),
        1000 * (t4 - t0), len(image_urls),
    )

    return ProcessedInput(tokens=tokens, mm_hashes=mm_hashes, image_ranges=image_ranges), raw_bytes_list


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_ranges: list[tuple[int, int]] | None,
) -> list[dict | None] | None:
    """Build per-block mm_info for KV-cache-aware routing.

    Uses a two-pointer scan over sorted image_ranges for O(num_blocks + num_images)
    instead of O(num_blocks × num_images).
    """
    if not mm_hashes or not image_ranges or len(mm_hashes) != len(image_ranges):
        return None

    num_blocks = (num_tokens + block_size - 1) // block_size

    # Sort images by start position (they should already be sorted, but be safe)
    sorted_imgs = sorted(
        zip(image_ranges, mm_hashes), key=lambda x: x[0][0]
    )

    result: list[dict | None] = []
    img_ptr = 0  # first image that hasn't ended before current block
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Advance past images that ended before this block
        while img_ptr < len(sorted_imgs) and sorted_imgs[img_ptr][0][1] <= block_start:
            img_ptr += 1

        # Collect overlapping images (scan forward from img_ptr)
        mm_objects = []
        for j in range(img_ptr, len(sorted_imgs)):
            img_start, img_end = sorted_imgs[j][0]
            if img_start >= block_end:
                break  # no more images can overlap
            mm_objects.append({"mm_hash": sorted_imgs[j][1], "offsets": []})

        result.append({"mm_objects": mm_objects} if mm_objects else None)
    return result


# =============================================================================
# Token expansion: fast path (dimensions) → slow path (HF processor)
# =============================================================================


def _apply_chat_template(messages: list[dict], tokenizer: Any, processor: Any) -> str:
    """Re-apply chat template for routing token expansion.

    Cannot reuse Frontend's token_ids because the Frontend tokenizer may lack
    vision-specific markers (e.g. <|vision_start|><|image_pad|><|vision_end|>
    for Qwen). The processor's template produces the correct placeholder
    structure needed for image token expansion and block_mm_infos.
    """
    for obj in (processor, tokenizer):
        if obj is not None and hasattr(obj, "apply_chat_template"):
            return obj.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    raise ValueError("Neither processor nor tokenizer provides apply_chat_template")


def _get_expanded_tokens(
    prompt: str,
    image_dims: list[tuple[int, int]],
    raw_bytes_list: list[bytes],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Expand image placeholder tokens. Fast path from dims, slow path via processor."""
    if processor is None:
        return tokenizer.encode(prompt), None

    try:
        return _expand_from_dims(prompt, image_dims, tokenizer, processor)
    except Exception as e:
        logger.info("Fast path failed (%s), falling back to processor", e)

    try:
        pil_images = [Image.open(BytesIO(raw)) for raw in raw_bytes_list]
        return _expand_with_processor(prompt, pil_images, tokenizer, processor)
    except Exception as e:
        logger.warning("Slow path also failed: %s", e, exc_info=True)
        return tokenizer.encode(prompt), None


# -- Fast path --


def _expand_from_dims(
    prompt: str,
    image_dims: list[tuple[int, int]],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Expand placeholders using dimension-based token counts (Qwen-style)."""
    image_processor = processor.image_processor
    get_num_patches = image_processor.get_number_of_image_patches
    merge_size = image_processor.merge_size
    image_token_id = processor.image_token_id

    tokens_per_image = []
    for w, h in image_dims:
        n_patches: int = int(get_num_patches(h, w, {}))  # type: ignore[arg-type]
        tokens_per_image.append(n_patches // (merge_size ** 2))

    base_tokens = tokenizer.encode(prompt)
    placeholders = [i for i, t in enumerate(base_tokens) if t == image_token_id]

    if len(placeholders) != len(image_dims):
        raise ValueError(
            f"Placeholder count ({len(placeholders)}) != image count ({len(image_dims)})"
        )

    expanded: list[int] = []
    ranges: list[tuple[int, int]] = []
    prev = 0
    for idx, pos in enumerate(placeholders):
        expanded.extend(base_tokens[prev:pos])
        start = len(expanded)
        n = tokens_per_image[idx]
        expanded.extend([image_token_id] * n)
        ranges.append((start, start + n))
        prev = pos + 1
    expanded.extend(base_tokens[prev:])
    return expanded, ranges


# -- Slow path --


def _expand_with_processor(
    prompt: str,
    pil_images: Sequence[Image.Image],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Expand using full HF processor (works for any model, ~55ms)."""
    output = processor(
        text=[prompt], images=pil_images, return_tensors="pt", padding=True
    )
    tokens = output["input_ids"][0].tolist()

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        return tokens, None

    # Find contiguous image token regions, then split by per-image token counts
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    grid_thw = output.get("image_grid_thw")
    if grid_thw is None:
        return tokens, None
    tokens_per_image = [int(t * h * w) // (merge_size ** 2) for t, h, w in grid_thw]

    # Find contiguous ranges of image_token_id
    contiguous: list[tuple[int, int]] = []
    run_start = None
    for i, t in enumerate(tokens):
        if t == image_token_id:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            contiguous.append((run_start, i))
            run_start = None
    if run_start is not None:
        contiguous.append((run_start, len(tokens)))

    # Split contiguous ranges into per-image ranges
    result: list[tuple[int, int]] = []
    img_idx = 0
    for rng_start, rng_end in contiguous:
        pos = rng_start
        while img_idx < len(tokens_per_image):
            needed = tokens_per_image[img_idx]
            if pos + needed <= rng_end:
                result.append((pos, pos + needed))
                pos += needed
                img_idx += 1
            else:
                break
    return tokens, result if len(result) == len(tokens_per_image) else None
