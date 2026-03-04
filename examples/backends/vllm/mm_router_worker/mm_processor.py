# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing utilities for vLLM MM Router Worker.

Key differences from TRT-LLM version:
- Image loading: async ImageLoader (no TRT-LLM dependency)
- mm_hash: SHA256 of normalized PNG bytes (matches vLLM multi_modal_uuids)
- Token replacement: NOT needed — vLLM keeps the original image_token_id as-is
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

logger = logging.getLogger(__name__)


def _elapsed_ms(start_time: float) -> float:
    """Return elapsed milliseconds from start_time."""
    return (time.perf_counter() - start_time) * 1000


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ProcessedInput:
    """Processed multimodal input."""

    tokens: list[int]
    mm_hashes: list[int] | None
    image_ranges: list[tuple[int, int]] | None  # [(start, end), ...] per image
    data_uris: list[str] | None  # base64 data URIs for forwarding to backend
    raw_bytes_list: list[bytes] | None = None  # raw image bytes for cache server


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
    image_loader: ImageLoader,
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

    # Load images as PIL + raw bytes in parallel (async, cached)
    load_images_start = time.perf_counter()
    results = await asyncio.gather(
        *[image_loader.load_image_with_raw_bytes(url) for url in image_urls]
    )
    load_images_ms = _elapsed_ms(load_images_start)
    pil_images = [r[0] for r in results]
    raw_bytes_list = [r[1] for r in results]

    # Get expanded tokens and image ranges (no token replacement for vLLM)
    expanded_tokens_start = time.perf_counter()
    tokens, image_ranges = _get_expanded_tokens(
        prompt, pil_images, tokenizer, processor
    )
    expanded_tokens_ms = _elapsed_ms(expanded_tokens_start)
    logger.info(f"Expanded: {len(tokens)} tokens, " f"image_ranges={image_ranges}")

    # Compute mm_hashes exactly like vLLM handler's multi_modal_uuids path.
    mm_hash_start = time.perf_counter()
    mm_uuids = compute_mm_uuids_from_images(pil_images)
    mm_hashes = [int(uuid[:16], 16) for uuid in mm_uuids]
    mm_hash_ms = _elapsed_ms(mm_hash_start)

    logger.info(f"mm_hashes={mm_hashes}")

    # Build data URIs from raw bytes (no PIL re-encode)
    data_uri_start = time.perf_counter()
    data_uris = _raw_bytes_to_data_uris(raw_bytes_list)
    data_uri_ms = _elapsed_ms(data_uri_start)

    total_ms = _elapsed_ms(total_start)
    logger.debug(
        "[perf][mm_router] process_multimodal: "
        "total=%.3fms prompt=%.3fms load_images=%.3fms "
        "expanded_tokens=%.3fms mm_hash=%.3fms data_uri=%.3fms "
        "images=%d prompt_chars=%d tokens=%d",
        total_ms,
        prompt_ms,
        load_images_ms,
        expanded_tokens_ms,
        mm_hash_ms,
        data_uri_ms,
        len(image_urls),
        len(prompt),
        len(tokens),
    )

    return ProcessedInput(
        tokens=tokens,
        mm_hashes=mm_hashes,
        image_ranges=image_ranges,
        data_uris=data_uris,
        raw_bytes_list=raw_bytes_list,
    )


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


def _detect_mime_type(raw_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _raw_bytes_to_data_uris(raw_bytes_list: list[bytes]) -> list[str]:
    """Convert raw image bytes to base64 data URIs (no PIL re-encode)."""
    data_uris = []
    for raw_bytes in raw_bytes_list:
        mime = _detect_mime_type(raw_bytes)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        data_uris.append(f"data:{mime};base64,{b64}")
    return data_uris


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

    total_start = time.perf_counter()
    fast_path_error: str | None = None

    try:
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor.image_token_id not found")

        try:
            expanded_tokens, image_ranges, fast_perf = _get_expanded_tokens_fast_path(
                prompt=prompt,
                pil_images=pil_images,
                tokenizer=tokenizer,
                processor=processor,
                image_token_id=image_token_id,
            )
            total_ms = _elapsed_ms(total_start)
            logger.debug(
                "[perf][mm_router] _get_expanded_tokens: "
                "mode=fast total=%.3fms fast=%.3fms "
                "compute_tokens_per_image=%.3fms tokenize_text=%.3fms "
                "expand=%.3fms images=%d prompt_chars=%d tokens=%d",
                total_ms,
                fast_perf["fast_total_ms"],
                fast_perf["compute_tokens_per_image_ms"],
                fast_perf["tokenize_text_ms"],
                fast_perf["expand_ms"],
                len(pil_images),
                len(prompt),
                len(expanded_tokens),
            )
            return expanded_tokens, image_ranges
        except Exception as fast_path_err:
            fast_path_error = f"{type(fast_path_err).__name__}: {fast_path_err}"
            logger.debug(
                "Fast MM token expansion path unavailable, falling back to HF processor: %s",
                fast_path_err,
                exc_info=True,
            )

        tokens, image_ranges, slow_perf = _get_expanded_tokens_slow_path(
            prompt=prompt,
            pil_images=pil_images,
            processor=processor,
            image_token_id=image_token_id,
        )
        total_ms = _elapsed_ms(total_start)
        logger.debug(
            "[perf][mm_router] _get_expanded_tokens: "
            "mode=slow total=%.3fms processor_call=%.3fms "
            "range_and_split=%.3fms images=%d prompt_chars=%d tokens=%d "
            "fast_path_error=%s",
            total_ms,
            slow_perf["processor_call_ms"],
            slow_perf["range_and_split_ms"],
            len(pil_images),
            len(prompt),
            len(tokens),
            fast_path_error,
        )

        return tokens, image_ranges

    except Exception as e:
        logger.warning(f"HF processor failed: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


def _get_expanded_tokens_fast_path(
    prompt: str,
    pil_images: list[Image.Image],
    tokenizer: Any,
    processor: Any,
    image_token_id: int,
) -> tuple[list[int], list[tuple[int, int]] | None, dict[str, float]]:
    """
    Expand image placeholder tokens using only prompt tokenization and image sizes.

    This avoids full pixel preprocessing in HF AutoProcessor for the hot path.
    """
    fast_start = time.perf_counter()

    tpi_start = time.perf_counter()
    tokens_per_image = _compute_tokens_per_image_fast(pil_images, processor)
    tpi_ms = _elapsed_ms(tpi_start)

    text_tokenize_start = time.perf_counter()
    text_tokens = _tokenize_prompt_text_only(prompt, tokenizer, processor)
    text_tokenize_ms = _elapsed_ms(text_tokenize_start)

    expand_start = time.perf_counter()
    expanded_tokens, image_ranges = _expand_image_tokens(
        text_tokens, image_token_id, tokens_per_image
    )
    expand_ms = _elapsed_ms(expand_start)

    return expanded_tokens, image_ranges, {
        "fast_total_ms": _elapsed_ms(fast_start),
        "compute_tokens_per_image_ms": tpi_ms,
        "tokenize_text_ms": text_tokenize_ms,
        "expand_ms": expand_ms,
    }


def _get_expanded_tokens_slow_path(
    prompt: str,
    pil_images: list[Image.Image],
    processor: Any,
    image_token_id: int,
) -> tuple[list[int], list[tuple[int, int]] | None, dict[str, float]]:
    """Fallback path using HF processor multimodal call output."""
    processor_call_start = time.perf_counter()
    output = processor(
        text=[prompt], images=pil_images, return_tensors="pt", padding=True
    )
    processor_call_ms = _elapsed_ms(processor_call_start)

    tokens = output["input_ids"][0].tolist()

    ranges_start = time.perf_counter()
    contiguous_ranges = _find_image_token_ranges(tokens, image_token_id)
    tokens_per_image = _compute_tokens_per_image(output, processor)
    image_ranges = _compute_per_image_ranges(contiguous_ranges, tokens_per_image)
    ranges_ms = _elapsed_ms(ranges_start)

    return tokens, image_ranges, {
        "processor_call_ms": processor_call_ms,
        "range_and_split_ms": ranges_ms,
    }


def _tokenize_prompt_text_only(prompt: str, tokenizer: Any, processor: Any) -> list[int]:
    """
    Tokenize prompt text without image expansion.

    We prefer processor.tokenizer for parity with the slow path behavior.
    """
    if processor is not None and callable(processor):
        # Use processor text path (without images) to preserve tokenizer kwargs
        # and special-token behavior as much as possible.
        out = processor(
            text=[prompt],
            images=None,
            return_tensors=None,
            padding=False,
        )
        return list(out["input_ids"][0])

    if hasattr(tokenizer, "__call__"):
        out = tokenizer([prompt], return_tensors=None)
        return list(out["input_ids"][0])

    return tokenizer.encode(prompt)


def _compute_tokens_per_image_fast(
    pil_images: list[Image.Image], processor: Any
) -> list[int]:
    """
    Compute per-image token counts from image sizes via HF image_processor utility.

    This avoids full image preprocessing (pixel value generation) in hot path.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor.image_processor not found")

    get_num_patches = getattr(image_processor, "get_number_of_image_patches", None)
    if not callable(get_num_patches):
        raise NotImplementedError(
            "image_processor.get_number_of_image_patches is not available"
        )

    merge_size = getattr(image_processor, "merge_size", None)
    if merge_size is None:
        raise ValueError("image_processor.merge_size not found")

    tokens_per_image = []
    images_kwargs: dict[str, Any] = {}
    for img in pil_images:
        num_patches = int(get_num_patches(img.height, img.width, images_kwargs))
        tokens_per_image.append(num_patches // (merge_size**2))
    return tokens_per_image


def _expand_image_tokens(
    tokens: list[int], image_token_id: int, tokens_per_image: list[int]
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Expand each image placeholder token and return per-image ranges."""
    if not tokens_per_image:
        return tokens, None

    expanded_tokens: list[int] = []
    image_ranges: list[tuple[int, int]] = []
    image_idx = 0

    for token in tokens:
        if token != image_token_id:
            expanded_tokens.append(token)
            continue

        if image_idx >= len(tokens_per_image):
            raise ValueError(
                "Found more image placeholder tokens than provided images "
                f"({image_idx + 1} > {len(tokens_per_image)})"
            )

        repeat = tokens_per_image[image_idx]
        if repeat <= 0:
            raise ValueError(
                f"Invalid image token count for image {image_idx}: repeat={repeat}"
            )

        start = len(expanded_tokens)
        expanded_tokens.extend([image_token_id] * repeat)
        end = len(expanded_tokens)
        image_ranges.append((start, end))
        image_idx += 1

    if image_idx != len(tokens_per_image):
        raise ValueError(
            "Not all images mapped to placeholder tokens: "
            f"{image_idx} < {len(tokens_per_image)}"
        )

    return expanded_tokens, image_ranges


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
