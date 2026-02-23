# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal processing utilities for MM Router Worker."""

import logging
from dataclasses import dataclass
from typing import Any

from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.inputs.utils import default_multimodal_input_loader, load_image
from transformers import AutoConfig

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


def build_prompt_from_messages(messages: list[dict]) -> str:
    """Build a simple prompt string from messages."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            if texts:
                parts.append(f"{role}: {' '.join(texts)}")
    return "\n".join(parts)


def process_multimodal(
    messages: list[dict],
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
    model: str,
    model_type: str,
) -> ProcessedInput:
    """Process multimodal request: load images, get expanded tokens and mm_hashes."""
    try:
        prompt = build_prompt_from_messages(messages)

        # Use TRT-LLM loader to process images and get mm data
        inputs = default_multimodal_input_loader(
            tokenizer=tokenizer,
            model_dir=model,
            model_type=model_type,
            modality="multiple_image" if len(image_urls) > 1 else "image",
            prompts=[prompt],
            media=[image_urls],
            image_data_format="pt",
            device="cuda",
        )

        mm_input = inputs[0]
        processed_prompt = mm_input.get("prompt", prompt)
        multi_modal_data = mm_input.get("multi_modal_data")

        # Get expanded tokens and image ranges
        tokens, image_ranges = _get_expanded_tokens(
            processed_prompt, image_urls, tokenizer, processor, model, model_type
        )

        # Compute mm_hash for each image
        mm_hashes = _compute_mm_hashes(multi_modal_data)

        return ProcessedInput(
            tokens=tokens, mm_hashes=mm_hashes, image_ranges=image_ranges
        )
    except Exception as e:
        logger.error(f"MM processing failed: {e}", exc_info=True)
        raise


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


def _get_expanded_tokens(
    prompt: str,
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
    model_path: str,
    model_type: str,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Get tokens with visual expansion and find each image's token range."""
    if processor is None:
        return tokenizer.encode(prompt), None

    try:
        # TODO @zdren: use async_load_image or batch load
        pil_images = [load_image(url, format="pil") for url in image_urls]
        output = processor(
            text=[prompt], images=pil_images, return_tensors="pt", padding=True
        )
        tokens = output["input_ids"][0].tolist()

        # Get image_token_id from processor
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor.image_token_id not found")

        # Get replacement_id: TRTLLM uses vocab_size + 1 in KV events
        replacement_id = _get_replacement_id(model_path)

        # Find contiguous image token ranges and replace them in one pass
        contiguous_ranges = _find_and_replace_image_tokens(
            tokens, image_token_id, replacement_id
        )

        # Compute tokens per image from processor output
        tokens_per_image = _compute_tokens_per_image(output, processor, model_type)

        # Split ranges according to tokens_per_image
        image_ranges = _compute_per_image_ranges(contiguous_ranges, tokens_per_image)

        return tokens, image_ranges

    except Exception as e:
        logger.warning(f"HF processor failed: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


def _compute_tokens_per_image(
    processor_output: dict, processor: Any, model_type: str
) -> list[int]:
    """Compute the number of visual tokens for each image from processor output."""
    if model_type == "qwen2_vl":
        grid_thw = processor_output.get("image_grid_thw")
        if grid_thw is None:
            raise ValueError(
                "image_grid_thw not found in processor output for Qwen2-VL"
            )

        merge_size = getattr(processor.image_processor, "merge_size", 2)
        return [int(t * h * w) // (merge_size**2) for t, h, w in grid_thw]
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported yet")


def _get_replacement_id(model_path: str) -> int:
    """
    Get the replacement token ID for image tokens to match TRTLLM's KV event format.

    TRTLLM replaces image placeholder tokens with (vocab_size + 1) in KV events.
    The vocab_size comes from the model config, not the tokenizer.
    """

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        replacement_id = config.vocab_size + 1
        logger.info(f"Got vocab_size={config.vocab_size} from AutoConfig")
        return replacement_id
    except Exception as e:
        raise RuntimeError(
            f"Failed to get vocab_size from model config '{model_path}': {e}"
        ) from e


def _find_and_replace_image_tokens(
    tokens: list[int], image_token_id: int, replacement_id: int
) -> list[tuple[int, int]]:
    """
    Find all contiguous ranges of image tokens and replace them in place.

    Returns: list of (start, end) ranges for contiguous image token regions.
    """
    ranges = []
    start = None

    for i, t in enumerate(tokens):
        if t == image_token_id:
            tokens[i] = replacement_id
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i))
            start = None

    if start is not None:
        ranges.append((start, len(tokens)))

    if ranges:
        logger.info(
            f"Replaced {sum(e - s for s, e in ranges)} image tokens: {image_token_id} -> {replacement_id}"
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


def _compute_mm_hashes(multi_modal_data: dict | None) -> list[int] | None:
    """Compute mm_hash for each image."""
    if not multi_modal_data:
        return None
    try:
        result = apply_mm_hashes(multi_modal_data)
        if "image" in result and result["image"]:
            return [int(h[:16], 16) for h in result["image"]]
    except Exception as e:
        logger.warning(f"Failed to compute mm_hashes: {e}")
    return None
