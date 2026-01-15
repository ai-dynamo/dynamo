# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processing utilities for MM Router Worker.

This module provides functions for:
- Processing multimodal inputs (images)
- Computing mm_hash using TRT-LLM's apply_mm_hashes
- Building block_mm_infos for KV-aware routing
"""

import logging
from dataclasses import dataclass
from typing import Any

from tensorrt_llm.inputs.multimodal import apply_mm_hashes
from tensorrt_llm.inputs.utils import default_multimodal_input_loader, load_image

logger = logging.getLogger(__name__)

# Qwen2-VL specific token IDs
QWEN2_VL_IMAGE_TOKEN_ID = 151655
QWEN2_VL_REPLACEMENT_ID = 151937


@dataclass
class ProcessedInput:
    """Processed input ready for routing."""

    tokens: list[int]
    mm_hashes: list[int] | None
    image_offsets_list: list[list[int]] | None


def extract_image_urls(messages: list[dict]) -> list[str]:
    """Extract image URLs from OpenAI-format messages."""
    image_urls = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url")
                    if url:
                        image_urls.append(url)
    return image_urls


def build_prompt_from_messages(messages: list[dict]) -> str:
    """Build a simple prompt string from messages."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                parts.append(f"{role}: {' '.join(text_parts)}")

    return "\n".join(parts)


def process_multimodal(
    messages: list[dict],
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
    model: str,
    model_type: str,
) -> ProcessedInput:
    """
    Process multimodal request: load images, compute tokens and mm_hashes.

    Args:
        messages: OpenAI-format messages
        image_urls: List of image URLs extracted from messages
        tokenizer: TRT-LLM tokenizer
        processor: HuggingFace AutoProcessor (for getting visual tokens)
        model: Model path/name
        model_type: Model type (e.g., "qwen2_vl")

    Returns:
        ProcessedInput with tokens, mm_hashes, and image_offsets_list
    """
    try:
        prompt = build_prompt_from_messages(messages)
        logger.info(f"Built prompt from messages: {prompt[:200]}...")

        # Use TRT-LLM's multimodal loader to process images
        modality = "multiple_image" if len(image_urls) > 1 else "image"
        logger.info(f"Calling default_multimodal_input_loader with modality={modality}")
        inputs = default_multimodal_input_loader(
            tokenizer=tokenizer,
            model_dir=model,
            model_type=model_type,
            modality=modality,
            prompts=[prompt],
            media=[image_urls],
            image_data_format="pt",
            device="cuda",
        )
        mm_input = inputs[0]
        processed_prompt = mm_input.get("prompt", prompt)
        multi_modal_data = mm_input.get("multi_modal_data")
        logger.info(f"TRT-LLM loader returned processed_prompt: {processed_prompt[:200] if processed_prompt else 'None'}...")
        logger.info(f"multi_modal_data keys: {multi_modal_data.keys() if multi_modal_data else 'None'}")

        # Get tokens with visual expansion
        tokens, image_offsets_list = get_mm_tokens(
            processed_prompt, image_urls, tokenizer, processor
        )

        # Compute mm_hash for each image
        mm_hashes = compute_mm_hashes(multi_modal_data)

        logger.debug(
            f"Processed MM input: {len(tokens)} tokens, "
            f"{len(mm_hashes) if mm_hashes else 0} images"
        )

        return ProcessedInput(
            tokens=tokens,
            mm_hashes=mm_hashes,
            image_offsets_list=image_offsets_list,
        )

    except Exception as e:
        logger.warning(f"MM processing failed: {e}, falling back to text-only")
        prompt = build_prompt_from_messages(messages)
        return ProcessedInput(
            tokens=tokenizer.encode(prompt),
            mm_hashes=None,
            image_offsets_list=None,
        )


def get_mm_tokens(
    prompt: str,
    image_urls: list[str],
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[list[int]] | None]:
    """
    Get tokens with visual expansion and find image token positions.

    Args:
        prompt: Text prompt (may contain image placeholders)
        image_urls: List of image URLs
        tokenizer: TRT-LLM tokenizer
        processor: HuggingFace AutoProcessor

    Returns:
        Tuple of (tokens, image_offsets_list)
    """
    if processor is None:
        logger.warning("HF processor is None, using tokenizer.encode() - no visual token expansion!")
        return tokenizer.encode(prompt), None

    try:
        # Load images as PIL
        logger.info(f"Loading {len(image_urls)} images for HF processor")
        pil_images = [load_image(url, format="pil") for url in image_urls]

        # Process with HuggingFace processor to get visual-expanded tokens
        logger.info(f"Calling HF processor with prompt length: {len(prompt)}")
        processor_output = processor(
            text=[prompt],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        tokens = processor_output["input_ids"][0].tolist()
        logger.info(f"HF processor returned {len(tokens)} tokens")
        logger.info(f"Token sample (first 50): {tokens[:50]}")

        # Find and replace image token positions
        image_token_id = getattr(processor, "image_token_id", QWEN2_VL_IMAGE_TOKEN_ID)
        logger.info(f"Looking for image_token_id: {image_token_id}")

        # Count occurrences of image token
        img_token_count = tokens.count(image_token_id)
        logger.info(f"Found {img_token_count} occurrences of image_token_id={image_token_id}")

        return replace_image_tokens(tokens, image_token_id, QWEN2_VL_REPLACEMENT_ID)

    except Exception as e:
        logger.warning(f"Failed to process with HF processor: {e}", exc_info=True)
        return tokenizer.encode(prompt), None


def replace_image_tokens(
    tokens: list[int],
    image_token_id: int,
    replacement_id: int,
) -> tuple[list[int], list[list[int]] | None]:
    """
    Replace image tokens and return their positions.

    Args:
        tokens: Token list from processor
        image_token_id: ID of image placeholder token
        replacement_id: ID to replace with

    Returns:
        Tuple of (modified tokens, list of [start, end] ranges for each image)
    """
    image_offsets_list: list[list[int]] = []
    current_start: int | None = None

    for i, t in enumerate(tokens):
        if t == image_token_id:
            if current_start is None:
                current_start = i
            tokens[i] = replacement_id
        else:
            if current_start is not None:
                image_offsets_list.append([current_start, i])
                current_start = None

    # Handle case where tokens end with image token
    if current_start is not None:
        image_offsets_list.append([current_start, len(tokens)])

    return tokens, image_offsets_list if image_offsets_list else None


def compute_mm_hashes(multi_modal_data: dict | None) -> list[int] | None:
    """
    Compute mm_hash for each image in multimodal data.

    Uses TRT-LLM's apply_mm_hashes which computes BLAKE3 hash of image tensors.

    Args:
        multi_modal_data: Dict containing processed image tensors

    Returns:
        List of 64-bit integer hashes, one per image
    """
    if not multi_modal_data:
        return None

    try:
        mm_hashes_dict = apply_mm_hashes(multi_modal_data)

        if "image" in mm_hashes_dict and mm_hashes_dict["image"]:
            # Convert 256-bit hex digest to 64-bit int (take first 16 hex chars)
            mm_hashes = [
                int(hex_digest[:16], 16) for hex_digest in mm_hashes_dict["image"]
            ]
            logger.debug(f"Computed mm_hashes for {len(mm_hashes)} images")
            return mm_hashes

    except Exception as e:
        logger.warning(f"Failed to compute mm_hashes: {e}")

    return None


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_offsets_list: list[list[int]] | None,
) -> list[dict | None] | None:
    """
    Build per-block mm_info for routing hash computation.

    Each block that overlaps with an image gets mm_info containing the
    mm_hash of that image. This is used by compute_block_hash_for_seq_py
    to compute MM-aware block hashes.

    Args:
        num_tokens: Total number of tokens
        block_size: KV cache block size
        mm_hashes: List of mm_hash values, one per image
        image_offsets_list: List of [start, end] token ranges for each image

    Returns:
        List of mm_info dicts (or None) for each block
    """
    if mm_hashes is None or image_offsets_list is None:
        return None

    if len(mm_hashes) != len(image_offsets_list):
        logger.warning(
            f"mm_hashes ({len(mm_hashes)}) and image_offsets_list "
            f"({len(image_offsets_list)}) length mismatch"
        )
        return None

    num_blocks = (num_tokens + block_size - 1) // block_size

    result: list[dict | None] = []
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Find all images that overlap with this block
        mm_objects = []
        for mm_hash, offsets in zip(mm_hashes, image_offsets_list):
            img_start, img_end = offsets
            # Check if block and image ranges overlap
            if block_end > img_start and block_start < img_end:
                mm_objects.append({"mm_hash": mm_hash, "offsets": [offsets]})

        # Only add mm_info if block contains image tokens
        if mm_objects:
            result.append({"mm_objects": mm_objects})
        else:
            result.append(None)

    return result
