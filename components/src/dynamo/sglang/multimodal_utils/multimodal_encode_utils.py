# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"


def normalize_model_name(model_name: str) -> str:
    """
    Extract and normalize model name from various formats including HuggingFace cache paths.

    Args:
        model_name: Model identifier which can be:
            - A simple model name: "Qwen/Qwen2.5-VL-7B-Instruct"
            - A HuggingFace cache path: "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/..."
            - A local path to a model directory

    Returns:
        Normalized model name in the format "organization/model-name"

    Examples:
        >>> normalize_model_name("Qwen/Qwen2.5-VL-7B-Instruct")
        "Qwen/Qwen2.5-VL-7B-Instruct"
        >>> normalize_model_name("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/...")
        "Qwen/Qwen2.5-VL-7B-Instruct"
    """
    # If it's already a simple model name (org/model format), return as-is
    if "/" in model_name and not model_name.startswith("/"):
        return model_name

    # Handle HuggingFace cache paths
    if "models--" in model_name:
        # Extract from cache path format: models--ORG--MODEL-NAME
        # Split on "models--" then on "--" to handle dashes in org/model names
        parts_after_models = model_name.split("models--", 1)
        if len(parts_after_models) > 1:
            # Split the remaining part on "--" and take the last two segments
            segments = parts_after_models[1].split("--")
            if len(segments) >= 2:
                # Take all segments except the last as org (rejoined with dashes)
                # and the last segment (before any slash) as model name
                org_segments = segments[:-1]
                model_segment = segments[-1].split("/")[
                    0
                ]  # Remove any path after model name

                org = "--".join(org_segments)  # Rejoin org parts with dashes
                model = model_segment
                return f"{org}/{model}"

    # Handle local directory paths - extract the last directory name
    path = Path(model_name)
    if path.exists() and path.is_dir():
        return path.name

    # If no pattern matches, return the original name
    return model_name


def is_model_supported(model_name: str, supported_model: str) -> bool:
    """
    Check if a model name matches a supported model, handling various naming formats.

    Args:
        model_name: The model name to check (may be path, cache name, etc.)
        supported_model: The supported model identifier

    Returns:
        True if the model is supported, False otherwise
    """
    normalized_name = normalize_model_name(model_name).lower()
    normalized_supported = normalize_model_name(supported_model).lower()

    return normalized_name == normalized_supported


def get_qwen_image_features(
    vision_encoder: torch.nn.Module, image_embeds: Dict[str, Any]
) -> torch.Tensor:
    """
    Extract image features using Qwen-style vision encoder.

    Args:
        vision_encoder: The vision encoder model
        image_embeds: Dictionary containing pixel values and grid information

    Returns:
        Processed image features tensor

    Raises:
        ValueError: If grid_thw is not provided for Qwen model
    """
    pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)

    grid_thw = image_embeds.get("image_grid_thw", None)
    if grid_thw is not None:
        grid_thw = grid_thw.to(vision_encoder.device)
        logger.debug(f"Qwen grid_thw shape: {grid_thw.shape}")
    else:
        raise ValueError("grid_thw is not provided")

    return (
        vision_encoder.get_image_features(pixel_values, grid_thw)  # type: ignore
        if grid_thw is not None
        else vision_encoder.get_image_features(pixel_values)  # type: ignore
    )


def encode_image_embeddings(
    model_name: str,
    image_embeds: Dict[str, Any],
    vision_encoder: torch.nn.Module,
    projector: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Encode image embeddings using the appropriate model-specific encoder.

    Args:
        model_name: The model identifier
        image_embeds: Dictionary containing processed image data
        vision_encoder: The vision encoder module
        projector: The multimodal projector (required for LLaVA-style models)

    Returns:
        Encoded embeddings tensor with normalized shape

    Raises:
        ValueError: If projector is missing for LLaVA models
        NotImplementedError: If model is not supported
    """
    with torch.no_grad():
        # Route through the correct encoder based on model
        if is_model_supported(model_name, SupportedModels.QWEN_2_5_VL_7B):
            embeddings = get_qwen_image_features(vision_encoder, image_embeds)

        else:
            # Provide more helpful error message with normalized model name
            normalized_name = normalize_model_name(model_name)
            raise NotImplementedError(
                f"Model not supported: {normalized_name} (original: {model_name})"
            )

        # Normalize output shape
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

        return embeddings
