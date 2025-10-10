# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"


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
        if model_name == SupportedModels.QWEN_2_5_VL_7B:
            embeddings = get_qwen_image_features(vision_encoder, image_embeds)

        else:
            raise NotImplementedError(f"Model not supported: {model_name}")

        # Normalize output shape
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

        return embeddings
