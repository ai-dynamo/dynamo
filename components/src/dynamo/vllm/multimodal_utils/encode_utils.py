# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any, Dict, Optional

import torch
from vllm.config import ECTransferConfig

from .model import SupportedModels, is_model_supported, is_qwen_vl_model

logger = logging.getLogger(__name__)


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
        if is_model_supported(model_name, SupportedModels.LLAVA_1_5_7B):
            pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)
            vision_outputs = vision_encoder(pixel_values)

            if projector is None:
                raise ValueError(f"Projector not found for LLaVA model: {model_name}")

            embeddings = projector(vision_outputs.last_hidden_state)

        elif is_qwen_vl_model(model_name):
            embeddings = get_qwen_image_features(vision_encoder, image_embeds)

        else:
            raise NotImplementedError(f"Model not supported: {model_name}")

        # Normalize output shape
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

        return embeddings


def get_encoder_components(
    model_name: str, vision_model: torch.nn.Module
) -> tuple[Any, Optional[Any]]:
    """
    Get the appropriate vision encoder and projector components for a given model.

    Args:
        model_name: The model identifier
        vision_model: The loaded vision model

    Returns:
        Tuple of (vision_encoder, projector) where types depend on the model

    Raises:
        NotImplementedError: If model is not supported
    """
    if is_model_supported(model_name, SupportedModels.LLAVA_1_5_7B):
        vision_encoder = vision_model.vision_tower
        projector = getattr(vision_model, "multi_modal_projector", None)
        return vision_encoder, projector

    elif is_qwen_vl_model(model_name):
        vision_encoder = vision_model
        projector = None
        return vision_encoder, projector

    else:
        raise NotImplementedError(f"Model not supported: {model_name}")


def create_ec_transfer_config(
    engine_id: str,
    ec_role: str,
    ec_connector_backend: str = "ECExampleConnector",
    ec_storage_path: Optional[str] = None,
    ec_extra_config: Optional[str] = None,
) -> ECTransferConfig:
    """
    Create ECTransferConfig for vLLM encoder disaggregation.

    Args:
        engine_id: Unique identifier for this engine instance
        ec_role: Role of this instance - "ec_producer" (encoder) or "ec_consumer" (PD worker)
        ec_connector_backend: ECConnector implementation class name
        ec_storage_path: Storage path for disk-based connectors
        ec_extra_config: Additional connector config as JSON string

    Returns:
        ECTransferConfig configured for the specified role
    """
    # Parse extra config if provided
    extra_config: Dict[str, Any] = {}
    if ec_extra_config:
        try:
            extra_config = json.loads(ec_extra_config)
            logger.debug(f"Parsed ec_extra_config: {extra_config}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --ec-extra-config: {e}")

    # Add storage path to config if provided
    if ec_storage_path:
        extra_config["shared_storage_path"] = ec_storage_path
    else:
        raise ValueError("ec_storage_path is not provided")

    logger.info(
        f"Creating ECTransferConfig: engine_id={engine_id}, role={ec_role}, "
        f"backend={ec_connector_backend}, config={extra_config}"
    )

    return ECTransferConfig(
        engine_id=engine_id,
        ec_role=ec_role,
        ec_connector=ec_connector_backend,
        ec_connector_extra_config=extra_config,
    )
