# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_3_VL_30B_A3B = "Qwen/Qwen3-VL-30B-A3B-Instruct"


# List of all Qwen VL model variants for easy extension
QWEN_VL_MODELS = [
    SupportedModels.QWEN_2_5_VL_7B,
    SupportedModels.QWEN_3_VL_30B_A3B,
]


def is_qwen_vl_model(model_name: str) -> bool:
    """Check if a model is any Qwen VL variant."""
    return any(is_model_supported(model_name, m) for m in QWEN_VL_MODELS)


def is_qwen3_vl_model(model_name: str) -> bool:
    """Check if a model is a Qwen3-VL variant."""
    return is_model_supported(model_name, SupportedModels.QWEN_3_VL_30B_A3B)


def _get_qwen_visual_inputs(
    vision_encoder: torch.nn.Module, image_embeds: Dict[str, Any], model_label: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract and move Qwen visual inputs onto the vision encoder device.

    Returns:
        A tuple of (pixel_values, grid_thw).
    """
    pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)

    grid_thw = image_embeds.get("image_grid_thw")
    if grid_thw is None:
        raise ValueError("grid_thw is not provided")

    grid_thw = grid_thw.to(vision_encoder.device)
    logger.debug(f"{model_label} grid_thw shape: {grid_thw.shape}")
    return pixel_values, grid_thw


def load_vision_model(model_id: str) -> torch.nn.Module:
    """
    Load the appropriate vision model for the given model ID.

    For Qwen3-VL, loads only the vision tower (Qwen3VLMoeVisionModel) to avoid
    pulling in the full LLM weights. For other models, loads via AutoModel.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        The loaded vision model on GPU with float16 precision
    """
    if is_qwen3_vl_model(model_id):
        from transformers import Qwen3VLMoeVisionModel

        logger.info(f"Loading Qwen3-VL vision encoder for {model_id}")
        return Qwen3VLMoeVisionModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    logger.info(f"Loading full model for {model_id}")
    return AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )


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

    # Exact match
    if normalized_name == normalized_supported:
        return True

    # Handle local path case: compare only the model name part (without organization)
    # e.g., "qwen2.5-vl-7b-instruct" matches "qwen/qwen2.5-vl-7b-instruct"
    if "/" in normalized_supported:
        model_part = normalized_supported.split("/")[-1]
        if normalized_name == model_part:
            return True

    return False


def get_qwen_image_features(
    vision_encoder: torch.nn.Module, image_embeds: Dict[str, Any]
) -> torch.Tensor:
    """
    Extract image features by calling the vision encoder's forward() method.

    Handles the HuggingFace dataclass return type:
      - Qwen3-VL: returns BaseModelOutputWithDeepstackFeatures with
        last_hidden_state (n, 2048) + 3 deepstack feature tensors (n, 2048 each).
        These are concatenated to produce (n, 8192) for SGLang's
        separate_deepstack_embeds to split at the decoder layers.
      - Qwen2.5-VL: returns output with last_hidden_state (n, hidden_size),
        no deepstack features.

    Args:
        vision_encoder: The vision encoder model
        image_embeds: Dictionary containing pixel values and grid information

    Returns:
        Single concatenated tensor: (n, hidden_size * (1 + num_deepstack))
        for deepstack models, or (n, hidden_size) otherwise.

    Raises:
        ValueError: If grid_thw is not provided
    """
    pixel_values, grid_thw = _get_qwen_visual_inputs(
        vision_encoder, image_embeds, model_label="Qwen"
    )
    output = vision_encoder(pixel_values, grid_thw=grid_thw)

    # HuggingFace vision models return different types depending on
    # return_dict (default False for standalone models):
    #   tuple:     (last_hidden_state, deepstack_features_tuple)
    #   dataclass: .last_hidden_state + .deepstack_features
    #   tensor:    raw tensor (unlikely but handled)

    if isinstance(output, tuple):
        last_hidden_state = output[0]
        deepstack = output[1] if len(output) > 1 else None
    elif hasattr(output, "last_hidden_state"):
        last_hidden_state = output.last_hidden_state
        deepstack = getattr(output, "deepstack_features", None)
    else:
        # Raw tensor — return as-is
        logger.warning(
            f"[DEBUG] vision_encoder returned raw tensor: shape={output.shape}"
        )
        return output

    # Concatenate deepstack features if present.
    # Qwen3-VL: deepstack is a tuple of 3 tensors, each (n, 2048).
    # Result: (n, 2048 + 2048*3) = (n, 8192)
    if deepstack:
        if isinstance(deepstack, (tuple, list)):
            result = torch.cat([last_hidden_state] + list(deepstack), dim=-1)
        else:
            result = torch.cat([last_hidden_state, deepstack], dim=-1)
    else:
        result = last_hidden_state

    logger.warning(
        f"[DEBUG] get_qwen_image_features: shape={result.shape}, dtype={result.dtype}"
    )
    return result


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
        NotImplementedError: If model is not supported
    """
    with torch.no_grad():
        if is_qwen_vl_model(model_name):
            embeddings = get_qwen_image_features(vision_encoder, image_embeds)
        else:
            normalized_name = normalize_model_name(model_name)
            raise NotImplementedError(
                f"Model not supported: {normalized_name} (original: {model_name})"
            )

        # Normalize output shape
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

        logger.warning(
            f"[DEBUG-3] encode_image_embeddings final: shape={embeddings.shape}"
        )
        return embeddings
