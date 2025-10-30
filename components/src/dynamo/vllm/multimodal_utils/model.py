# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"
    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    LLAVA_NEXT_VIDEO_7B = "llava-hf/LLaVA-NeXT-Video-7B-hf"


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


def load_vision_model(model_id: str) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    model = AutoModel.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model


def construct_mm_data(
    model: str,
    embeddings_dtype: torch.dtype,
    image_embeds: Optional[torch.Tensor] = None,
    video_numpy: Optional[Any] = None,
    image_grid_thw: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""

    # Handle video models
    if is_model_supported(model, SupportedModels.LLAVA_NEXT_VIDEO_7B):
        if video_numpy is None:
            raise ValueError("No video frames provided.")
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    # Model-specific image handling
    if is_model_supported(model, SupportedModels.QWEN_2_5_VL_7B):
        return _construct_qwen_image_data(image_embeds, image_grid_thw)
    else:
        # Default image handling for other models (e.g., LLAVA_1_5_7B)
        return {"image": image_embeds}


def _construct_qwen_image_data(
    image_embeds: torch.Tensor, image_grid_thw: Optional[List[Any]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct image data specifically for Qwen models."""
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for Qwen model.")

    grid_thw_tensor = torch.tensor(image_grid_thw)

    return {
        "image": {
            "image_embeds": image_embeds.squeeze(0),
            "image_grid_thw": grid_thw_tensor,
        }
    }
