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

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"
    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    LLAVA_NEXT_VIDEO_7B = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    QWEN_2_AUDIO_7B = "Qwen/Qwen2-Audio-7B-Instruct"


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
    audio_embeds: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""
    if model == SupportedModels.QWEN_2_AUDIO_7B:
        audio_embeds = audio_embeds.to(torch.bfloat16)
        assert audio_embeds.ndim == 2, "Audio embeddings must be 2D"
        return {"audio": [audio_embeds]}
    # Handle video models
    if model == SupportedModels.LLAVA_NEXT_VIDEO_7B:
        if video_numpy is None:
            raise ValueError("No video frames provided.")
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    # Model-specific image handling
    if model == SupportedModels.QWEN_2_5_VL_7B:
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
