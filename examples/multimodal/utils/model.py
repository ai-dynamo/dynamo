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
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class SupportedModels:
    """Supported multimodal model identifiers"""

    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"
    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    LLAVA_NEXT_VIDEO_7B = "llava-hf/LLaVA-NeXT-Video-7B-hf"


def load_vision_model(model_id: str) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    model = AutoModel.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model


def get_vision_embeddings_info(
    model_id: str,
) -> Tuple[Tuple[int, int, int], torch.dtype]:
    """Calculate vision embeddings size and dtype using model config
    Returns a tuple of (batch_size, seq_len, hidden_dim), dtype.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if model_id == SupportedModels.LLAVA_1_5_7B:
        seq_len = 577
    elif model_id == SupportedModels.QWEN_2_5_VL_7B:
        seq_len = 345
    else:
        seq_len = 0

    if not hasattr(config, "torch_dtype"):
        raise ValueError("Model config missing required 'torch_dtype' attribute")
    if not hasattr(config, "hidden_size"):
        logger.warning(
            "Model config missing required 'hidden_size' attribute, using 4096"
        )
        hidden_size = 4096
    else:
        hidden_size = config.hidden_size
    return (1, seq_len, hidden_size), config.torch_dtype


def construct_mm_data(
    model: str,
    image_embeds: torch.Tensor,
    embeddings_dtype: torch.dtype,
    image_grid_thw: Optional[List[Any]],
) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""
    image_embeds = image_embeds.to(embeddings_dtype)
    if model == SupportedModels.QWEN_2_5_VL_7B:
        if image_grid_thw is not None and len(image_grid_thw) > 0:
            grid_thw_tensor = torch.tensor(image_grid_thw)
        else:
            raise ValueError("No image grid provided.")

        return {
            "image": {
                "image_embeds": image_embeds.squeeze(0),
                "image_grid_thw": grid_thw_tensor,
            }
        }
    else:
        return {"image": image_embeds}
