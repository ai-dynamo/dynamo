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

from typing import Any, Dict

import torch
from transformers import AutoConfig
from utils.protocol import EncodeResponse
from vllm import AsyncEngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker


def load_vision_model(model_id: str) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    engine_args = AsyncEngineArgs(model=model_id, trust_remote_code=True)

    engine_config = engine_args.create_engine_config()
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    worker = Worker(
        vllm_config=engine_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
    )
    # Initialize the worker.
    worker.init_device()
    worker.load_model()
    return worker.model_runner.model


def get_vision_embeddings_size(model_id: str, num_patches: int) -> tuple[int, int, int]:
    """Calculate vision embeddings size using model config and image processor
    Returns a tuple of (batch_size, num_patches, hidden_dim).
    """
    config = AutoConfig.from_pretrained(model_id)
    assert num_patches > 0, "Number of patches must be positive"
    return 1, num_patches, getattr(config, "hidden_size", 4096)


def construct_mm_data(
    model: str, encode_output: EncodeResponse, image_embeds: torch.Tensor
) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""
    if "Qwen2" in model:
        return {
            "image": {
                "image_embeds": image_embeds.squeeze(0),
                "image_grid_thw": torch.tensor(encode_output.image_grid_thw),
            }
        }
    elif "MiniCPM-V" in model:
        return {
            "image": {
                "image_embeds": image_embeds,
                "image_sizes": encode_output.image_sizes,
            }
        }
    else:
        return {"image": image_embeds}
