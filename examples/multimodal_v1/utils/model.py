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
from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig
from utils.protocol import EncodeResponse
from vllm import AsyncEngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker

# from transformers import AutoImageProcessor, LlavaForConditionalGeneration
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


logger = logging.getLogger(__name__)


# [gluo NOTE] in vLLM v1, Worker() usage below will results in NotImplementedError,
# must find another way to properly load the vision model given the model name (model_id).
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
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id, device_map="auto", torch_dtype=torch.float16
    # ).eval()

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_id, torch_dtype="auto", device_map="auto"
    # ).eval()
    # return model


def get_vision_embeddings_info(
    model_id: str, num_patches: int
) -> Tuple[Tuple[int, int, int], torch.dtype]:
    """Calculate vision embeddings size and dtype using model config
    Returns a tuple of (batch_size, num_patches, hidden_dim), dtype.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    assert num_patches > 0, "Number of patches must be positive"
    if not hasattr(config, "torch_dtype"):
        raise ValueError("Model config missing required 'torch_dtype' attribute")
    if not hasattr(config, "hidden_size"):
        logger.warning(
            "Model config missing required 'hidden_size' attribute, using 4096"
        )
        hidden_size = 4096
    else:
        hidden_size = config.hidden_size
    return (1, num_patches, hidden_size), config.torch_dtype


def construct_mm_data(
    model: str,
    encode_output: EncodeResponse,
    image_embeds: torch.Tensor,
    embeddings_dtype: torch.dtype,
) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""
    image_embeds = image_embeds.to(embeddings_dtype)
    if "Qwen2" in model:
        return {
            "image": {
                "image_embeds": image_embeds.squeeze(0),
                "image_grid_thw": torch.tensor(encode_output.image_grid_thw).squeeze(0),
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
