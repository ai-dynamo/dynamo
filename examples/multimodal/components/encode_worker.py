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
import os
from functools import lru_cache
from io import BytesIO
from typing import AsyncIterator, Optional, Tuple

import requests
import torch
from components.worker import VllmWorker
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from utils.llava_model import LlavaConfig, LlavaForCausalLM
from utils.protocol import EncodeRequest, EncodeResponse

from dynamo.sdk import depends, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class EncodeWorker:
    worker = depends(VllmWorker)

    def __init__(self) -> None:
        self.VISION_MODEL_ID = "openai/clip-vit-large-patch14-336"
        self.MODEL_ID = "llava-hf/llava-1.5-7b-hf"
        self.device = "cpu"
        self.model = CLIPVisionModel.from_pretrained(
            self.VISION_MODEL_ID, device_map=None
        )
        self.model.requires_grad_(False)

        self.select_layer = -2  # hardcoded based on the llava lora
        self.processor = CLIPImageProcessor.from_pretrained(self.VISION_MODEL_ID)
        logger.info("Embedding model loaded.")

    def _feature_select(self, image_forward_outs):
        print("Entering _feature_select:")
        print(
            f"Available hidden states: {len(image_forward_outs.hidden_states)} layers"
        )
        image_features = image_forward_outs.hidden_states[self.select_layer]
        print(
            f"Selected layer: {self.select_layer}, image features shape: {image_features.shape}"
        )
        image_features = image_features[:, 1:]
        print(f"Removed first token, new shape: {image_features.shape}")
        return image_features

    @dynamo_endpoint()
    async def generate(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        image_features = None
        lora_path = self.download_lora()
        lora_name = "llava-v1.5-7b-task-lora"
        image_features = self.embed(request.request_id, request.image_url)
        print(image_features)
        print("Finished embedding step.")
        image_features = self.multi_modal_project(image_features, lora_name, lora_path)
        print(image_features)
        yield EncodeResponse(image_features=image_features.tolist()).model_dump_json()

    def embed(self, request_id: str, image_url: str) -> AsyncIterator[EncodeResponse]:
        logger.info(f"Embedding image from: {image_url}, request_id: {request_id}")
        # Check if the input image is a URL or a file path
        if image_url.startswith("http"):
            logger.info(f"Downloading image from: {image_url}")
            response = requests.get(image_url)
            logger.info(f"Downloaded image, response status: {response.status_code}")
            image_data = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            logger.info(f"Loading image from file path: {image_url}")
            image_data = Image.open(image_url).convert("RGB")

        print("Processing image data...")
        inputs = self.processor(image_data, return_tensors="pt")["pixel_values"]
        print(f"Processed image, input tensor shape: {inputs.shape}")

        print(f"Moving inputs to device {self.device}, converting to float16...")
        inputs = inputs.to(device=self.device, dtype=torch.float16)

        print("Running model forward pass...")
        image_forward_outs = self.model(
            inputs.to(device=self.device, dtype=torch.float32),
            output_hidden_states=True,
        )
        print(
            f"Model forward pass complete, received {len(image_forward_outs.hidden_states)} hidden states"
        )

        print("Selecting features....")
        image_features = self._feature_select(image_forward_outs).to(torch.float16)
        print(f"Image features shape after selection: {image_features.shape}")

        # yield EncodeResponse(image_features=image_features.tolist()).model_dump_json()
        return image_features

    def download_lora(self) -> str:
        # downloaded_lora_path = self.peft_provider.download_entire_model(lora_name, checkpoint_name, lora_name)
        # print("LoRA downloaded to temp directory %s", downloaded_lora_path)
        # download the model to a subdirectory in the current directory
        downloaded_lora_path = "/checkpoints/llava-v1.5-7b-task-lora"
        return downloaded_lora_path

    def multi_modal_project(
        self, image: torch.Tensor, lora_name: str, checkpoint_name: Optional[str]
    ) -> Tuple[torch.Tensor, str]:
        print("Loading multi modal projector...")
        mm_projector = self.load_multi_modal_projector(lora_name, checkpoint_name)
        print("Projecting image...")
        image_features = mm_projector.mm_project(image.to(device="cuda:1")).to(
            self.device
        )
        print("Projected image features:", image_features)
        return image_features, ""

    @lru_cache(maxsize=5)
    def load_multi_modal_projector(
        self, lora_name: str, lora_path: str
    ) -> LlavaForCausalLM:
        print("load_multi_modal_projector...")
        device_map = "cuda:1"
        kwargs = {"device_map": device_map, "torch_dtype": torch.float16}

        lora_cfg_pretrained = LlavaConfig.from_pretrained(
            lora_path, low_cpu_mem_usage=True
        )
        print("LlavaConfig loaded...")
        model = LlavaForCausalLM.from_pretrained(
            self.MODEL_ID, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
        )
        print("LlavaForCausalLM loaded...")

        if os.path.exists(os.path.join(lora_path, "non_lora_trainables.bin")):
            print("non_lora_trainables.bin exists...")
            non_lora_trainables = torch.load(
                os.path.join(lora_path, "non_lora_trainables.bin"), map_location="cpu"
            )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)
        else:
            print("non_lora_trainables.bin does not exist...")

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path, low_cpu_mem_usage=True)
        model = model.merge_and_unload(progressbar=True)
        model.requires_grad_(False)
        print("Multi modal projector model loaded for ", lora_name)
        return model
