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

import asyncio
import logging
import signal
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, LlavaForConditionalGeneration
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker:
    def __init__(self):
        self.client = None
        self.min_workers = 1
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )

        if self.engine_args.remote_prefill:
            raise NotImplementedError(
                "Remote prefill is not supported for aggregated multimodal example"
            )

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )
        # For encoding
        self.MODEL_ID = self.engine_args.model

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )

        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID, device_map="auto", torch_dtype=torch.float16
        ).eval()

        logger.info("VllmWorker has been initialized")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def open_image(self, image: str) -> Image.Image:
        try:
            if image.startswith("http") or image.startswith("https"):
                response = requests.get(image)
                image_data = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image_data = Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            raise e
        return image_data

    @dynamo_endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        image = self.open_image(request.image_url)
        image_embeds = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logger.debug(f"Vision model device: {self.vision_model.device}")
            vision_outputs = self.vision_model.vision_tower(
                image_embeds["pixel_values"].to(self.vision_model.device)
            )

            image_features = vision_outputs.last_hidden_state
            image_features = self.vision_model.multi_modal_projector(image_features)

        remote_prefill_params = None
        logger.info(
            f"Prefilling locally for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
        )

        # rust HTTP requires Delta streaming
        request.sampling_params.output_kind = RequestOutputKind.DELTA
        async for response in self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data={"image": image_features},
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
