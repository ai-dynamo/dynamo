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
from typing import AsyncIterator

import connect
import torch
from components.worker import VllmPDWorker
from transformers import AutoImageProcessor, LlavaForConditionalGeneration
from utils.args import parse_vllm_args
from utils.image_loader import ImageLoader
from utils.logging import check_required_workers
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmEncodeWorker:
    decode_worker = depends(VllmPDWorker)

    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.MODEL_ID = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        # self.vision_model = load_vision_model(self.MODEL_ID)
        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID, device_map="auto", torch_dtype=torch.float16
        ).eval()

        self.min_workers = 1

    @endpoint()
    async def encode(
        self, request: vLLMMultimodalRequest
    ) -> AsyncIterator[MyRequestOutput]:
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        # The following steps encode the requested image and provided useful embeddings.
        # 1. Open the image from the provided URL.
        # 2. Process the image using the image processor.
        # 3. Run the image through the vision model's vision tower.
        # 4. Run the results of the vision tower through the multi-modal projector.
        # 5. Create a descriptor for the embeddings.
        # 6. Create a write operation using the serialized request and the descriptor.
        # 7. Await for the write operation to complete.
        # 8. Yield the encode response.

        try:
            image = await self.image_loader.load_image(request.image_url)

            logger.debug(f"Processing image for request: {{ id: {request_id} }}")
            image_embeds = self.image_processor(images=image, return_tensors="pt")
            # # Add a batch dimension to everything
            # for item in image_embeds:
            #     image_embeds[item] = image_embeds[item].unsqueeze(0).to(DEVICE)
            # logger.debug(f"Image embeds: {image_embeds}")

            # image_grid_thw = (
            #     image_embeds["image_grid_thw"].tolist()
            #     if "image_grid_thw" in image_embeds
            #     else None
            # )
            # image_sizes = (
            #     image_embeds["image_sizes"].tolist()
            #     if "image_sizes" in image_embeds
            #     else [image.size]
            # )
            # logger.debug(
            #     f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
            # )

            # with torch.no_grad():
            #     embeddings = self.vision_model.get_multimodal_embeddings(**image_embeds)
            #     if isinstance(embeddings, tuple) or isinstance(embeddings, list):
            #         # The result multimodal_embeddings may be a list or tuple of tensors, with each
            #         # tensor corresponding to a multimodal data item (image or video).
            #         # TODO: for multi-image support, this result will contain multiple tensors.
            #         embeddings = embeddings[0].unsqueeze(0)
            #     logger.debug(
            #         f"Embeddings: {{ shape: {embeddings.shape}, dtype: {embeddings.dtype}, device: {embeddings.device}, ptr: {embeddings.data_ptr()}, elements: {{ count: {embeddings.numel()}, size: {embeddings.element_size()} }} }}."
            #     )

            #     yield EncodeResponse(
            #         request_id=request.request_id,
            #         image_grid_thw=image_grid_thw,
            #         image_sizes=image_sizes,
            #     ).model_dump_json()

            with torch.no_grad():
                logger.debug(f"Vision model device: {self.vision_model.device}")
                vision_outputs = self.vision_model.vision_tower(
                    image_embeds["pixel_values"].to(self.vision_model.device)
                )
                logger.debug("Vision model completed.")

                embeddings = vision_outputs.last_hidden_state
                embeddings = self.vision_model.multi_modal_projector(embeddings)

            descriptor = connect.Descriptor(embeddings)

            with self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.to_serialized()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.image_url = None

                logger.debug(f"Request: {request.model_dump_json()}")

                # Get the response generator
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )
                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    @async_on_start
    async def async_init(self):
        logger.info("Startup started.")
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = VllmPDWorker.dynamo_address()  # type: ignore
        self.pd_worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        await check_required_workers(self.pd_worker_client, self.min_workers)

        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector(runtime=runtime, namespace=comp_ns)
        await self._connector.initialize()

        logger.info("Startup completed.")
