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
import base64
import binascii
import logging
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import connect
import httpx
import torch
from PIL import Image
from transformers import AutoImageProcessor
from utils.model import load_vision_model
from utils.protocol import EncodeRequest, EncodeResponse
from utils.vllm import parse_vllm_args

from dynamo.sdk import async_on_start, endpoint, service

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
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmEncodeWorker:
    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.MODEL_ID = self.engine_args.model

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.vision_model = load_vision_model(self.MODEL_ID)

        self._image_cache: dict[str, Image.Image] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_timeout = 30.0

    async def load_image(self, image_url: str) -> Image.Image:
        parsed_url = urlparse(image_url)

        # For HTTP(S) URLs, check cache first
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                return self._image_cache[image_url_lower]

        try:
            if parsed_url.scheme == "data":
                # Parse data URL format: data:[<media type>][;base64],<data>
                if not parsed_url.path.startswith("image/"):
                    raise ValueError("Data URL must be an image type")

                # Split the path into media type and data
                media_type, data = parsed_url.path.split(",", 1)
                if ";base64" not in media_type:
                    raise ValueError("Data URL must be base64 encoded")

                try:
                    image_bytes = base64.b64decode(data)
                    image_data = BytesIO(image_bytes)
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif parsed_url.scheme in ("http", "https"):
                if not self._http_client:
                    raise RuntimeError("HTTP client not initialized")

                response = await self._http_client.get(image_url)
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Empty response content from image URL")

                image_data = BytesIO(response.content)
            else:
                raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")

            # PIL is sync, so offload to a thread to avoid blocking the event loop
            image = await asyncio.to_thread(Image.open, image_data)

            # Validate image format and convert to RGB
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError(f"Unsupported image format: {image.format}")

            image_converted = image.convert("RGB")

            # Cache HTTP(S) URLs
            if parsed_url.scheme in ("http", "https"):
                image_url_lower = image_url.lower()
                # Cache the image for future use, and evict the oldest image if the cache is full
                if self._cache_queue.full():
                    oldest_image_url = self._cache_queue.get()
                    del self._image_cache[oldest_image_url]

                self._image_cache[image_url_lower] = image_converted
                self._cache_queue.put(image_url_lower)

            return image

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")

    @endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
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
            image = await self.load_image(request.image_url)

            logger.debug(f"Processing image for request: {{ id: {request_id} }}")
            image_embeds = self.image_processor(images=image, return_tensors="pt")
            # Add a batch dimension to everything
            for item in image_embeds:
                image_embeds[item] = image_embeds[item].unsqueeze(0).to(DEVICE)
            logger.debug(f"Image embeds: {image_embeds}")

            image_grid_thw = (
                image_embeds["image_grid_thw"].tolist()
                if "image_grid_thw" in image_embeds
                else None
            )
            image_sizes = (
                image_embeds["image_sizes"].tolist()
                if "image_sizes" in image_embeds
                else [image.size]
            )
            logger.debug(
                f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
            )

            with torch.no_grad():
                embeddings = self.vision_model.get_multimodal_embeddings(**image_embeds)
                if isinstance(embeddings, tuple) or isinstance(embeddings, list):
                    # The result multimodal_embeddings may be a list or tuple of tensors, with each
                    # tensor corresponding to a multimodal data item (image or video).
                    # TODO: for multi-image support, this result will contain multiple tensors.
                    embeddings = embeddings[0].unsqueeze(0)
                logger.debug(
                    f"Embeddings: {{ shape: {embeddings.shape}, dtype: {embeddings.dtype}, device: {embeddings.device}, ptr: {embeddings.data_ptr()}, elements: {{ count: {embeddings.numel()}, size: {embeddings.element_size()} }} }}."
                )

                if request.serialized_request is None:
                    logger.error(
                        f"Request serialized_request is None for request: {{ id: {request_id} }}."
                    )

                # Create a descriptor for the embeddings, this will register the memory with the connector (and the NIXL runtime).
                descriptor = connect.Descriptor(embeddings)
                # Create a write operation using the serialized request and the descriptor.
                # This will begin the RDMA transfer of the embeddings to the remote worker.
                write_op = await self._connector.begin_write(
                    descriptor,
                    request.serialized_request,
                )
                # Await for the write operation to complete.
                # This will block until the data has been written to the remote worker or an error occurs.
                await write_op.wait_for_completion()

                yield EncodeResponse(
                    request_id=request.request_id,
                    image_grid_thw=image_grid_thw,
                    image_sizes=image_sizes,
                ).model_dump_json()
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    @async_on_start
    async def async_init(self):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()
        # Initialize HTTP client with default limits
        self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        logger.info("Startup completed.")
