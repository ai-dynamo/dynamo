# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncIterator, List

import torch
from transformers import AutoImageProcessor, AutoModel

import dynamo.nixl_connect as connect
from dynamo._core import Client, Component, Context
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils import ImageLoader, encode_image_embeddings
from dynamo.sglang.protocol import SglangMultimodalRequest
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

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


class MultimodalEncodeWorkerHandler(BaseWorkerHandler):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.
    """

    def __init__(
        self,
        component: Component,
        config: Config,
        pd_worker_client: Client,
    ) -> None:
        super().__init__(component, engine=None, config=config)
        self.pd_worker_client = pd_worker_client
        self.model = config.server_args.model_path
        self.served_model_name = config.server_args.served_model_name

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = AutoModel.from_pretrained(
            self.model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.min_workers = 1

    def cleanup(self):
        pass

    async def generate(
        self, request: SglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Generate precomputed embeddings for multimodal input.
        Supports multiple images per request.

        Args:
            request: Multimodal request with image/video data.
            context: Context object for cancellation handling.
        """
        if not isinstance(request, SglangMultimodalRequest):
            if isinstance(request, str):
                request = SglangMultimodalRequest.model_validate_json(request)
            else:
                request = SglangMultimodalRequest.model_validate(request)

        # The following steps encode the requested images for SGLang:
        # 1. Open all images from the provided URLs.
        # 2. Process each image using the processor.
        # 3. Run images through the vision model to get precomputed embeddings.
        # 4. Concatenate embeddings from all images.
        # 5. Create a descriptor for the embeddings and send to downstream worker.
        # Note: Token IDs are NOT modified - the tokenizer already generates correct
        # number of image tokens based on image resolution.

        try:
            image_urls = request.multimodal_input.image_urls
            if not image_urls:
                raise ValueError("image_urls is required for the encode worker.")

            num_images = len(image_urls)
            logger.debug(f"Processing {num_images} image(s)")

            # Process all images and collect embeddings
            all_embeddings = []
            all_grid_thw = []
            all_num_tokens = []

            for idx, image_url in enumerate(image_urls):
                logger.debug(f"Processing image {idx + 1}/{num_images}: {image_url[:50]}...")

                image = await self.image_loader.load_image(image_url)
                image_embeds = self.image_processor(images=image, return_tensors="pt")

                embeddings = encode_image_embeddings(
                    model_name=self.served_model_name,
                    image_embeds=image_embeds,
                    vision_encoder=self.vision_model,
                    projector=None,
                )

                all_embeddings.append(embeddings)
                all_num_tokens.append(embeddings.shape[1])  # Number of image patches

                if "image_grid_thw" in image_embeds:
                    all_grid_thw.extend(image_embeds["image_grid_thw"].tolist())

            # Concatenate all embeddings along the sequence dimension
            # Shape: [1, total_tokens, hidden_dim]
            precomputed_embeddings = torch.cat(all_embeddings, dim=1)

            logger.debug(
                f"Total embeddings shape: {precomputed_embeddings.shape}, "
                f"from {num_images} images with token counts: {all_num_tokens}"
            )

            # Store the image data info in the request for downstream
            request.image_grid_thw = all_grid_thw
            request.embeddings_shape = tuple(precomputed_embeddings.shape)
            request.num_images = num_images
            # Store per-image token counts for unpacking on the worker side
            request.per_image_num_tokens = all_num_tokens

            # NOTE: Do NOT modify token_ids here!
            # The tokenizer has already generated the correct number of <|image_pad|> tokens
            # for each image based on its resolution. Modifying them would cause a mismatch
            # between token count and embedding count.

            # Create descriptor for the multimodal data
            descriptor = connect.Descriptor(precomputed_embeddings)

            with await self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()

                logger.debug(f"Request: {request.model_dump_json()}")

                # Get the response generator from downstream worker
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )
                await readable.wait_for_completion()

                async for response in response_generator:
                    yield response.data() if hasattr(response, "data") else str(
                        response
                    )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()

        logger.info("Startup completed.")
