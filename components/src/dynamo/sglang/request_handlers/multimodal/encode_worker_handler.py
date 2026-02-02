# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncIterator, List, Tuple

import torch
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

import dynamo.nixl_connect as connect
from dynamo._core import Client, Component, Context
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils import ImageLoader, encode_image_embeddings
from dynamo.sglang.protocol import MultiModalInputGroup, SglangMultimodalRequest
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

        # Load tokenizer to convert image token string to integer ID
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )

        # Get image token string and handle it properly
        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )

        # For Qwen2.5-VL, the image token might be multiple tokens
        if image_token_str == "<|vision_start|><|image_pad|><|vision_end|>":
            # These are likely the individual special tokens for Qwen2.5-VL
            image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

            # Use the image_pad token as the main image token
            self.image_token_id = image_pad_id
        else:
            # Fallback for other models
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token_str)

        self.min_workers = 1
        self.readables = []  # Store NIXL readables (matches vLLM pattern)

    def cleanup(self):
        pass

    async def generate(
        self, request: SglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Generate precomputed embeddings for multimodal inputs.

        Supports multiple images in a single request. Each image is processed
        individually, and tokens are expanded for each image's embeddings.

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
        # 1. Open each image from the provided URLs.
        # 2. Process each image using the processor.
        # 3. Run each image through the vision model to get precomputed embeddings.
        # 4. Find all image token positions and expand them based on embedding shapes.
        # 5. Create NIXL descriptors for each embedding and send to downstream worker.

        if not request.multimodal_inputs:
            raise ValueError("At least one multimodal input is required.")

        # Find all image token positions in the token sequence
        token_ids = request.request.token_ids
        image_token_positions = [
            i for i, t in enumerate(token_ids) if t == self.image_token_id
        ]

        num_images = len(request.multimodal_inputs)
        if len(image_token_positions) != num_images:
            raise ValueError(
                f"Mismatch: found {len(image_token_positions)} image tokens "
                f"but {num_images} multimodal inputs"
            )

        logger.debug(f"Processing {num_images} image(s)")

        # Process each image and compute embeddings
        embeddings_list: List[Tuple[torch.Tensor, MultiModalInputGroup]] = []
        for idx, mm_group in enumerate(request.multimodal_inputs):
            if not mm_group.multimodal_input or not mm_group.multimodal_input.image_url:
                raise ValueError(f"image_url is required for multimodal input {idx}.")

            image = await self.image_loader.load_image(
                mm_group.multimodal_input.image_url
            )

            image_embeds = self.image_processor(images=image, return_tensors="pt")
            precomputed_embeddings = encode_image_embeddings(
                model_name=self.served_model_name,
                image_embeds=image_embeds,
                vision_encoder=self.vision_model,
                projector=None,
            )

            image_grid_thw = (
                image_embeds["image_grid_thw"].tolist()
                if "image_grid_thw" in image_embeds
                else None
            )

            # Update group with computed metadata
            mm_group.image_grid_thw = image_grid_thw
            mm_group.embeddings_shape = tuple(precomputed_embeddings.shape)
            embeddings_list.append((precomputed_embeddings, mm_group))

        # Expand image tokens in reverse order to preserve earlier indices
        for idx in range(num_images - 1, -1, -1):
            pos = image_token_positions[idx]
            num_patches = embeddings_list[idx][0].shape[1]
            token_ids = (
                token_ids[:pos]
                + [self.image_token_id] * num_patches
                + token_ids[pos + 1 :]
            )

        request.request.token_ids = token_ids

        # Create NIXL readables for each embedding (matches vLLM multi-image pattern)
        for embeddings, mm_group in embeddings_list:
            # Move embeddings to CPU for NIXL transfer (RDMA requires host memory)
            embeddings_cpu = embeddings.cpu()
            descriptor = connect.Descriptor(embeddings_cpu)
            self.readables.append(await self._connector.create_readable(descriptor))
            mm_group.serialized_request = self.readables[-1].metadata()

        # Send request to downstream worker
        response_generator = await self.pd_worker_client.round_robin(
            request.model_dump_json()
        )

        # Yield responses
        async for response in response_generator:
            yield response.data() if hasattr(response, "data") else str(response)

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()

        logger.info("Startup completed.")
