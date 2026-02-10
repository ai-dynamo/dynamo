# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncIterator

from sglang.srt.parser.conversation import chat_templates
from transformers import AutoProcessor

import dynamo.nixl_connect as connect
from dynamo._core import Client, Component, Context
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils import ImageLoader, encode_image_embeddings
from dynamo.sglang.multimodal_utils.multimodal_encode_utils import load_vision_model
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

        # AutoProcessor bundles image_processor + tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model(self.model)

        # Resolve image token ID for manual token expansion
        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )

        # For Qwen VL models, the image token is a composite of special tokens:
        # <|vision_start|><|image_pad|><|vision_end|>
        tokenizer = self.processor.tokenizer
        if image_token_str == "<|vision_start|><|image_pad|><|vision_end|>":
            self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        else:
            self.image_token_id = tokenizer.convert_tokens_to_ids(image_token_str)

        self.min_workers = 1

    def cleanup(self):
        pass

    async def generate(
        self, request: SglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Generate precomputed embeddings for multimodal input.

        Args:
            request: Multimodal request with image/video data.
            context: Context object for cancellation handling.
        """
        if not isinstance(request, SglangMultimodalRequest):
            if isinstance(request, str):
                request = SglangMultimodalRequest.model_validate_json(request)
            else:
                request = SglangMultimodalRequest.model_validate(request)

        # The following steps encode the requested image for SGLang:
        # 1. Open the image from the provided URL.
        # 2. Process the image using the processor (which handles tokenization).
        # 3. Extract input_ids and image data from processed result.
        # 4. Run the image through the vision model to get precomputed embeddings.
        # 5. Create SGLang-specific multimodal data format.
        # 6. Create a descriptor for the embeddings and send to downstream worker.

        try:
            if not request.multimodal_input.image_url:
                raise ValueError("image_url is required for the encode worker.")

            image = await self.image_loader.load_image(
                request.multimodal_input.image_url
            )

            # Process image through the HF image processor
            image_data = self.processor.image_processor(
                images=image, return_tensors="pt"
            )
            precomputed_embeddings = encode_image_embeddings(
                model_name=self.served_model_name,
                image_embeds=image_data,
                vision_encoder=self.vision_model,
                projector=None,
            )

            # Build JSON-serializable processor output for downstream.
            # This dict becomes the base of the mm_item with
            # format="processor_output" — SGLang uses image_grid_thw for
            # mRoPE and skips the vision encoder when precomputed_embeddings
            # is present.
            image_grid_thw = (
                image_data["image_grid_thw"].tolist()
                if "image_grid_thw" in image_data
                else None
            )
            request.processor_output = {"image_grid_thw": image_grid_thw}
            request.image_grid_thw = image_grid_thw
            request.embeddings_shape = tuple(precomputed_embeddings.shape)

            # Expand the single <|image_pad|> token to match the number of
            # image patches produced by the vision encoder.
            image_token_id_index = request.request.token_ids.index(self.image_token_id)
            num_image_tokens = precomputed_embeddings.shape[1]
            request.request.token_ids = (
                request.request.token_ids[:image_token_id_index]
                + [self.image_token_id] * num_image_tokens
                + request.request.token_ids[image_token_id_index + 1 :]
            )

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
