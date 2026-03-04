# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import AsyncIterator, Optional

import torch

# MMEncoder chain imports compiled CUDA ops; may fail in CPU-only environments.
try:
    from sglang.srt.disaggregation.encode_server import MMEncoder
except (ImportError, OSError):
    MMEncoder = None  # type: ignore[assignment]
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

import dynamo.nixl_connect as connect
from dynamo._core import Client, Context
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
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


class MultimodalEncodeWorkerHandler(BaseWorkerHandler):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.
    """

    def __init__(
        self,
        config: Config,
        pd_worker_client: Client,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(engine=None, config=config, shutdown_event=shutdown_event)
        self.pd_worker_client = pd_worker_client
        self.model = config.server_args.model_path

        if MMEncoder is None:
            raise RuntimeError(
                "MMEncoder is not available. "
                "Multimodal encode worker requires a CUDA environment."
            )

        # torch.distributed requires a dist_init_method even for tp=1;
        # port 0 lets the OS assign a free port.
        self.encoder = MMEncoder(
            server_args=config.server_args,
            dist_init_method="tcp://127.0.0.1:0",
            rank=0,
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
        # 1. Pass the image URL to MMEncoder which loads, preprocesses, and
        #    runs the vision encoder.
        # 2. Add a batch dimension and store metadata on the request.
        # 3. Expand the single image placeholder token to match patch count.
        # 4. Create a NIXL descriptor and send embeddings to downstream worker.
        # 5. Stream the downstream worker's response back to the caller.

        try:
            if not request.multimodal_input.image_url:
                raise ValueError("image_url is required for the encode worker.")

            image_grid_dim, mm_embedding = await self.encoder._encode(
                [request.multimodal_input.image_url]
            )

            image_grid_thw = (
                image_grid_dim.tolist()
                if isinstance(image_grid_dim, torch.Tensor)
                else image_grid_dim
            )

            # Store the image data info in the request for downstream
            request.processor_output = {"image_grid_thw": image_grid_thw}
            request.image_grid_thw = image_grid_thw
            request.embeddings_shape = tuple(mm_embedding.shape)

            # Replace the single image token with multiple image tokens based on embedding shape
            image_token_id_index = request.request.token_ids.index(self.image_token_id)

            num_image_tokens = mm_embedding.shape[0]  # Number of image patches

            # Replace single image token with multiple image tokens
            request.request.token_ids = (
                request.request.token_ids[:image_token_id_index]
                + [self.image_token_id] * num_image_tokens
                + request.request.token_ids[
                    image_token_id_index + 1 :
                ]  # Skip the original token
            )

            # Create descriptor for the multimodal data
            descriptor = connect.Descriptor(mm_embedding)

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
