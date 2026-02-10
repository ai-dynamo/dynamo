# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncIterator

import torch
from sglang.srt.disaggregation.encode_server import MMEncoder
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

import dynamo.nixl_connect as connect
from dynamo._core import Client, Component, Context
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config, reserve_free_port
from dynamo.sglang.protocol import SglangMultimodalRequest
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

logger = logging.getLogger(__name__)


class MultimodalEncodeWorkerHandler(BaseWorkerHandler):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.

    Uses SGLang's MMEncoder for model-agnostic vision encoding, replacing
    manual HuggingFace model loading and feature extraction.
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

        # Initialize MMEncoder from SGLang for model-agnostic vision encoding.
        # Reserve a free port for the NCCL distributed init (required by
        # MMEncoder even for single-GPU / tp=1 encode workers).
        server_args = config.server_args
        with reserve_free_port() as port:
            nccl_port = port
        dist_init_method = f"tcp://127.0.0.1:{nccl_port}"

        self.encoder = MMEncoder(
            server_args=server_args,
            dist_init_method=dist_init_method,
            rank=0,
        )

        # Resolve image token ID for manual token expansion.
        # MMEncoder handles vision encoding but does not produce token IDs,
        # so we still need the tokenizer to map the image placeholder token.
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )

        # For Qwen VL models, the image token is a composite of special tokens:
        # <|vision_start|><|image_pad|><|vision_end|>
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

        Uses SGLang's MMEncoder._encode() for model-agnostic vision encoding:
        1. Pass the image URL to MMEncoder which loads, processes, and encodes.
        2. Transfer the resulting embeddings via NIXL to the downstream worker.
        3. Expand token IDs to match the number of image patches.

        Args:
            request: Multimodal request with image/video data.
            context: Context object for cancellation handling.
        """
        if not isinstance(request, SglangMultimodalRequest):
            if isinstance(request, str):
                request = SglangMultimodalRequest.model_validate_json(request)
            else:
                request = SglangMultimodalRequest.model_validate(request)

        try:
            if not request.multimodal_input.image_url:
                raise ValueError("image_url is required for the encode worker.")

            # Use SGLang's MMEncoder for model-agnostic vision encoding.
            # _encode() handles image loading, HF image processing, vision
            # model forward pass, and deepstack feature concatenation internally.
            image_grid_dim, mm_embedding = await self.encoder._encode(
                [request.multimodal_input.image_url]
            )

            # mm_embedding is (n, d) on CPU. Add batch dim for NIXL transfer
            # so the downstream worker can allocate a matching buffer.
            precomputed_embeddings = mm_embedding.unsqueeze(0)  # (1, n, d)

            # Build JSON-serializable processor output for downstream.
            # This dict becomes the base of the mm_item with
            # format="processor_output" — SGLang uses image_grid_thw for
            # mRoPE and skips the vision encoder when precomputed_embeddings
            # is present.
            image_grid_thw = (
                image_grid_dim.tolist()
                if isinstance(image_grid_dim, torch.Tensor)
                else image_grid_dim
            )
            request.processor_output = {"image_grid_thw": image_grid_thw}
            request.image_grid_thw = image_grid_thw
            request.embeddings_shape = tuple(precomputed_embeddings.shape)

            # Expand the single <|image_pad|> token to match the number of
            # image patches produced by the vision encoder.
            num_image_tokens = mm_embedding.shape[0]
            image_token_id_index = request.request.token_ids.index(self.image_token_id)
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
