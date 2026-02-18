# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import AsyncIterator

import safetensors
import torch
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.runtime import DistributedRuntime

from ..multimodal_utils import (
    ImageLoader,
    encode_image_embeddings,
    get_encoder_components,
    load_vision_model,
    vLLMMultimodalRequest,
)
from ..multimodal_utils.embedding_cache import EmbeddingCache
from ..multimodal_utils.model import is_qwen_vl_model

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8

TRANSFER_LOCAL = int(os.getenv("TRANSFER_LOCAL", 1))


@dataclass
class EmbeddingItem:
    key: str
    image_grid_thw: list
    embeddings_cpu: torch.Tensor


class EncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
    ) -> None:
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model(self.model)
        self.min_workers = 1

        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )
        self._connector: connect.Connector | None = None
        self._accumulated_time = 0.0
        self._processed_requests = 0
        self.readables = []
        self.embedding_cache = EmbeddingCache()

        # Use system temp directory for encoder cache files
        self._cache_dir = os.path.join(tempfile.gettempdir(), "encoder_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

    def cleanup(self):
        pass

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize the connector for RDMA transfers"""
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        logger.info("Encode worker startup completed.")

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
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
            time_start = time.perf_counter()
            # Before batch process images, check cache first
            need_encode_indexes = []
            embedding_lists = [None] * len(request.multimodal_inputs)
            for idx in range(len(request.multimodal_inputs)):
                if not request.multimodal_inputs[idx].multimodal_input.image_url:
                    raise ValueError("image_url is required for the encode worker.")

                image_url = request.multimodal_inputs[idx].multimodal_input.image_url
                # see if we have local cache
                embedding_key = self.embedding_cache.generate_hash_key(image_url)
                if self.embedding_cache.has_key(embedding_key):
                    (image_grid_thw, embeddings_cpu) = self.embedding_cache.get(
                        embedding_key
                    )
                    embedding_lists[idx] = EmbeddingItem(
                        embedding_key, image_grid_thw, embeddings_cpu
                    )
                # compute
                else:
                    # keep track of key to avoid recompute of it
                    need_encode_indexes.append((idx, embedding_key))

            # Load and generate image tensors
            image_futures = []
            image_to_load = []
            for idx, _ in need_encode_indexes:
                url = request.multimodal_inputs[idx].multimodal_input.image_url
                image_futures.append(self.image_loader.load_image(url))
                image_to_load.append(url)
            results = await asyncio.gather(*image_futures, return_exceptions=True)
            loaded_images = []
            collective_exceptions = ""
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    url = image_to_load[i]
                    logger.error(f"Failed to load image from {url[:80]}...: {result}")
                    collective_exceptions += (
                        f"Failed to load image from {url[:80]}...: {result}\n"
                    )
                    continue
                loaded_images.append(result)
            if collective_exceptions:
                raise ValueError(
                    f"Errors occurred during image loading:\n{collective_exceptions}"
                )

            if loaded_images:
                image_embeds = self.image_processor(
                    images=loaded_images, return_tensors="pt"
                )

                # Encode the image embeddings using model-specific encoder
                embeddings = await asyncio.to_thread(
                    encode_image_embeddings,
                    model_name=self.model,
                    image_embeds=image_embeds,
                    vision_encoder=self.vision_encoder,
                    projector=self.projector,
                )

                # [gluo FIXME] This is specific to qwen vision processing..
                # Split concatenated embeddings for each image item.
                if is_qwen_vl_model(self.model):
                    merge_size = self.vision_encoder.spatial_merge_size
                    sizes = (
                        image_embeds["image_grid_thw"].prod(-1)
                        // merge_size
                        // merge_size
                    ).tolist()
                    splitted_embeddings = embeddings.cpu().squeeze(0).split(sizes)
                    logger.debug(
                        f"Splitted embeddings lengths: {[e.shape for e in splitted_embeddings]}"
                    )
                else:
                    # Validated on llava (NOTE need to double check on other models) that the
                    # embeddings already has batch dimension for images, so we can directly
                    # split by batch dimension
                    logger.debug(f"image embedding shape: {embeddings.shape}")
                    splitted_embeddings = embeddings.cpu()

                image_grid_thw = (
                    image_embeds["image_grid_thw"].tolist()
                    if "image_grid_thw" in image_embeds
                    else None
                )

            # fill in the embedding_lists with new computed embeddings and cache them
            for split_idx, (list_idx, key) in enumerate(need_encode_indexes):
                embedding_lists[list_idx] = EmbeddingItem(
                    key,
                    [image_grid_thw[split_idx]] if image_grid_thw else None,
                    splitted_embeddings[split_idx].unsqueeze(0),
                )
                # Cache the computed value for future use
                self.embedding_cache.set(
                    embedding_lists[list_idx].key,
                    (
                        embedding_lists[list_idx].image_grid_thw,
                        embedding_lists[list_idx].embeddings_cpu,
                    ),
                )

            for idx, embedding_item in enumerate(embedding_lists):
                # Update request for transfer metadata
                request.multimodal_inputs[idx].multimodal_input.image_url = None
                request.multimodal_inputs[
                    idx
                ].image_grid_thw = embedding_item.image_grid_thw
                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    embedding_item.embeddings_cpu.shape
                )

                # Prepare transfer
                if TRANSFER_LOCAL:
                    logger.debug(
                        f"ENCODER: saving local safetensors file with key {embedding_item.key}, {embedding_item.embeddings_cpu.numel()} * {embedding_item.embeddings_cpu.element_size()} bytes"
                    )
                    tensors = {"ec_cache": embedding_item.embeddings_cpu}
                    cache_path = os.path.join(
                        self._cache_dir, f"{embedding_item.key}.safetensors"
                    )
                    safetensors.torch.save_file(tensors, cache_path)
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[idx].serialized_request = cache_path
                else:
                    descriptor = connect.Descriptor(embedding_item.embeddings_cpu)
                    assert (
                        self._connector is not None
                    ), "Connector not initialized; call async_init() first"
                    self.readables.append(
                        await self._connector.create_readable(descriptor)
                    )
                    request.multimodal_inputs[idx].serialized_request = self.readables[
                        -1
                    ].metadata()

            logger.debug(f"Request: {request.model_dump_json()}")

            time_end = time.perf_counter()
            self._accumulated_time += time_end - time_start
            self._processed_requests += 1
            logger.debug(
                f"Encoded image(s) for request {{ id: {request_id} }} in {time_end - time_start:.4f} seconds. "
                f"Average encoding time: {self._accumulated_time / self._processed_requests:.4f} seconds over {self._processed_requests} requests."
            )

            # Yield transformed request back
            yield request.model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise
