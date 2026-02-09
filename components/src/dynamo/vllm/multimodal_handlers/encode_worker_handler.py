# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator

import safetensors
import torch
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.multimodal.hasher import MultiModalHasher
from vllm.sampling_params import SamplingParams

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime

from ..multimodal_utils import (
    ImageLoader,
    VLLMNativeEncoderRequest,
    VLLMNativeEncoderResponse,
    encode_image_embeddings,
    get_encoder_components,
    load_vision_model,
    vLLMMultimodalRequest,
)
from ..multimodal_utils.embedding_cache import EmbeddingCache
from ..multimodal_utils.model import is_qwen_vl_model

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
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
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
        self._connector = None
        self._accumulated_time = 0.0
        self._processed_requests = 0
        self.readables = []
        self.embedding_cache = EmbeddingCache()

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
                    safetensors.torch.save_file(
                        tensors,
                        f"/tmp/encoder_cache.{embedding_item.key}.safetensors",
                    )
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[
                        idx
                    ].serialized_request = (
                        f"/tmp/encoder_cache.{embedding_item.key}.safetensors"
                    )
                else:
                    descriptor = connect.Descriptor(embedding_item.embeddings_cpu)
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


class VLLMEncodeWorkerHandler:
    """
    Handler for vLLM-native encoder worker using ECConnector.
    """

    def __init__(self, runtime, component, engine_client, config):
        """
        Initialize the handler.

        Args:
            runtime: Dynamo distributed runtime
            component: Dynamo component instance
            engine_client: vLLM AsyncLLM instance
            config: Dynamo Config object with CLI arguments
        """
        self.runtime = runtime
        self.component = component
        self.engine_client = engine_client
        self.config = config
        self.temp_dirs = []
        self.image_loader = ImageLoader()

        logger.info(
            f"VLLMNativeEncoderWorkerHandler initialized with "
            f"backend={config.ec_connector_backend}, "
            f"storage_path={config.ec_storage_path}"
        )

    def add_temp_dir(self, temp_dir):
        """Add temporary directory for cleanup."""
        if temp_dir:
            self.temp_dirs.append(temp_dir)

    async def generate(self, request, context) -> AsyncGenerator[str, None]:
        """
        Process encoder request and trigger vLLM encoder execution.

        Args:
            request: VLLMNativeEncoderRequest with multimodal_inputs (list of MultiModalGroup)
            context: Request context from Dynamo runtime

        Yields:
            JSON-encoded VLLMNativeEncoderResponse for each processed item
        """
        # Parse request
        if not isinstance(request, VLLMNativeEncoderRequest):
            if isinstance(request, str):
                request = VLLMNativeEncoderRequest.model_validate_json(request)
            else:
                request = VLLMNativeEncoderRequest.model_validate(request)

        if not request.multimodal_inputs:
            raise ValueError("No multimodal inputs provided in request")

        logger.info(
            f"Processing {len(request.multimodal_inputs)} multimodal item(s) "
            f"for request_id={request.request_id}"
        )

        # Load all images
        # TODO: support video and audio encoding later
        media_list = []
        modality = "image"
        for idx, mm_group in enumerate(request.multimodal_inputs):
            mm_input = mm_group.multimodal_input
            if mm_input.image_url:
                media = await self.image_loader.load_image(mm_input.image_url)
                media_list.append(media)
            elif mm_input.video_url:
                raise NotImplementedError("Video encoding not yet supported")
            else:
                raise ValueError(
                    f"No media URL provided in multimodal_input[{idx}]. "
                    "Specify image_url or video_url."
                )

        # Process all images in one vLLM request
        prompt_dict = TokensPrompt(
            prompt_token_ids=request.token_ids,
            multi_modal_data={"image": media_list},
        )

        try:
            gen = self.engine_client.generate(
                prompt=prompt_dict,
                sampling_params=SamplingParams(max_tokens=1, min_tokens=0),
                request_id=request.request_id,
            )

            # Consume generator to trigger encoder execution
            async for _ in gen:
                pass

            logger.info(
                f"[{request.request_id}] Encoder execution completed for all {len(media_list)} image(s)"
            )

        except Exception as e:
            logger.error(f"[{request.request_id}] Encoder execution failed: {e}")
            raise

        # Compute mm_hash for each image and yield responses
        for idx, media in enumerate(media_list):
            item_request_id = f"{request.request_id}_mm_{idx}"

            try:
                mm_hash = MultiModalHasher.hash_kwargs(
                    model_id=self.config.model, image=media
                )
                logger.debug(f"[{item_request_id}] Computed mm_hash: {mm_hash}")
            except Exception as e:
                logger.error(f"[{item_request_id}] Failed to compute mm_hash: {e}")
                raise

            response = VLLMNativeEncoderResponse(
                request_id=item_request_id,
                mm_hash=mm_hash,
                modality=modality,
                connector_metadata={
                    "ec_connector": self.config.ec_connector_backend,
                    "storage_path": self.config.ec_storage_path,
                },
            )

            logger.debug(f"[{item_request_id}] Returning response: {response}")
            yield response.model_dump_json()

        logger.info(
            f"All {len(request.multimodal_inputs)} multimodal items processed "
            f"for request_id={request.request_id}"
        )

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up VLLMNativeEncoderWorkerHandler")

        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")
