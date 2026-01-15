# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time
from typing import AsyncGenerator, AsyncIterator

import safetensors
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TextPrompt
from vllm.multimodal.hasher import MultiModalHasher
from vllm.sampling_params import SamplingParams

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime

from ..multimodal_utils import (
    ImageLoader,
    VLLMNativeEncoderRequest,
    VLLMNativeEncoderResponse,
    encode_image_embeddings,
    get_embedding_hash,
    get_encoder_components,
    load_vision_model,
    vLLMMultimodalRequest,
)

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
        self.cached_embeddings = {}

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
            for idx in range(len(request.multimodal_inputs)):
                if not request.multimodal_inputs[idx].multimodal_input.image_url:
                    raise ValueError("image_url is required for the encode worker.")

                image_url = request.multimodal_inputs[idx].multimodal_input.image_url
                # see if we have local cache
                if image_url in self.cached_embeddings:
                    (
                        embedding_key,
                        image_grid_thw,
                        embeddings_shape,
                    ) = self.cached_embeddings[image_url]
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[
                        idx
                    ].serialized_request = (
                        f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    request.multimodal_inputs[idx].multimodal_input.image_url = None
                    request.multimodal_inputs[idx].image_grid_thw = image_grid_thw
                    request.multimodal_inputs[idx].embeddings_shape = embeddings_shape
                    continue

                image = await self.image_loader.load_image(image_url)

                logger.debug(
                    f"Processing image {image_url} for request: {{ id: {request_id} }}"
                )
                image_embeds = self.image_processor(images=image, return_tensors="pt")

                # Encode the image embeddings using model-specific encoder
                embeddings = encode_image_embeddings(
                    model_name=self.model,
                    image_embeds=image_embeds,
                    vision_encoder=self.vision_encoder,
                    projector=self.projector,
                )

                image_grid_thw = (
                    image_embeds["image_grid_thw"].tolist()
                    if "image_grid_thw" in image_embeds
                    else None
                )
                logger.debug(
                    f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
                )

                # Move embeddings to CPU for NIXL transfer to avoid UCX/InfiniBand issues
                embeddings_cpu = embeddings.cpu()

                request.multimodal_inputs[idx].image_grid_thw = image_grid_thw
                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    embeddings.shape
                )

                if TRANSFER_LOCAL:
                    embedding_key = get_embedding_hash(image_url)
                    logger.debug(
                        f"ENCODER: saving local safetensors file with key {embedding_key}, {embeddings_cpu.numel()} * {embeddings_cpu.element_size()} bytes"
                    )
                    tensors = {"ec_cache": embeddings_cpu}
                    safetensors.torch.save_file(
                        tensors, f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    # [gluo FIXME] need mechanism to clean up local files
                    request.multimodal_inputs[
                        idx
                    ].serialized_request = (
                        f"/tmp/encoder_cache.{embedding_key}.safetensors"
                    )
                    self.cached_embeddings[image_url] = (
                        embedding_key,
                        request.multimodal_inputs[idx].image_grid_thw,
                        request.multimodal_inputs[idx].embeddings_shape,
                    )
                else:
                    # [gluo FIXME] nixl_connector path needs to be update to handle multiple embeddings
                    descriptor = connect.Descriptor(embeddings_cpu)
                    self.readables.append(
                        await self._connector.create_readable(descriptor)
                    )
                    request.multimodal_inputs[idx].serialized_request = self.readables[
                        -1
                    ].metadata()

                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_inputs[idx].multimodal_input.image_url = None

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
            request: VLLMNativeEncoderRequest with multimodal_input
            context: Request context from Dynamo runtime

        Yields:
            JSON-encoded VLLMNativeEncoderResponse with mm_hash and connector metadata
        """
        # Parse request
        if not isinstance(request, VLLMNativeEncoderRequest):
            if isinstance(request, str):
                request = VLLMNativeEncoderRequest.model_validate_json(request)
            else:
                request = VLLMNativeEncoderRequest.model_validate(request)

        # Load media (image/video/audio)
        # TODO: Add support for video_url and audio
        if request.multimodal_input.image_url:
            media = await self.image_loader.load_image(
                request.multimodal_input.image_url
            )
            media_key = "image"
        else:
            raise ValueError(
                "No media URL provided. Specify image_url in multimodal_input."
            )

        # Compute mm_hash using vLLM's hasher
        try:
            mm_hash = MultiModalHasher.hash_kwargs(
                model_id=self.config.model, **{media_key: media}
            )
            logger.debug(f"Computed mm_hash: {mm_hash}")
        except Exception as e:
            logger.error(f"Failed to compute mm_hash: {e}")
            raise

        try:
            # Prompt can be a random string as the encoder is only interested in the multimodal data
            prompt_dict = TextPrompt(
                prompt=request.prompt, multi_modal_data={media_key: media}
            )

            gen = self.engine_client.generate(
                prompt=prompt_dict,
                sampling_params=SamplingParams(max_tokens=1, min_tokens=0),
                request_id=request.request_id,
            )

            # Consume generator to trigger encoder execution
            async for _ in gen:
                pass

            logger.info(
                f"Encoder execution completed for request_id={request.request_id}"
            )

        except Exception as e:
            logger.error(f"Encoder execution failed: {e}")
            raise

        # Return metadata for PD workers
        response = VLLMNativeEncoderResponse(
            request_id=request.request_id,
            mm_hash=mm_hash,
            modality=request.modality,
            connector_metadata={
                "ec_connector": self.config.ec_connector_backend,
                "storage_path": self.config.ec_storage_path,
            },
        )

        logger.debug(f"Returning response: {response}")
        yield response.model_dump_json()

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
