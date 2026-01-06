# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import shutil
from typing import AsyncGenerator

from vllm.inputs.data import TextPrompt
from vllm.multimodal.hasher import MultiModalHasher
from vllm.sampling_params import SamplingParams

from dynamo.vllm.multimodal_utils.image_loader import ImageLoader
from dynamo.vllm.multimodal_utils.protocol import (
    VLLMNativeEncoderRequest,
    VLLMNativeEncoderResponse,
)

logger = logging.getLogger(__name__)


class VLLMNativeEncoderWorkerHandler:
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
                prompt="<image>", multi_modal_data={media_key: media}
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
