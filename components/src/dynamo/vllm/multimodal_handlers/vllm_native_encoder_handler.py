# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-Native Encoder Worker Handler (ECConnector Mode Only)

This handler is a minimal wrapper around vLLM's native encoder execution.
When configured as a producer (ec_role=ec_producer), vLLM automatically:
1. Executes multimodal encoder via _execute_mm_encoder()
2. Caches outputs by mm_hash in encoder_cache
3. Saves to ECConnector storage via save_caches()
4. Returns empty output (no text generation)

The handler only needs to:
1. Load media (image/video/audio) from URL
2. Call engine_client.generate() with multimodal data
3. Return metadata (mm_hash, shape, modality) for PD workers
"""

import logging
import shutil
from typing import AsyncGenerator

from vllm.inputs.data import TokensPrompt
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
    
    This is a minimal wrapper that triggers vLLM's encoder execution.
    vLLM handles all the heavy lifting: encoding, caching, and storage.
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

    async def generate(
        self, request, context
    ) -> AsyncGenerator[str, None]:
        """
        Process encoder request and trigger vLLM encoder execution.

        vLLM (configured as producer) will:
        1. Execute encoder via _execute_mm_encoder()
        2. Cache by mm_hash in encoder_cache
        3. Save to ECConnector storage
        4. Return empty output (no text generation)

        Args:
            request: VLLMNativeEncoderRequest or JSON string
            context: Request context from Dynamo runtime

        Yields:
            JSON-encoded VLLMNativeEncoderResponse with metadata
        """
        # Parse request
        if not isinstance(request, VLLMNativeEncoderRequest):
            if isinstance(request, str):
                request = VLLMNativeEncoderRequest.model_validate_json(request)
            else:
                request = VLLMNativeEncoderRequest.model_validate(request)

        logger.info(
            f"Processing encoder request: request_id={request.request_id}, "
            f"modality={request.modality}"
        )

        # Load media (image/video/audio)
        # For now, we only support images via image_url
        # TODO: Add support for video_url and audio when needed
        if request.multimodal_input.image_url:
            media = await self.image_loader.load_image(
                request.multimodal_input.image_url
            )
            media_key = "image"
        else:
            raise ValueError(
                "No media URL provided. Specify image_url in multimodal_input."
            )

        # Compute mm_hash using vLLM's hasher (BEFORE calling vLLM)
        # This ensures consistency with vLLM's internal hash computation
        try:
            mm_hash = MultiModalHasher.hash_kwargs(
                model_id=self.config.model,
                **{media_key: media}
            )
            logger.debug(f"Computed mm_hash: {mm_hash}")
        except Exception as e:
            logger.error(f"Failed to compute mm_hash: {e}")
            raise

        # Call vLLM generate with multimodal data
        # vLLM will automatically run encoder and save to ECConnector
        try:
            gen = self.engine_client.generate(
                prompt=TokensPrompt(
                    prompt_token_ids=[],  # Empty tokens for encoder-only
                    multi_modal_data={media_key: media}  # vLLM processes this
                ),
                sampling_params=SamplingParams(
                    max_tokens=0,  # Encoder-only, no text generation
                    min_tokens=0
                ),
                request_id=request.request_id
            )

            # Consume generator to trigger encoder execution
            async for _ in gen:
                pass

            logger.info(f"Encoder execution completed for request_id={request.request_id}")

        except Exception as e:
            logger.error(f"Encoder execution failed: {e}")
            raise

        # TODO: Get actual embeddings shape from vLLM instead of hardcoded value
        # For now, using typical Llama 3.2 Vision shape as placeholder
        embeddings_shape = (1, 576, 4096)

        # Return metadata for PD workers
        response = VLLMNativeEncoderResponse(
            request_id=request.request_id,
            mm_hash=mm_hash,
            modality=request.modality,
            embeddings_shape=embeddings_shape,
            connector_metadata={
                "ec_connector": self.config.ec_connector_backend,
                "storage_path": self.config.ec_storage_path
            }
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

