# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video generation request handler for TensorRT-LLM backend.

This handler processes video generation requests using diffusion models.
It handles MediaOutput from TensorRT-LLM's visual_gen pipelines, which
can contain video, image, and/or audio tensors depending on the model.
"""

import asyncio
import base64
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from dynamo._core import Context
from dynamo.common.protocols.image_protocol import (
    ImageData,
    ImageNvExt,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.common.storage import get_fs, upload_to_fs
from dynamo.common.utils.image_utils import encode_to_png_bytes
from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine
from dynamo.trtllm.request_handlers.base_generative_handler import BaseGenerativeHandler

logger = logging.getLogger(__name__)


class ImageGenerationHandler(BaseGenerativeHandler):
    """Handler for image generation requests.

    This handler receives generation requests, runs the diffusion pipeline
    via DiffusionEngine, encodes the output to the appropriate media format,
    and returns the media URL or base64-encoded data.

    Supports MediaOutput with:
    - video: logged as unsupported (use an image handler instead)
    - image: torch.Tensor (H, W, 3) uint8
    - audio: logged (future: mux into MP4)

    Inherits from BaseGenerativeHandler to share the common interface with
    LLM handlers.
    """

    def __init__(
        self,
        engine: DiffusionEngine,
        config: DiffusionConfig,
    ):
        """Initialize the handler.

        Args:
            engine: The DiffusionEngine instance.
            config: Diffusion generation configuration.
        """
        self.engine = engine
        self.config = config
        if not config.media_output_fs_url:
            raise ValueError(
                "media_output_fs_url must be set; use --media-output-fs-url or DYN_MEDIA_OUTPUT_FS_URL."
            )
        self.media_output_fs = get_fs(config.media_output_fs_url)
        self.media_output_http_url = config.media_output_http_url
        # Serialize pipeline access — the diffusion pipeline is not thread-safe
        # (mutable instance state, unprotected CUDA graph cache).
        # asyncio.Lock suspends waiting coroutines cooperatively so the event
        # loop stays free for health checks and signal handling.
        self._generate_lock = asyncio.Lock()

    def _parse_size(self, size: Optional[str]) -> tuple[int, int]:
        """Parse 'WxH' string to (width, height) tuple.

        The API accepts size as a string (e.g., "832x480") to match the format
        used by OpenAI's image generation API (/v1/images/generations).
        This method converts that string to a (width, height) tuple for the engine.

        Args:
            size: Size string in 'WxH' format (e.g., '832x480').

        Returns:
            Tuple of (width, height).

        Raises:
            ValueError: If dimensions exceed configured max_width/max_height.
        """
        if not size:
            width, height = self.config.default_width, self.config.default_height
        else:
            try:
                w, h = size.split("x")
                width, height = int(w), int(h)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format: {size}, using defaults")
                width, height = self.config.default_width, self.config.default_height

        # Validate dimensions to prevent OOM
        self._validate_dimensions(width, height)
        return width, height

    def _validate_dimensions(self, width: int, height: int) -> None:
        """Validate that dimensions don't exceed configured limits.

        Args:
            width: Requested width in pixels.
            height: Requested height in pixels.

        Raises:
            ValueError: If width or height exceeds the configured maximum.
        """
        errors = []
        if not (1 <= width <= self.config.max_width):
            errors.append(f"width {width} must be in [1, {self.config.max_width}]")
        if not (1 <= height <= self.config.max_height):
            errors.append(f"height {height} must be in [1, {self.config.max_height}]")

        if errors:
            raise ValueError(
                f"Requested dimensions out of range: {', '.join(errors)}. "
                f"This is a safety check to prevent out-of-memory errors. "
                f"To allow larger sizes, increase --max-width and/or --max-height."
            )

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate video/image from request.

        This is the main entry point called by Dynamo's endpoint.serve_endpoint().

        Handles MediaOutput from the pipeline:
        - video tensor → unsupported (raises error)
        - image tensor → PNG
        - audio tensor → unsupported (raises error)

        Args:
            request: Request dictionary with generation parameters.
            context: Dynamo context for request tracking.

        Yields:
            Response dictionary with generated media data.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"Received generation request: {request_id}")

        # Parse request
        req = NvCreateImageRequest(**request)
        nvext = req.nvext or ImageNvExt()

        # Parse parameters
        width, height = self._parse_size(req.size)
        num_images_per_prompt = (
            req.n if req.n is not None else self.config.default_num_images_per_prompt
        )
        num_inference_steps = (
            nvext.num_inference_steps
            if nvext.num_inference_steps is not None
            else self.config.default_num_inference_steps
        )
        guidance_scale = (
            nvext.guidance_scale
            if nvext.guidance_scale is not None
            else self.config.default_guidance_scale
        )

        logger.info(
            f"Request {request_id}: prompt='{req.prompt[:50]}...', "
            f"size={width}x{height}, images={num_images_per_prompt}, steps={num_inference_steps}"
        )

        # Run generation in thread pool (blocking operation).
        # Lock ensures only one request uses the pipeline at a time.
        # TODO: Add cancellation support. This requires:
        # 1. The pipeline to expose a cancellation hook in the denoising loop
        # 2. Passing a cancellation token/event to engine.generate()
        # 3. Checking context.cancelled() and propagating to the pipeline
        async with self._generate_lock:
            output = await asyncio.to_thread(
                self.engine.generate,
                prompt=req.prompt,
                negative_prompt=nvext.negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=nvext.seed,
            )

        if output is None:
            raise RuntimeError("Pipeline returned no output (MediaOutput is None)")

        # Determine output format
        response_format = req.response_format or "url"

        # Encode media based on what the pipeline returned
        if output.image is not None:
            # MediaOutput.image is (B, H, W, C) uint8 since TRT-LLM rc9;
            images = output.image
            assert (
                images.ndim == 4 and images.shape[3] == 3
            ), f"Expected image shape (B, H, W, C), got {images.shape}"
            # [gluo FIXME] currently only take the first image but the protocol supports multiple images
            # verify if TRT-LLM will generate multiple images, relax this constraint if that's the case
            image_np = images[0].cpu().numpy()
            logger.info(
                f"Request {request_id}: encoding image output "
                f"(shape={image_np.shape}) to PNG"
            )
            image_bytes = await asyncio.to_thread(encode_to_png_bytes, image_np)

        elif output.video is not None:
            raise RuntimeError(
                "Pipeline returned video-only output, but this handler "
                "only supports image. Use a video generation handler instead."
            )

        # Log audio if present (unsupported)
        elif output.audio is not None:
            raise RuntimeError(
                "Pipeline returned audio-only output, but this handler "
                "only supports image. Use an audio generation handler instead."
            )

        else:
            raise RuntimeError(
                "Pipeline returned MediaOutput with no video or image or audio data. "
                f"MediaOutput fields: video={output.video is not None}, "
                f"image={output.image is not None}, audio={output.audio is not None}"
            )

        # Return media via URL or base64
        if response_format == "url":
            storage_path = f"images/{request_id}.png"
            image_url = await upload_to_fs(
                self.media_output_fs,
                storage_path,
                image_bytes,
                self.media_output_http_url,
            )
            image_data = ImageData(url=image_url)
        else:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            image_data = ImageData(b64_json=b64_image)

        inference_time = time.time() - start_time

        response = NvImagesResponse(
            created=int(time.time()),
            data=[image_data],
        )

        logger.info(f"Request {request_id} completed in {inference_time:.2f}s")

        yield response.model_dump()

    def cleanup(self) -> None:
        """Cleanup handler resources."""
        logger.info("ImageGenerationHandler cleanup")
