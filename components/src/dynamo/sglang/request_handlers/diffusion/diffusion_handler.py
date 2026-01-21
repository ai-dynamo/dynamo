# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import torch
from PIL import Image
from sglang.multimodal_gen import DiffGenerator

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import CreateImageRequest, ImageData, ImagesResponse
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)


class DiffusionWorkerHandler(BaseGenerativeHandler):
    """Handler for diffusion image generation.

    Inherits from BaseGenerativeHandler for common infrastructure like
    tracing, metrics publishing, and cancellation support.
    """

    def __init__(
        self,
        component: Component,
        generator: Any,  # DiffGenerator, not sgl.Engine
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        s3_client: Any = None,  # For S3 uploads
    ):
        """Initialize diffusion worker handler.

        Args:
            component: The Dynamo runtime component.
            generator: The SGLang DiffGenerator instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher (not used for diffusion currently).
            s3_client: Optional S3 client for image storage.
        """
        # Call parent constructor for common setup
        super().__init__(component, config, publisher)

        # Diffusion-specific initialization
        self.generator = generator  # DiffGenerator, not Engine
        self.s3_client = s3_client
        self.s3_bucket = config.dynamo_args.diffusion_s3_bucket
        logger.info("Diffusion worker handler initialized")

    def cleanup(self) -> None:
        """Cleanup generator resources"""
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        logger.info("Diffusion generator cleanup complete")
        # Call parent cleanup for any base class cleanup
        super().cleanup()

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate image(s) from text prompt.

        Unlike LLM streaming, diffusion returns complete image(s) at end.

        Args:
            request: Request dict with prompt and generation parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dict with generated images (OpenAI-compatible format).
        """
        logger.debug(f"Diffusion request: {request}")

        # Get trace header for distributed tracing (for logging/observability)
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug(f"Diffusion request with trace: {trace_header}")

        try:
            # Check for cancellation before starting generation
            if await self._check_cancellation(context):
                logger.info(f"Request cancelled before generation: {context.id()}")
                return

            req = CreateImageRequest(**request)

            # Parse size
            width, height = self._parse_size(req.size)

            # Check for cancellation after parsing
            if await self._check_cancellation(context):
                logger.info(f"Request cancelled during setup: {context.id()}")
                return

            # Generate images (may batch multiple requests at same step)
            images = await self._generate_images(
                prompt=req.prompt,
                num_images=req.n,
                width=width,
                height=height,
                num_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
            )

            # Check for cancellation after generation
            if await self._check_cancellation(context):
                logger.info(f"Request cancelled after generation: {context.id()}")
                return

            # Upload to S3 and get URLs
            image_data = []
            for img in images:
                if req.response_format == "url":
                    if self.s3_client is None or self.s3_bucket is None:
                        logger.warning("S3 client not implemented, using local URL")
                        url = f"https://localhost:8000/not-implemented/storage/of_files/{uuid.uuid4()}.png"
                    else:
                        url = await self._upload_to_s3(img, context.id())
                    image_data.append(ImageData(url=url))
                else:  # b64_json
                    b64 = self._encode_base64(img)
                    image_data.append(ImageData(b64_json=b64))

            response = ImagesResponse(created=int(time.time()), data=image_data)

            yield response.model_dump()

        except Exception as e:
            logger.error(f"Error in diffusion generation: {e}", exc_info=True)
            # Return error response
            error_response = {
                "created": int(time.time()),
                "data": [],
                "error": str(e),
            }
            yield error_response

    async def _generate_images(
        self,
        prompt: str,
        num_images: int,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> list[bytes]:
        """Generate images using SGLang DiffGenerator"""
        # DiffGenerator handles batching internally if multiple images
        # Run in thread pool to avoid blocking event loop
        images = await asyncio.to_thread(
            self.generator.generate,
            prompt=prompt,
            num_images=num_images,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Convert images to bytes (handle PIL Images, numpy arrays, or bytes)
        image_bytes_list = []
        for img in images:
            if isinstance(img, bytes):
                image_bytes_list.append(img)
            elif Image is not None and isinstance(img, Image.Image):
                # Convert PIL Image to bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes_list.append(buf.getvalue())
            else:
                # Try to convert numpy array or other formats
                try:
                    import numpy as np

                    if isinstance(img, np.ndarray):
                        # Convert numpy array to PIL Image then to bytes
                        if Image is None:
                            raise RuntimeError("PIL/Pillow required for numpy array conversion")
                        pil_img = Image.fromarray(img)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        image_bytes_list.append(buf.getvalue())
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")
                except ImportError:
                    raise RuntimeError(
                        "Cannot convert image format. Install Pillow: pip install Pillow"
                    )

        return image_bytes_list

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse '1024x1024' -> (1024, 1024)"""
        w, h = size_str.split("x")
        return int(w), int(h)

    async def _upload_to_s3(self, image_bytes: bytes, request_id: str) -> str:
        """Upload image to S3 and return public URL"""
        if self.s3_client is None or self.s3_bucket is None:
            raise RuntimeError("S3 client and bucket not configured")

        key = f"generations/{request_id}/{uuid.uuid4()}.png"
        await asyncio.to_thread(
            self.s3_client.put_object,
            Bucket=self.s3_bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
        )
        return f"https://{self.s3_bucket}.s3.amazonaws.com/{key}"

    def _encode_base64(self, image_bytes: bytes) -> str:
        """Encode image as base64 string"""
        return base64.b64encode(image_bytes).decode("utf-8")
