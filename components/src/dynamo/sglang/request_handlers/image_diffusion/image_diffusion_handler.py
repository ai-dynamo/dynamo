# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import logging
import random
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import torch
from PIL import Image

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import CreateImageRequest, ImageData, ImagesResponse, NvExt
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)

MAX_NUM_INFERENCE_STEPS = 50


class ImageDiffusionWorkerHandler(BaseGenerativeHandler):
    """Handler for diffusion image generation.

    Inherits from BaseGenerativeHandler for common infrastructure like
    tracing, metrics publishing
    """

    def __init__(
        self,
        component: Component,
        generator: Any,  # DiffGenerator, not sgl.Engine
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        fs: Any = None,  # fsspec.AbstractFileSystem for primary storage
    ):
        """Initialize diffusion worker handler.

        Args:
            component: The Dynamo runtime component.
            generator: The SGLang DiffGenerator instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher (not used for diffusion currently).
            fs: Optional fsspec filesystem for primary image storage.
        """
        super().__init__(component, config, publisher)

        self.generator = generator  # DiffGenerator, not Engine
        self.fs = fs
        self.fs_url = config.dynamo_args.image_diffusion_fs_url
        self.base_url = config.dynamo_args.image_diffusion_base_url

        logger.info(
            f"Image diffusion worker handler initialized with fs_url={self.fs_url}, url_base={self.base_url}"
        )

    def cleanup(self) -> None:
        """Cleanup generator resources"""
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        logger.info("Image diffusion generator cleanup complete")
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
        logger.info(f"Image diffusion request: {request}")

        # Get trace header for distributed tracing (for logging/observability)
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug(f"Image diffusion request with trace: {trace_header}")

        try:
            req = CreateImageRequest(**request)

            # get extra parameters
            nvext = req.nvext or NvExt()
            nvext.num_inference_steps = min(
                nvext.num_inference_steps or 50, MAX_NUM_INFERENCE_STEPS
            )

            width, height = self._parse_size(req.size)

            images = await self._generate_images(
                prompt=req.prompt,
                negative_prompt=nvext.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=nvext.num_inference_steps,
                guidance_scale=nvext.guidance_scale,
                seed=nvext.seed,
            )

            user_id = req.user if req.user else context.id()

            image_data = []
            for img in images:
                # uploading or encoding the image
                if req.response_format == "url":
                    url = await self._upload_to_fs(img, user_id, context.id())
                    image_data.append(ImageData(url=url))
                else:
                    b64 = self._encode_base64(img)
                    image_data.append(ImageData(b64_json=b64))

            response = ImagesResponse(created=int(time.time()), data=image_data)

            yield response.model_dump()

        except Exception as e:
            logger.error(f"Error in diffusion generation: {e}", exc_info=True)
            error_response = {
                "created": int(time.time()),
                "data": [],
                "error": str(e),
            }
            yield error_response

    async def _generate_images(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        negative_prompt: Optional[str] = None,
    ) -> list[bytes]:
        """Generate images using SGLang DiffGenerator"""
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "save_output": False,  # We handle saving ourselves
            "guidance_scale": guidance_scale,
            "seed": seed if seed else random.randint(0, 1000000),
        }
        result = await asyncio.to_thread(
            self.generator.generate,
            sampling_params_kwargs=args,
        )

        if result is None:
            raise RuntimeError("No result from generator")

        images = result["frames"] if "frames" in result else []

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
                try:
                    import numpy as np

                    if isinstance(img, np.ndarray):
                        # Convert numpy array to PIL Image then to bytes
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

    def _parse_size(self, size_str: Optional[str]) -> tuple[int, int]:
        """Parse '1024x1024' -> (1024, 1024)"""
        if size_str is None:
            return 1024, 1024

        w, h = size_str.split("x")
        return int(w), int(h)

    async def _upload_to_fs(
        self, image_bytes: bytes, user_id: str, request_id: str
    ) -> str:
        """Upload image to filesystem and return URL.

        Uses per-user storage path:
            users/{user_id}/generations/{request_id}/{image_uuid}.png

        Args:
            image_bytes: Image data as bytes.
            user_id: User identifier from request or context.
            request_id: Request context ID.

        Returns:
            Public URL for the uploaded image.
        """
        image_uuid = str(uuid.uuid4())
        image_filename = f"{image_uuid}.png"

        # Per-user storage path
        storage_path = f"users/{user_id}/generations/{request_id}/{image_filename}"

        # send image to filesystem
        await asyncio.to_thread(self.fs.pipe, storage_path, image_bytes)

        return f"{self.base_url}/{storage_path}"

    def _encode_base64(self, image_bytes: bytes) -> str:
        """Encode image as base64 string"""
        return base64.b64encode(image_bytes).decode("utf-8")
