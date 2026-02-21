# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import logging
import random
import time
from typing import Any, AsyncGenerator, Optional

import torch

from dynamo._core import Context
from dynamo.common.storage import upload_to_fs
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    CreateVideoRequest,
    VideoData,
    VideoGenerationResponse,
    VideoNvExt,
)
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)


class VideoGenerationWorkerHandler(BaseGenerativeHandler):
    """Handler for video generation (T2V/I2V).

    Inherits from BaseGenerativeHandler for common infrastructure like
    tracing, metrics publishing, and cancellation support.
    """

    def __init__(
        self,
        generator: Any,  # DiffGenerator, not sgl.Engine
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        fs: Any = None,  # fsspec.AbstractFileSystem for primary storage
    ):
        """Initialize video generation worker handler.

        Args:
            generator: The SGLang DiffGenerator instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher (not used for video currently).
            fs: Optional fsspec filesystem for primary video storage.
        """
        # Call parent constructor for common setup
        super().__init__(config, publisher)

        # Video generation-specific initialization
        self.generator = generator  # DiffGenerator, not Engine
        self._generate_lock = asyncio.Lock()  # Serialize generator access
        self.fs = fs
        self.fs_url = config.dynamo_args.media_output_fs_url
        self.base_url = config.dynamo_args.media_output_http_url

        logger.info(
            f"Video generation worker handler initialized with fs_url={self.fs_url}"
        )

    def cleanup(self) -> None:
        """Cleanup generator resources"""
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        logger.info("Video generation generator cleanup complete")
        # Call parent cleanup for any base class cleanup
        super().cleanup()

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate video from text/image prompt.

        Unlike LLM streaming, video returns complete video at end.

        Args:
            request: Request dict with prompt and generation parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dict with generated video (OpenAI-compatible format).
        """
        start_time = time.time()

        # Get trace header for distributed tracing (for logging/observability)
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug(f"Video generation request with trace: {trace_header}")

        try:
            req = CreateVideoRequest(**request)
            nvext = req.nvext or VideoNvExt()

            logger.info(
                f"Video generation request: model={req.model}, "
                f"size={req.size}, steps={nvext.num_inference_steps}"
            )

            # Parse size
            width, height = self._parse_size(req.size)

            # Calculate num_frames if not explicitly provided
            num_frames = nvext.num_frames
            if num_frames is None:
                num_frames = nvext.fps * req.seconds

            # Generate video
            video_bytes = await self._generate_video(
                prompt=req.prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=nvext.fps,
                num_inference_steps=nvext.num_inference_steps,
                guidance_scale=nvext.guidance_scale,
                seed=nvext.seed,
                request_id=context.id(),
                negative_prompt=nvext.negative_prompt,
                input_reference=req.input_reference,
            )

            video_data = []
            if req.response_format == "url":
                url = await self._upload_to_fs(video_bytes, context.id())
                video_data.append(VideoData(url=url))
            else:  # b64_json
                b64 = self._encode_base64(video_bytes)
                video_data.append(VideoData(b64_json=b64))

            inference_time = time.time() - start_time

            response = VideoGenerationResponse(
                id=f"video-{context.id()}",
                model=req.model,
                created=int(time.time()),
                data=video_data,
                inference_time_s=inference_time,
            )

            yield response.model_dump()

        except Exception as e:
            logger.error(f"Error in video generation: {e}", exc_info=True)
            # Return error response
            error_response = VideoGenerationResponse(
                id=f"video-{context.id()}",
                model=request.get("model", "unknown"),
                created=int(time.time()),
                status="failed",
                progress=0,
                data=[],
                error=str(e),
            )
            yield error_response.model_dump()

    async def _generate_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        request_id: str,
        negative_prompt: Optional[str] = None,
        input_reference: Optional[str] = None,
    ) -> bytes:
        """Generate video using SGLang DiffGenerator.

        Args:
            prompt: Text prompt for video generation.
            width: Video width in pixels.
            height: Video height in pixels.
            num_frames: Number of frames to generate.
            fps: Frames per second for output video.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale for generation.
            seed: Random seed for reproducibility.
            request_id: Request ID for logging.
            negative_prompt: Optional negative prompt.
            input_reference: Optional image path for I2V.

        Returns:
            Video bytes (mp4 format).
        """
        # Build args for DiffGenerator
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "save_output": False,  # We handle saving ourselves
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else random.randint(0, 1000000),
        }

        # Add image_path for I2V if provided
        if input_reference:
            args["image_path"] = input_reference

        logger.info(
            f"Generating video with {num_frames} frames at {width}x{height}, "
            f"{num_inference_steps} steps, request_id={request_id}"
        )

        # Serialize access -- DiffGenerator has mutable state (CUDA graph
        # caches, shared config objects) and is not thread-safe.
        async with self._generate_lock:
            # Run in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(
                self.generator.generate,
                sampling_params_kwargs=args,
            )

        # Result contains 'frames' with list of frames
        frames = result.get("frames", [])
        if not frames:
            raise RuntimeError("DiffGenerator returned no frames")

        # Convert frames to video bytes
        video_bytes = await self._frames_to_video(frames, fps)
        return video_bytes

    async def _frames_to_video(
        self, frames: list, fps: int, codec: str = "libx264"
    ) -> bytes:
        """Convert list of frames to video bytes.

        Args:
            frames: List of frames (PIL Images or numpy arrays).
            fps: Frames per second.
            codec: Video codec to use.

        Returns:
            Video bytes in mp4 format.
        """
        try:
            import numpy as np
            from PIL import Image

            # Convert frames to numpy arrays if needed
            np_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    np_frames.append(np.array(frame))
                elif isinstance(frame, np.ndarray):
                    np_frames.append(frame)
                else:
                    raise ValueError(f"Unsupported frame type: {type(frame)}")

            # Use imageio to write video
            import imageio

            output_buffer = io.BytesIO()
            with imageio.get_writer(
                output_buffer,
                format="mp4",
                fps=fps,
                codec=codec,
                output_params=["-pix_fmt", "yuv420p"],
            ) as writer:
                for frame in np_frames:
                    writer.append_data(frame)

            output_buffer.seek(0)
            return output_buffer.read()

        except ImportError as e:
            raise RuntimeError(
                f"Missing dependency for video encoding: {e}. "
                "Install with: pip install imageio imageio-ffmpeg"
            )

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse 'WxH' -> (width, height)"""
        try:
            w, h = size_str.split("x")
            return int(w), int(h)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid size format '{size_str}', expected 'WxH' (e.g. '832x480')"
            ) from e

    async def _upload_to_fs(self, video_bytes: bytes, request_id: str) -> str:
        """Upload video to filesystem and return URL.

        Args:
            video_bytes: Video data as bytes.
            request_id: Request context ID.

        Returns:
            URL for the uploaded video.
        """
        storage_path = f"{request_id}.mp4"
        return await upload_to_fs(self.fs, storage_path, video_bytes, self.base_url)

    def _encode_base64(self, video_bytes: bytes) -> str:
        """Encode video as base64 string"""
        return base64.b64encode(video_bytes).decode("utf-8")
