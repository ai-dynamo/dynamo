# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo request handler for the native FastVideo backend."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)
from dynamo.common.storage import get_fs, upload_to_fs
from dynamo.llm import ModelInput, ModelType, register_model

if TYPE_CHECKING:
    from dynamo._core import Context

    from .args import FastVideoConfig

logger = logging.getLogger(__name__)

DEFAULT_RESPONSE_FORMAT = "url"
DEFAULT_OUTPUT_FORMAT = "mp4"


def _coerce_optional_float(value: object) -> float | None:
    """Best-effort conversion for optional numeric metrics from FastVideo."""
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FastVideoHandler:
    """Serve FastVideo generation requests through Dynamo."""

    def __init__(
        self,
        config: FastVideoConfig,
        generator: Any | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self._generate_lock = asyncio.Lock()
        self.media_output_fs = get_fs(config.media_output_fs_url)
        self.media_output_http_url = config.media_output_http_url
        self._apply_environment()

    def _apply_environment(self) -> None:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = self.config.attention_backend

    async def initialize(self) -> None:
        """Load the FastVideo model if one was not injected."""
        if self.generator is not None:
            return

        logger.info("Loading FastVideo model from %s", self.config.model_path)
        self.generator = await asyncio.to_thread(self._load_generator)
        logger.info("FastVideo model ready")

    def _load_generator(self) -> Any:
        try:
            from fastvideo import VideoGenerator
        except ImportError as exc:
            raise ImportError(
                "FastVideo backend dependencies are not installed. "
                "Install FastVideo and its runtime requirements before running "
                "python -m dynamo.fastvideo."
            ) from exc

        return VideoGenerator.from_config(self.config.to_generator_config())

    def _parse_size(self, size: str | None) -> tuple[int, int]:
        size_value = size or self.config.default_size
        try:
            width_str, height_str = size_value.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid size format '{size_value}', expected 'WxH'"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid size '{size_value}', width and height must be positive"
            )

        return width, height

    def _compute_num_frames(
        self,
        request: NvCreateVideoRequest,
        nvext: VideoNvExt,
    ) -> int:
        if nvext.num_frames is not None:
            num_frames = nvext.num_frames
        elif request.seconds is None and nvext.fps is None:
            num_frames = self.config.default_num_frames
        else:
            seconds = (
                request.seconds
                if request.seconds is not None
                else self.config.default_seconds
            )
            fps = nvext.fps if nvext.fps is not None else self.config.default_fps
            num_frames = seconds * fps

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        return num_frames

    def _resolve_response_format(self, response_format: str | None) -> str:
        resolved = response_format or DEFAULT_RESPONSE_FORMAT
        if resolved not in {"b64_json", "url"}:
            raise ValueError("response_format must be one of: b64_json, url")
        return resolved

    def _resolve_output_format(self, output_format: str | None) -> str:
        resolved = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
        if resolved != DEFAULT_OUTPUT_FORMAT:
            raise ValueError("FastVideo backend only supports mp4 output_format")
        return resolved

    def _validate_request_support(self, request: NvCreateVideoRequest) -> None:
        if request.input_reference is not None:
            raise ValueError(
                "FastVideo backend does not support input_reference "
                "(image-to-video) requests yet"
            )

    def _build_generation_request(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        nvext: VideoNvExt,
    ) -> Any:
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        sampling_kwargs: dict[str, Any] = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        if nvext.boundary_ratio is not None:
            sampling_kwargs["boundary_ratio"] = nvext.boundary_ratio
        if nvext.guidance_scale_2 is not None:
            sampling_kwargs["guidance_scale_2"] = nvext.guidance_scale_2

        return GenerationRequest(
            prompt=prompt,
            negative_prompt=nvext.negative_prompt,
            sampling=SamplingConfig(**sampling_kwargs),
            output=OutputConfig(
                output_path=output_path,
                save_video=True,
                return_frames=False,
            ),
        )

    def _generate_video_bytes(
        self,
        *,
        prompt: str,
        request_id: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        nvext: VideoNvExt,
    ) -> tuple[bytes, float | None]:
        if self.generator is None:
            raise RuntimeError("FastVideo generator is not initialized")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"{request_id}.mp4")
            generation_request = self._build_generation_request(
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                nvext=nvext,
            )
            result = self.generator.generate(generation_request)
            result = self._coerce_single_result(result)

            video_path_value = getattr(result, "video_path", None)
            generation_time = _coerce_optional_float(
                getattr(result, "generation_time", None)
            )
            if not video_path_value:
                raise RuntimeError("FastVideo generation did not return a video_path")

            video_path = Path(video_path_value)
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"FastVideo output video was not found at {video_path}"
                )

            return video_path.read_bytes(), generation_time

    def _coerce_single_result(self, result: Any) -> Any:
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(
                    f"Expected one FastVideo result, received {len(result)}"
                )
            return result[0]
        return result

    async def _make_video_data(
        self,
        *,
        output_format: str,
        response_format: str,
        request_id: str,
        video_bytes: bytes,
    ) -> VideoData:
        if response_format == "url":
            video_url = await upload_to_fs(
                self.media_output_fs,
                f"videos/{request_id}.{output_format}",
                video_bytes,
                self.media_output_http_url,
            )
            return VideoData(output_format=output_format, url=video_url)

        return VideoData(
            output_format=output_format,
            b64_json=base64.b64encode(video_bytes).decode("utf-8"),
        )

    async def generate(
        self,
        request: dict[str, Any],
        context: Context,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate a video response from a shared Dynamo video request."""
        started_at = time.time()
        context_id = self._context_id(context)
        request_token = context_id or uuid.uuid4().hex
        request_id = f"video_{request_token}"

        try:
            if self.generator is None:
                raise RuntimeError("FastVideo generator is not initialized")

            req = NvCreateVideoRequest(**request)
            self._validate_request_support(req)
            nvext = req.nvext or VideoNvExt()

            width, height = self._parse_size(req.size)
            fps = nvext.fps if nvext.fps is not None else self.config.default_fps
            if fps <= 0:
                raise ValueError("fps must be positive")

            num_frames = self._compute_num_frames(req, nvext)
            num_inference_steps = (
                nvext.num_inference_steps
                if nvext.num_inference_steps is not None
                else self.config.default_num_inference_steps
            )
            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be positive")

            guidance_scale = (
                nvext.guidance_scale
                if nvext.guidance_scale is not None
                else self.config.default_guidance_scale
            )
            seed = nvext.seed if nvext.seed is not None else self.config.default_seed
            response_format = self._resolve_response_format(req.response_format)
            output_format = self._resolve_output_format(req.output_format)

            logger.info(
                "[%s] prompt_len=%d size=%dx%d frames=%d fps=%d steps=%d",
                request_id,
                len(req.prompt),
                width,
                height,
                num_frames,
                fps,
                num_inference_steps,
            )

            async with self._generate_lock:
                video_bytes, generation_time = await asyncio.to_thread(
                    self._generate_video_bytes,
                    prompt=req.prompt,
                    request_id=request_id,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    nvext=nvext,
                )

            if generation_time is not None:
                logger.info(
                    "[%s] FastVideo generation time: %.2fs",
                    request_id,
                    generation_time,
                )

            response = NvVideosResponse(
                id=request_id,
                model=req.model,
                created=int(time.time()),
                data=[
                    await self._make_video_data(
                        output_format=output_format,
                        response_format=response_format,
                        request_id=request_id,
                        video_bytes=video_bytes,
                    )
                ],
                inference_time_s=time.time() - started_at,
            )
            yield response.model_dump()
        except Exception as exc:
            logger.exception("[%s] FastVideo request failed", request_id)
            yield NvVideosResponse(
                id=request_id,
                model=request.get(
                    "model",
                    self.config.served_model_name or self.config.model_path,
                ),
                created=int(time.time()),
                status="failed",
                progress=0,
                data=[],
                error=str(exc),
                inference_time_s=time.time() - started_at,
            ).model_dump()

    def cleanup(self) -> None:
        """Release generator resources."""
        generator = self.generator
        self.generator = None
        if generator is None:
            return
        shutdown = getattr(generator, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.exception("Error during FastVideo generator shutdown")

    @staticmethod
    def _context_id(context: Context) -> str | None:
        try:
            value = context.id()
        except Exception:
            return None
        if value is None:
            return None
        return str(value)


async def register_fastvideo_model(endpoint: Any, config: FastVideoConfig) -> None:
    await register_model(
        ModelInput.Text,
        ModelType.Videos,
        endpoint,
        config.model_path,
        config.served_model_name,
    )
