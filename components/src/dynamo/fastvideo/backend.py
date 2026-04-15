# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import yaml

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

DEFAULT_STAGE_LOGGING = "1"
DEFAULT_RMSNORM_FP4_PREQUANT = "0"
DEFAULT_RESPONSE_FORMAT = "url"


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
        os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", DEFAULT_STAGE_LOGGING)
        os.environ.setdefault(
            "FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT",
            DEFAULT_RMSNORM_FP4_PREQUANT,
        )

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

        pipeline_config = self._create_pipeline_config()
        generator_kwargs = self._build_generator_kwargs()

        return VideoGenerator.from_pretrained(
            self.config.model_path,
            num_gpus=self.config.num_gpus,
            pipeline_config=pipeline_config,
            **generator_kwargs,
        )

    def _read_extra_generator_args_file(self, path: str) -> dict[str, Any]:
        config_path = os.path.expanduser(os.path.expandvars(path))
        with open(config_path, encoding="utf-8") as file:
            text = file.read()

        suffix = os.path.splitext(config_path)[1].lower()
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Extra args file must contain a mapping, got {type(data).__name__}"
            )
        return data

    def _parse_override_generator_args_json(self, override: str) -> dict[str, Any]:
        data = json.loads(override)
        if not isinstance(data, dict):
            raise ValueError(
                "--override-generator-args-json must decode to a JSON object"
            )
        return data

    def _deep_update(self, target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            existing = target.get(key)
            if isinstance(value, dict) and isinstance(existing, dict):
                self._deep_update(existing, value)
            else:
                target[key] = value

    def _create_pipeline_config(self) -> Any:
        from fastvideo.configs.pipelines.base import PipelineConfig

        pipeline_config = PipelineConfig.from_pretrained(self.config.model_path)
        if not self.config.enable_fp4_quantization:
            return pipeline_config

        import torch

        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            logger.warning(
                "FP4 quantization is only supported on Blackwell GPUs "
                "(compute capability 10.0+). Detected %d.%d; continuing "
                "without FP4 quantization.",
                major,
                minor,
            )
        else:
            logger.info("Enabling FastVideo FP4 quantization")
            try:
                from fastvideo.layers.quantization.fp4_config import FP4Config
            except ImportError as exc:
                raise RuntimeError(
                    "FastVideo FP4 quantization requires "
                    "fastvideo.layers.quantization.fp4_config, but this "
                    "FastVideo build does not provide it. Re-run without "
                    "--fp4-quantization or install a build that includes "
                    "FP4 support."
                ) from exc
            pipeline_config.dit_config.quant_config = FP4Config()

        return pipeline_config

    def _build_generator_kwargs(self) -> dict[str, Any]:
        generator_kwargs: dict[str, Any] = {
            "enable_torch_compile": self.config.enable_torch_compile,
            "dit_cpu_offload": self.config.dit_cpu_offload,
            "dit_layerwise_offload": self.config.dit_layerwise_offload,
            "use_fsdp_inference": self.config.use_fsdp_inference,
            "vae_cpu_offload": self.config.vae_cpu_offload,
            "image_encoder_cpu_offload": self.config.image_encoder_cpu_offload,
            "text_encoder_cpu_offload": self.config.text_encoder_cpu_offload,
            "pin_cpu_memory": self.config.pin_cpu_memory,
            "disable_autocast": self.config.disable_autocast,
        }
        if self.config.enable_torch_compile:
            generator_kwargs["torch_compile_kwargs"] = {
                "backend": "inductor",
                "fullgraph": self.config.torch_compile_fullgraph,
                "mode": self.config.torch_compile_mode,
            }
        if self.config.extra_generator_args_file:
            self._deep_update(
                generator_kwargs,
                self._read_extra_generator_args_file(
                    self.config.extra_generator_args_file
                ),
            )
        if self.config.override_generator_args_json:
            self._deep_update(
                generator_kwargs,
                self._parse_override_generator_args_json(
                    self.config.override_generator_args_json
                ),
            )

        return generator_kwargs

    def _parse_size(self, size: Optional[str]) -> tuple[int, int]:
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

    def _resolve_response_format(self, response_format: Optional[str]) -> str:
        resolved = response_format or DEFAULT_RESPONSE_FORMAT
        if resolved not in {"b64_json", "url"}:
            raise ValueError("response_format must be one of: b64_json, url")
        return resolved

    def _validate_request_support(self, request: NvCreateVideoRequest) -> None:
        if request.input_reference is not None:
            raise ValueError(
                "FastVideo backend does not support input_reference "
                "(image-to-video) requests"
            )

    def _generate_mp4(
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
        negative_prompt: str | None,
    ) -> bytes:
        """Generate a video clip and return MP4 bytes."""
        if self.generator is None:
            raise RuntimeError("Generator is not initialized")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp4")
            kwargs: dict[str, Any] = {
                "save_video": True,
                "return_frames": False,
                "output_path": output_path,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            if seed is not None:
                kwargs["seed"] = seed
            if negative_prompt is not None:
                kwargs["negative_prompt"] = negative_prompt

            result = self.generator.generate_video(prompt=prompt, **kwargs)
            result_dict = result if isinstance(result, dict) else {}
            generation_time = _coerce_optional_float(result_dict.get("generation_time"))
            e2e_latency = _coerce_optional_float(result_dict.get("e2e_latency"))

            if generation_time is not None:
                logger.info("[%s] Generation time: %.2fs", request_id, generation_time)
            if e2e_latency is not None:
                logger.info("[%s] E2E latency: %.2fs", request_id, e2e_latency)

            with open(output_path, "rb") as file:
                return file.read()

    async def _make_video_data(
        self,
        response_format: str,
        request_id: str,
        video_bytes: bytes,
    ) -> VideoData:
        if response_format == "url":
            video_url = await upload_to_fs(
                self.media_output_fs,
                f"videos/{request_id}.mp4",
                video_bytes,
                self.media_output_http_url,
            )
            return VideoData(url=video_url)

        return VideoData(b64_json=base64.b64encode(video_bytes).decode("utf-8"))

    async def generate(
        self,
        request: dict[str, Any],
        context: Context,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate a video response from a shared Dynamo video request."""
        started_at = time.time()
        context_id = context.id()
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
                video_bytes = await asyncio.to_thread(
                    self._generate_mp4,
                    prompt=req.prompt,
                    request_id=request_id,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    negative_prompt=nvext.negative_prompt,
                )

            response = NvVideosResponse(
                id=request_id,
                model=req.model,
                created=int(time.time()),
                data=[
                    await self._make_video_data(
                        response_format=response_format,
                        request_id=request_token,
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
        self.generator = None


async def register_fastvideo_model(endpoint: Any, config: FastVideoConfig) -> None:
    await register_model(
        ModelInput.Text,
        ModelType.Videos,
        endpoint,
        config.model_path,
        config.served_model_name,
    )
