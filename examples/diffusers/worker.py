#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Worker for Dynamo (non-streaming)

Registers a VideoGenerator as a Dynamo backend endpoint compatible with the
/v1/videos frontend endpoint.  The endpoint generates a full video
clip from the request parameters and returns it as a single response containing
the complete MP4 file base64-encoded in data[0].b64_json.

Generation parameters (size, fps, num_frames, etc.) are taken from the
request body's nvext field, so the same worker instance can serve requests
with different resolutions and quality settings without restarting.

One request at a time (asyncio.Lock — VideoGenerator is not re-entrant).

Usage:
  python worker.py [--model MODEL] [--num-gpus N] [--disable-optimizations]

Options:
  --model          HuggingFace model path
                   (default: FastVideo/LTX2-Distilled-Diffusers)
  --num-gpus       Number of GPUs (default: 1)

Request format (sent to /v1/videos):
  prompt:   text description of the desired video
  model:    HuggingFace model path (must match what the worker registered)
  size:     "WxH" string, e.g. "1920x1088" (default: "1920x1088")
  seconds:  clip duration when nvext.num_frames is not set (default: 5)
  nvext:
    fps:                frames per second (default: 24)
    num_frames:         total frames; overrides fps * seconds when set (default: 121)
    num_inference_steps diffusion steps (default: 5)
    guidance_scale:     CFG scale (default: 1.0)
    seed:               RNG seed (default: 10)
    negative_prompt:    text to avoid (optional)
"""

import argparse
import asyncio
import base64
import logging
import os
import tempfile
import time
import uuid

import uvloop
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.fp4_config import FP4Config
from pydantic import BaseModel, Field

from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "FastVideo/LTX2-Distilled-Diffusers"

# ── Request / Response models ─────────────────────────────────────────────────


def _get_worker_namespace() -> str:
    """
    Resolve Dynamo namespace for endpoint registration.

    Kubernetes operator injects DYN_NAMESPACE (and optionally a rollout suffix).
    Compose/local runs keep using the historical "dynamo" default.
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


class NvExtVideoCreateRequest(BaseModel):
    fps: int = Field(default=24, description="Frames per second")
    num_frames: int | None = Field(
        default=121, description="Total frames; overrides fps * seconds"
    )
    num_inference_steps: int = Field(default=5, description="Diffusion inference steps")
    guidance_scale: float = Field(
        default=1.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(default=10, description="RNG seed for reproducibility")
    negative_prompt: str | None = Field(
        default=None, description="Text to avoid in generation"
    )


class VideoCreateRequest(BaseModel):
    prompt: str = Field(description="Text description of the desired video")
    model: str = Field(description="HuggingFace model path")
    size: str = Field(default="1920x1088", description="Frame dimensions as 'WxH'")
    seconds: int = Field(
        default=5, description="Clip duration; used when nvext.num_frames is unset"
    )
    user: str | None = Field(default=None)
    nvext: NvExtVideoCreateRequest = Field(default_factory=NvExtVideoCreateRequest)


class VideoData(BaseModel):
    b64_json: str | None = Field(default=None, description="Base64-encoded MP4 video")
    mime_type: str = Field(default="video/mp4")


class VideoCreateResponse(BaseModel):
    id: str
    object: str = "video"
    created: int
    model: str
    status: str = "complete"
    data: list[VideoData]


# ── Backend ───────────────────────────────────────────────────────────────────


def _coerce_optional_float(value: object) -> float | None:
    """Best-effort conversion for optional numeric metrics from backend results."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FastVideoBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.num_gpus: int = args.num_gpus
        self.disable_optimizations: bool = args.disable_optimizations

        # One request at a time — VideoGenerator is not re-entrant
        self._generate_lock = asyncio.Lock()
        self.generator: VideoGenerator | None = None

        attn_backend = "TORCH_SDPA" if self.disable_optimizations else "FLASH_ATTN"
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = attn_backend
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
        os.environ["FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT"] = "0"

    async def initialize_model(self) -> None:
        logger.info("Loading VideoGenerator model=%s", self.model_name)
        loop = asyncio.get_running_loop()

        def _load():
            pipeline_config = PipelineConfig.from_pretrained(self.model_name)
            if not self.disable_optimizations:
                logger.info(
                    "Using FP4 quantization for VideoGenerator model=%s",
                    self.model_name,
                )
                pipeline_config.dit_config.quant_config = FP4Config()

            return VideoGenerator.from_pretrained(
                self.model_name,
                num_gpus=self.num_gpus,
                ltx2_refine_enabled=True,
                ltx2_refine_lora_path="",  # disable refine lora for distilled model
                ltx2_refine_num_inference_steps=2,
                ltx2_refine_guidance_scale=1.0,
                ltx2_refine_add_noise=True,
                pipeline_config=pipeline_config,
                enable_torch_compile=not self.disable_optimizations,
                enable_torch_compile_text_encoder=not self.disable_optimizations,
                torch_compile_kwargs={
                    "backend": "inductor",
                    "fullgraph": True,
                    "mode": "max-autotune-no-cudagraphs",
                },
                dit_cpu_offload=False,
                vae_cpu_offload=False,
                text_encoder_cpu_offload=False,
                ltx2_vae_tiling=False,
            )

        self.generator = await loop.run_in_executor(None, _load)
        logger.info("VideoGenerator ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_mp4(
        self,
        prompt: str,
        video_id: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        negative_prompt: str | None,
    ) -> bytes:
        """Generate a video clip and return it as MP4 bytes."""
        assert self.generator is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp4")
            kwargs: dict = dict(
                save_video=True,
                return_frames=False,
                output_path=output_path,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            if seed is not None:
                kwargs["seed"] = seed
            if negative_prompt is not None:
                kwargs["negative_prompt"] = negative_prompt

            result = self.generator.generate_video(prompt=prompt, **kwargs)
            result_dict = result if isinstance(result, dict) else {}
            generation_time = _coerce_optional_float(result_dict.get("generation_time"))
            e2e_latency = _coerce_optional_float(result_dict.get("e2e_latency"))
            logger.info("[%s] MP4 written to %s", video_id, output_path)
            if generation_time is not None:
                logger.info(
                    "[%s] Generation time: %.2f seconds", video_id, generation_time
                )
            else:
                logger.info("[%s] Generation time: unavailable", video_id)

            if e2e_latency is not None:
                logger.info("[%s] E2E latency: %.2f seconds", video_id, e2e_latency)
            else:
                logger.info("[%s] E2E latency: unavailable", video_id)

            time_start = time.perf_counter()
            with open(output_path, "rb") as f:
                data = f.read()
            time_end = time.perf_counter()
            logger.info(
                "[%s] File read time: %.2f seconds", video_id, time_end - time_start
            )

            return data

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(VideoCreateRequest, VideoCreateResponse)
    async def create_video(self, request: VideoCreateRequest):
        """
        Non-streaming endpoint.

        Generates one video clip using the parameters from the request's nvext
        field, then yields a single VideoCreateResponse with data[0].b64_json
        containing the complete MP4 file encoded in base64.
        """
        if self.generator is None:
            raise RuntimeError("Generator is not initialized")

        nvext = request.nvext
        try:
            width_str, height_str = request.size.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid size format '{request.size}', expected 'WxH'"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid size '{request.size}', width and height must be positive"
            )

        num_frames = (
            nvext.num_frames
            if nvext.num_frames is not None
            else nvext.fps * request.seconds
        )
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        fps = nvext.fps
        if fps <= 0:
            raise ValueError("fps must be positive")

        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(time.time())

        logger.info(
            "[%s] create_video: prompt='%s...' size=%s frames=%d steps=%d",
            video_id,
            request.prompt[:60],
            request.size,
            num_frames,
            nvext.num_inference_steps,
        )
        logger.info(
            "[%s] Waiting for generate lock (locked=%s)",
            video_id,
            self._generate_lock.locked(),
        )
        async with self._generate_lock:
            t = time.perf_counter()
            logger.info(
                "[%s] Generating video (%dx%d, %d frames, %d steps) ...",
                video_id,
                width,
                height,
                num_frames,
                nvext.num_inference_steps,
            )
            try:
                mp4_bytes = await asyncio.to_thread(
                    self._generate_mp4,
                    prompt=request.prompt,
                    video_id=video_id,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    num_inference_steps=nvext.num_inference_steps,
                    guidance_scale=nvext.guidance_scale,
                    seed=nvext.seed,
                    negative_prompt=nvext.negative_prompt,
                )
            except Exception as exc:
                logger.exception("[%s] Generation failed", video_id)
                raise RuntimeError(
                    f"Video generation failed for request {video_id}"
                ) from exc

            elapsed = time.perf_counter() - t
            logger.info(
                "[%s] Generation done in %.1fs — encoding %.2f MB MP4",
                video_id,
                elapsed,
                len(mp4_bytes) / 1_048_576,
            )

            yield VideoCreateResponse(
                id=video_id,
                created=created_ts,
                model=request.model,
                data=[VideoData(b64_json=base64.b64encode(mp4_bytes).decode())],
            ).model_dump()
        logger.info("[%s] Generation request finished", video_id)


# ── Dynamo wiring ─────────────────────────────────────────────────────────────


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Videos,
            endpoint,
            model_name,
            model_name,
        )
        logger.info("Successfully registered model: %s", model_name)
    except Exception as e:
        logger.error("Failed to register model: %s", e, exc_info=True)
        raise RuntimeError("Model registration failed") from e


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    component_name = "backend"
    endpoint_name = "generate"

    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")
    logger.info(
        "Serving endpoint %s/%s/%s", namespace_name, component_name, endpoint_name
    )

    backend = FastVideoBackend(args)
    await backend.initialize_model()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.create_video),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastVideo Worker for Dynamo (non-streaming)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        dest="num_gpus",
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--disable-optimizations",
        action="store_true",
        dest="disable_optimizations",
        help="Disable FP4 quantization, torch.compile, and use TORCH_SDPA attention",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    # Use Kubernetes discovery in-cluster and file discovery for local compose by default.
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, "tcp", False)
    await backend_worker(runtime, args)


if __name__ == "__main__":
    _args = _parse_args()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("FASTVIDEO_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(_args))
