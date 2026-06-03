#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastVideo example worker for Dynamo's /v1/videos endpoint.

This custom worker intentionally stays under examples/diffusers rather than
becoming a native Dynamo backend. It uses FastVideo's typed API only:

* VideoGenerator.from_config(GeneratorConfig(...))
* generator.generate(GenerationRequest(...))

One request is processed at a time because FastVideo generators are not
re-entrant.
"""

import argparse
import asyncio
import base64
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import uvloop
from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    PipelineSelection,
    QuantizationConfig,
    SamplingConfig,
)
from pydantic import BaseModel, Field

from dynamo.common.configuration import add_negatable_bool_argument
from dynamo.llm import (  # type: ignore[attr-defined]
    ModelInput,
    ModelType,
    WorkerType,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)


def _default_output_dir() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return str(Path(runtime_dir) / "dynamo-fastvideo" / "outputs")

    home_dir = Path.home()
    if str(home_dir) not in {"", "/"}:
        return str(home_dir / ".cache" / "dynamo" / "fastvideo" / "outputs")

    uid = os.getuid() if hasattr(os, "getuid") else "unknown"
    return str(Path("/tmp") / f"dynamo-fastvideo-{uid}" / "outputs")


DEFAULT_MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
DEFAULT_ATTENTION_BACKEND = "VIDEO_SPARSE_ATTN"
DEFAULT_TORCH_COMPILE_BACKEND = "inductor"
DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
DEFAULT_SIZE = "1280x720"
DEFAULT_SECONDS = 5
DEFAULT_FPS = 24
DEFAULT_NUM_FRAMES = 125
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = None
DEFAULT_MAX_VIDEO_WIDTH = 4096
DEFAULT_MAX_VIDEO_HEIGHT = 4096
DEFAULT_RESPONSE_FORMAT = "b64_json"
DEFAULT_OUTPUT_FORMAT = "mp4"
DEFAULT_OUTPUT_DIR = _default_output_dir()
DEFAULT_VSA_SPARSITY = 0.8
ATTENTION_BACKEND_CHOICES = (
    "FLASH_ATTN",
    "TORCH_SDPA",
    "SAGE_ATTN",
    "SAGE_ATTN_THREE",
    "VIDEO_SPARSE_ATTN",
    "VMOBA_ATTN",
    "SLA_ATTN",
    "SAGE_SLA_ATTN",
)
TORCH_COMPILE_MODE_CHOICES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

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
    fps: int | None = Field(default=None, description="Frames per second")
    num_frames: int | None = Field(
        default=None, description="Total frames; overrides fps * seconds"
    )
    num_inference_steps: int | None = Field(
        default=None, description="Diffusion inference steps"
    )
    guidance_scale: float | None = Field(
        default=None, description="Classifier-free guidance scale"
    )
    guidance_scale_2: float | None = Field(
        default=None, description="Secondary classifier-free guidance scale"
    )
    boundary_ratio: float | None = Field(
        default=None, description="Expert switching boundary ratio"
    )
    seed: int | None = Field(default=None, description="RNG seed")
    negative_prompt: str | None = Field(
        default=None, description="Text to avoid in generation"
    )


class VideoCreateRequest(BaseModel):
    prompt: str = Field(description="Text description of the desired video")
    model: str = Field(description="HuggingFace model path")
    input_reference: str | None = Field(default=None)
    size: str | None = Field(default=None, description="Frame dimensions as 'WxH'")
    seconds: int | None = Field(default=None)
    user: str | None = Field(default=None)
    response_format: Literal["url", "b64_json"] | None = Field(default=None)
    output_format: str | None = Field(default=None)
    stream: bool | None = Field(default=None)
    nvext: NvExtVideoCreateRequest | None = Field(default=None)


class VideoData(BaseModel):
    output_format: str
    url: str | None = None
    b64_json: str | None = None


class VideoCreateResponse(BaseModel):
    id: str
    object: str = "video"
    model: str
    status: str = "completed"
    progress: int = 100
    created: int
    data: list[VideoData] = Field(default_factory=list)
    error: str | None = None
    inference_time_s: float | None = None


# ── Backend ───────────────────────────────────────────────────────────────────


def _coerce_optional_float(value: object) -> float | None:
    """Best-effort conversion for optional numeric metrics from backend results."""
    if value is None or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _supports_fp4_quantization() -> bool:
    try:
        import torch
    except ImportError as exc:
        logger.warning(
            "NVFP4 quantization requested but torch is unavailable. "
            "Continuing without NVFP4 quantization: %s",
            exc,
        )
        return False

    try:
        major, minor = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError) as exc:
        logger.warning(
            "NVFP4 quantization requested but CUDA device capability could not be "
            "detected. Continuing without NVFP4 quantization: %s",
            exc,
        )
        return False

    if major < 10:
        logger.warning(
            "NVFP4 quantization is only supported on NVIDIA Blackwell GPUs "
            "(compute capability 10.0+). Detected compute capability: %d.%d. "
            "Continuing without NVFP4 quantization.",
            major,
            minor,
        )
        return False

    try:
        from fastvideo.layers.quantization import get_quantization_config

        get_quantization_config("NVFP4")
    except (ImportError, ValueError) as exc:
        logger.warning(
            "NVFP4 quantization requested but this FastVideo build does not provide "
            "NVFP4 transformer quantization. Continuing without NVFP4 "
            "quantization: %s",
            exc,
        )
        return False

    return True


class FastVideoBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name: str = args.model
        self.output_dir = Path(args.output_dir)

        # One request at a time — VideoGenerator is not re-entrant
        self._generate_lock = asyncio.Lock()
        self.generator: VideoGenerator | None = None

        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
        os.environ["FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT"] = "0"

    async def initialize_model(self) -> None:
        logger.info("Loading VideoGenerator model=%s", self.model_name)
        self.generator = await asyncio.to_thread(
            VideoGenerator.from_config,
            self._build_generator_config(),
        )
        logger.info("VideoGenerator ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_generator_config(self) -> GeneratorConfig:
        enable_torch_compile = self.args.torch_compile or self.args.enable_optimizations
        enable_fp4_quantization = (
            self.args.fp4_quantization or self.args.enable_optimizations
        )

        compile_config = CompileConfig(enabled=enable_torch_compile)
        if enable_torch_compile:
            compile_config.backend = self.args.torch_compile_backend
            compile_config.fullgraph = self.args.torch_compile_fullgraph
            compile_config.mode = self.args.torch_compile_mode
            compile_config.dynamic = self.args.torch_compile_dynamic

        quantization = None
        experimental: dict[str, object] = {}
        if self.args.vsa_sparsity is not None:
            experimental["VSA_sparsity"] = self.args.vsa_sparsity
        if enable_fp4_quantization and _supports_fp4_quantization():
            quantization = QuantizationConfig(transformer_quant="NVFP4")

        return GeneratorConfig(
            model_path=self.model_name,
            engine=EngineConfig(
                num_gpus=self.args.num_gpus,
                use_fsdp_inference=self.args.use_fsdp_inference,
                disable_autocast=self.args.disable_autocast,
                offload=OffloadConfig(
                    dit=self.args.dit_cpu_offload,
                    dit_layerwise=self.args.dit_layerwise_offload,
                    text_encoder=self.args.text_encoder_cpu_offload,
                    image_encoder=self.args.image_encoder_cpu_offload,
                    vae=self.args.vae_cpu_offload,
                    pin_cpu_memory=self.args.pin_cpu_memory,
                ),
                compile=compile_config,
                quantization=quantization,
            ),
            pipeline=PipelineSelection(experimental=experimental),
        )

    def _parse_size(self, size: str | None) -> tuple[int, int]:
        size_value = size or self.args.default_size
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
        if width > self.args.max_video_width or height > self.args.max_video_height:
            raise ValueError(
                f"Invalid size '{size_value}', exceeds maximum "
                f"{self.args.max_video_width}x{self.args.max_video_height}"
            )
        return width, height

    def _compute_num_frames(
        self,
        request: VideoCreateRequest,
        nvext: NvExtVideoCreateRequest,
    ) -> int:
        if nvext.num_frames is not None:
            num_frames = nvext.num_frames
        elif request.seconds is None and nvext.fps is None:
            num_frames = self.args.default_num_frames
        else:
            seconds = (
                request.seconds
                if request.seconds is not None
                else self.args.default_seconds
            )
            fps = nvext.fps if nvext.fps is not None else self.args.default_fps
            num_frames = seconds * fps

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        return num_frames

    def _resolve_response_format(self, response_format: str | None) -> str:
        resolved = response_format or DEFAULT_RESPONSE_FORMAT
        if resolved != "b64_json":
            raise ValueError(
                "FastVideo example worker only supports response_format='b64_json'; "
                "url responses require configured media storage"
            )
        return resolved

    def _resolve_output_format(self, output_format: str | None) -> str:
        resolved = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
        if resolved != DEFAULT_OUTPUT_FORMAT:
            raise ValueError("FastVideo worker only supports mp4 output_format")
        return resolved

    def _build_generation_request(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        nvext: NvExtVideoCreateRequest,
    ) -> GenerationRequest:
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

        generation_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "sampling": SamplingConfig(**sampling_kwargs),
            "output": OutputConfig(
                output_path=str(output_path),
                save_video=True,
                return_frames=False,
            ),
        }
        if nvext.negative_prompt is not None:
            generation_kwargs["negative_prompt"] = nvext.negative_prompt

        return GenerationRequest(**generation_kwargs)

    def _coerce_single_result(self, result: Any) -> Any:
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(f"Expected one FastVideo result, got {len(result)}")
            return result[0]
        return result

    def _generate_video(
        self,
        *,
        prompt: str,
        video_id: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        nvext: NvExtVideoCreateRequest,
    ) -> tuple[Path, float | None]:
        if self.generator is None:
            raise RuntimeError("Generator is not initialized")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{video_id}.mp4"
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
        try:
            result = self.generator.generate(generation_request)
        except Exception:
            self._cleanup_staging_file(output_path, video_id)
            raise
        result = self._coerce_single_result(result)
        video_path = Path(getattr(result, "video_path", None) or output_path)
        generation_time = _coerce_optional_float(
            getattr(result, "generation_time", None)
        )

        if not video_path.is_file():
            raise FileNotFoundError(f"FastVideo output video not found: {video_path}")
        return video_path, generation_time

    def _cleanup_staging_file(self, video_path: Path | None, video_id: str) -> None:
        if video_path is None:
            return
        try:
            video_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "[%s] Failed to delete staging video %s",
                video_id,
                video_path,
                exc_info=True,
            )

    async def _make_video_data(
        self,
        *,
        video_id: str,
        output_format: str,
        response_format: str,
        video_path: Path,
    ) -> VideoData:
        if response_format != "b64_json":
            raise ValueError(
                "FastVideo example worker only supports response_format='b64_json'"
            )

        try:
            video_bytes = await asyncio.to_thread(video_path.read_bytes)
            return VideoData(
                output_format=output_format,
                b64_json=base64.b64encode(video_bytes).decode("utf-8"),
            )
        finally:
            self._cleanup_staging_file(video_path, video_id)

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(VideoCreateRequest, VideoCreateResponse)
    async def create_video(self, request: VideoCreateRequest):
        """Generate one video clip and yield a single /v1/videos response."""
        started_at = time.time()
        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(started_at)
        video_path: Path | None = None

        try:
            if self.generator is None:
                raise RuntimeError("Generator is not initialized")
            if request.input_reference is not None:
                raise ValueError(
                    "FastVideo example worker does not support input_reference "
                    "(image-to-video) requests"
                )

            nvext = request.nvext or NvExtVideoCreateRequest()
            width, height = self._parse_size(request.size)
            fps = nvext.fps if nvext.fps is not None else self.args.default_fps
            if fps <= 0:
                raise ValueError("fps must be positive")

            num_frames = self._compute_num_frames(request, nvext)
            num_inference_steps = (
                nvext.num_inference_steps
                if nvext.num_inference_steps is not None
                else self.args.default_num_inference_steps
            )
            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be positive")

            guidance_scale = (
                nvext.guidance_scale
                if nvext.guidance_scale is not None
                else self.args.default_guidance_scale
            )
            seed = nvext.seed if nvext.seed is not None else self.args.default_seed
            response_format = self._resolve_response_format(request.response_format)
            output_format = self._resolve_output_format(request.output_format)

            logger.info(
                "[%s] create_video size=%dx%d frames=%d fps=%d steps=%d",
                video_id,
                width,
                height,
                num_frames,
                fps,
                num_inference_steps,
            )
            logger.info(
                "[%s] Waiting for generate lock (locked=%s)",
                video_id,
                self._generate_lock.locked(),
            )
            async with self._generate_lock:
                time_start = time.perf_counter()
                generate_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._generate_video,
                        prompt=request.prompt,
                        video_id=video_id,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        nvext=nvext,
                    )
                )
                try:
                    video_path, generation_time = await asyncio.shield(generate_task)
                except asyncio.CancelledError:
                    logger.info(
                        "[%s] Request cancelled; waiting for FastVideo generation "
                        "thread before releasing lock",
                        video_id,
                    )
                    cancelled_video_path: Path | None = None
                    while True:
                        try:
                            cancelled_video_path, _ = await asyncio.shield(
                                generate_task
                            )
                            break
                        except asyncio.CancelledError:
                            if not generate_task.done():
                                continue
                            if generate_task.cancelled():
                                break
                            try:
                                cancelled_video_path, _ = generate_task.result()
                            except Exception:
                                logger.exception(
                                    "[%s] FastVideo generation failed after "
                                    "cancellation",
                                    video_id,
                                )
                            break
                        except Exception:
                            logger.exception(
                                "[%s] FastVideo generation failed after cancellation",
                                video_id,
                            )
                            break
                    self._cleanup_staging_file(cancelled_video_path, video_id)
                    raise
                elapsed = time.perf_counter() - time_start

            if generation_time is not None:
                logger.info(
                    "[%s] FastVideo generation time: %.2fs",
                    video_id,
                    generation_time,
                )
            logger.info("[%s] Request finished in %.2fs", video_id, elapsed)

            yield VideoCreateResponse(
                id=video_id,
                created=created_ts,
                model=request.model,
                data=[
                    await self._make_video_data(
                        video_id=video_id,
                        output_format=output_format,
                        response_format=response_format,
                        video_path=video_path,
                    )
                ],
                inference_time_s=time.time() - started_at,
            ).model_dump()
        except asyncio.CancelledError:
            self._cleanup_staging_file(video_path, video_id)
            raise
        except Exception as exc:
            logger.exception("[%s] Generation failed", video_id)
            yield VideoCreateResponse(
                id=video_id,
                created=created_ts,
                model=request.model,
                status="failed",
                progress=0,
                data=[],
                error=str(exc),
                inference_time_s=time.time() - started_at,
            ).model_dump()


# ── Dynamo wiring ─────────────────────────────────────────────────────────────


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Videos,
            endpoint,
            model_name,
            model_name,
            worker_type=WorkerType.Aggregated,
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
        "--model-path",
        dest="model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--num-gpus", type=int, default=1, dest="num_gpus")
    parser.add_argument(
        "--discovery-backend",
        choices=("etcd", "file", "mem", "kubernetes"),
        default=None,
        help="Dynamo discovery backend. Defaults to env, then kubernetes/file.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default=DEFAULT_ATTENTION_BACKEND,
        dest="attention_backend",
        help="Attention backend to set via FASTVIDEO_ATTENTION_BACKEND.",
    )
    parser.add_argument(
        "--vsa-sparsity",
        type=float,
        default=DEFAULT_VSA_SPARSITY,
        help="VSA_sparsity value routed through PipelineSelection.experimental.",
    )
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        dest="enable_optimizations",
        help="Backward-compatible shortcut for --torch-compile --fp4-quantization.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        dest="torch_compile",
        help="Enable torch.compile through FastVideo CompileConfig.",
    )
    parser.add_argument(
        "--torch-compile-backend",
        default=DEFAULT_TORCH_COMPILE_BACKEND,
        help="torch.compile backend when --torch-compile is enabled.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=TORCH_COMPILE_MODE_CHOICES,
        default=DEFAULT_TORCH_COMPILE_MODE,
        help="torch.compile mode when --torch-compile is enabled.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--torch-compile-fullgraph",
        env_var="DYN_FASTVIDEO_TORCH_COMPILE_FULLGRAPH",
        dest="torch_compile_fullgraph",
        default=True,
        help="Enable torch.compile fullgraph mode.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--torch-compile-dynamic",
        env_var="DYN_FASTVIDEO_TORCH_COMPILE_DYNAMIC",
        dest="torch_compile_dynamic",
        default=False,
        help="Enable dynamic shapes for torch.compile.",
    )
    parser.add_argument(
        "--fp4-quantization",
        action="store_true",
        dest="fp4_quantization",
        help="Request NVFP4 transformer quantization through QuantizationConfig.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--use-fsdp-inference",
        env_var="DYN_FASTVIDEO_USE_FSDP_INFERENCE",
        dest="use_fsdp_inference",
        default=False,
        help="Enable FastVideo FSDP inference.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--dit-cpu-offload",
        env_var="DYN_FASTVIDEO_DIT_CPU_OFFLOAD",
        dest="dit_cpu_offload",
        default=True,
        help="Enable DiT CPU offload.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--dit-layerwise-offload",
        env_var="DYN_FASTVIDEO_DIT_LAYERWISE_OFFLOAD",
        dest="dit_layerwise_offload",
        default=True,
        help="Enable DiT layerwise CPU offload.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--vae-cpu-offload",
        env_var="DYN_FASTVIDEO_VAE_CPU_OFFLOAD",
        dest="vae_cpu_offload",
        default=True,
        help="Enable VAE CPU offload.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--image-encoder-cpu-offload",
        env_var="DYN_FASTVIDEO_IMAGE_ENCODER_CPU_OFFLOAD",
        dest="image_encoder_cpu_offload",
        default=True,
        help="Enable image encoder CPU offload.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--text-encoder-cpu-offload",
        env_var="DYN_FASTVIDEO_TEXT_ENCODER_CPU_OFFLOAD",
        dest="text_encoder_cpu_offload",
        default=True,
        help="Enable text encoder CPU offload.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--pin-cpu-memory",
        env_var="DYN_FASTVIDEO_PIN_CPU_MEMORY",
        dest="pin_cpu_memory",
        default=True,
        help="Pin host memory for CPU offload transfers.",
    )
    add_negatable_bool_argument(
        parser,
        flag_name="--disable-autocast",
        env_var="DYN_FASTVIDEO_DISABLE_AUTOCAST",
        dest="disable_autocast",
        default=False,
        help="Disable autocast in FastVideo denoising/decoding paths.",
    )
    parser.add_argument("--default-size", default=DEFAULT_SIZE)
    parser.add_argument("--default-seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--default-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--default-num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument(
        "--default-num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--default-guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
    )
    parser.add_argument("--default-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-video-width",
        type=int,
        default=DEFAULT_MAX_VIDEO_WIDTH,
        help="Maximum request video width accepted by the worker.",
    )
    parser.add_argument(
        "--max-video-height",
        type=int,
        default=DEFAULT_MAX_VIDEO_HEIGHT,
        help="Maximum request video height accepted by the worker.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated MP4 staging files.",
    )

    args = parser.parse_args()
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be > 0")
    if args.max_video_width <= 0:
        parser.error("--max-video-width must be > 0")
    if args.max_video_height <= 0:
        parser.error("--max-video-height must be > 0")
    return args


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = args.discovery_backend or os.environ.get(
        "DYN_DISCOVERY_BACKEND"
    )
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, "tcp")
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
