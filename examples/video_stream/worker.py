#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Streaming Video Generation Worker for Dynamo

Registers a VideoStreamBackend as a Dynamo backend endpoint compatible with
the /v1/videos/stream frontend endpoint.  Unlike the non-streaming diffusion
worker (examples/diffusers/worker.py), this worker yields one response *per
frame* — each carrying a single JPEG-encoded frame as data[0].b64_json.  The
HTTP frontend reassembles these into an MJPEG multipart/x-mixed-replace stream.

Response protocol (matches NvVideosResponse in lib/llm/src/protocols/openai/videos.rs):
  Each yielded dict must contain:
    id:       unique request identifier
    object:   "video"
    model:    model name echoed from request
    status:   "in_progress" for intermediate frames, "completed" for the last
    progress: 0-100 (percentage of frames generated)
    created:  Unix timestamp
    data:     [{"b64_json": "<base64-encoded JPEG>"}]

Usage:
  python worker.py [--model MODEL] [--num-gpus N]

Options:
  --model     Model identifier (default: video-stream-model)
  --num-gpus  Number of GPUs (default: 1)

Request format (sent to /v1/videos/stream):
  prompt:   text description of the desired video
  model:    model identifier (must match what the worker registered)
  size:     "WxH" string, e.g. "832x480" (default: "832x480")
  seconds:  clip duration in seconds (default: 5)
  nvext:
    fps:                frames per second (default: 25)
    num_frames:         total frames; overrides fps * seconds when set
    num_inference_steps diffusion steps per frame (default: 1)
    guidance_scale:     CFG scale (default: 1.0)
    seed:               RNG seed (optional)
    negative_prompt:    text to avoid (optional)
"""

import argparse
import asyncio
import base64
import logging
import os
import time
import uuid

import torch
import uvloop

from dynamo.common.protocols.video_protocol import (  # type: ignore[import]
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "video-stream-model"
DEFAULT_SIZE = "832x480"
DEFAULT_FPS = 25
DEFAULT_SECONDS = 5
DEFAULT_NUM_INFERENCE_STEPS = 1
DEFAULT_GUIDANCE_SCALE = 1.0


# ── Namespace helper ──────────────────────────────────────────────────────────


def _get_worker_namespace() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


# ── Video file loading ────────────────────────────────────────────────────────


def load_video_frames(path: str) -> tuple["torch.Tensor", float]:
    """Read a video file and decode it into a frame tensor.

    Args:
        path: Absolute or relative path to the video file.

    Returns:
        frames: UInt8 tensor of shape (T, H, W, C) with values in [0, 255].
        fps:    Frame rate reported by the container.

    Raises:
        ImportError: If torchvision is not installed.
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be decoded.
    """
    try:
        from torchvision.io import read_video
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for video decoding. "
            "Install with: pip install torchvision"
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        # read_video returns uint8 (T, H, W, C), audio tensor, and info dict.
        frames_uint8, _, info = read_video(path, output_format="THWC", pts_unit="sec")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video file '{path}': {exc}") from exc

    fps: float = float(info.get("video_fps", 0.0))
    frames = frames_uint8

    logger.info(
        "Loaded video: path=%s frames=%d size=%dx%d fps=%.2f",
        path,
        frames.shape[0],
        frames.shape[2],
        frames.shape[1],
        fps,
    )
    return frames, fps


# ── Backend ───────────────────────────────────────────────────────────────────


class VideoStreamBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.video_file_path: str | None = args.video_file_path

        self.frames: torch.Tensor | None = None
        self.video_fps: float = 0.0

        # One request at a time — generation pipeline is not re-entrant
        self._generate_lock = asyncio.Lock()

    async def initialize_model(self) -> None:
        """Load model weights and warm up the generation pipeline."""
        self.frames, self.video_fps = await asyncio.to_thread(
            load_video_frames, self.video_file_path
        )

    # ── Frame generation ──────────────────────────────────────────────────────

    def _generate_frame(
        self,
        prompt: str,
        frame_index: int,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        negative_prompt: str | None,
    ) -> bytes:
        """
        Generate a single video frame and return it as raw JPEG bytes.

        The returned bytes must be a valid JPEG image — the HTTP frontend
        writes them directly into a multipart/x-mixed-replace boundary
        without further encoding.
        """
        raise NotImplementedError

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(NvCreateVideoRequest, NvVideosResponse)
    async def generate(self, request: NvCreateVideoRequest):
        """
        Streaming endpoint — yields one NvVideosResponse per frame.

        Each response carries a single JPEG frame in data[0].b64_json.
        The 'status' field is "in_progress" for all frames except the last,
        which uses "completed".  The 'progress' field advances from 0 to 100
        across the sequence so callers can display a progress indicator.
        """
        nvext = request.nvext

        size = request.size or DEFAULT_SIZE
        try:
            width_str, height_str = size.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid size format '{size}', expected 'WxH'") from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid size '{size}', width and height must be positive"
            )

        fps = nvext.fps if nvext and nvext.fps is not None else DEFAULT_FPS
        seconds = request.seconds or DEFAULT_SECONDS
        num_frames = (
            nvext.num_frames
            if nvext and nvext.num_frames is not None
            else fps * seconds
        )
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        num_inference_steps = (
            nvext.num_inference_steps
            if nvext and nvext.num_inference_steps is not None
            else DEFAULT_NUM_INFERENCE_STEPS
        )
        guidance_scale = (
            nvext.guidance_scale
            if nvext and nvext.guidance_scale is not None
            else DEFAULT_GUIDANCE_SCALE
        )
        seed = nvext.seed if nvext else None
        negative_prompt = nvext.negative_prompt if nvext else None

        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(time.time())

        logger.info(
            "[%s] generate: prompt='%s...' size=%s frames=%d fps=%d steps=%d",
            video_id,
            request.prompt[:60],
            size,
            num_frames,
            fps,
            num_inference_steps,
        )

        async with self._generate_lock:
            for frame_idx in range(num_frames):
                t = time.perf_counter()

                jpeg_bytes = await asyncio.to_thread(
                    self._generate_frame,
                    prompt=request.prompt,
                    frame_index=frame_idx,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    negative_prompt=negative_prompt,
                )

                elapsed = time.perf_counter() - t
                is_last = frame_idx == num_frames - 1
                progress = int((frame_idx + 1) / num_frames * 100)

                logger.debug(
                    "[%s] frame %d/%d done in %.3fs",
                    video_id,
                    frame_idx + 1,
                    num_frames,
                    elapsed,
                )

                yield NvVideosResponse(
                    id=video_id,
                    model=request.model,
                    status="completed" if is_last else "in_progress",
                    progress=progress,
                    created=created_ts,
                    data=[VideoData(b64_json=base64.b64encode(jpeg_bytes).decode())],
                    inference_time_s=elapsed,
                ).model_dump()

        logger.info("[%s] Streaming request finished (%d frames)", video_id, num_frames)


# ── Dynamo wiring ─────────────────────────────────────────────────────────────


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Videos,
            endpoint,
            "Qwen/Qwen3-0.6B",  # Mock model needs a real path.
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

    backend = VideoStreamBackend(args)
    await backend.initialize_model()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.generate),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming Video Generation Worker for Dynamo"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--video-file-path",
        default=os.path.join(os.path.dirname(__file__), "sample.mp4"),
        dest="video_file_path",
        help="Path to a video file to use as the source for frame generation",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, "tcp")
    await backend_worker(runtime, args)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG
        if os.environ.get("DYN_LOG_LEVEL") == "DEBUG"
        else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(args))
