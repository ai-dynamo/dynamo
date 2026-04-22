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
DEFAULT_SECONDS = 5


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


def to_mp4(
    frames: "torch.Tensor",
    fps: float,
    interval: float | None = None,
) -> "bytes | list[bytes]":
    """Encode a frame tensor to MP4 bytes.

    Args:
        frames:   UInt8 tensor of shape (T, H, W, C) with values in [0, 255].
        fps:      Frame rate for the output video.
        interval: If given, split the video into chunks of at most this many
                  seconds.  Returns a list of MP4 byte strings, one per chunk.
                  If None, returns a single MP4 byte string.

    Returns:
        A single ``bytes`` object when *interval* is None, or a ``list[bytes]``
        with one entry per chunk when *interval* is provided.

    Raises:
        ImportError: If torchvision is not installed.
        ValueError:  If *frames* is empty or *fps* is not positive.
    """
    try:
        from torchvision.io import write_video
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for video encoding. "
            "Install with: pip install torchvision"
        ) from exc

    import tempfile

    if frames.shape[0] == 0:
        raise ValueError("frames tensor is empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    def _encode_chunk(chunk: "torch.Tensor") -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            write_video(tmp_path, chunk, fps=fps, video_codec="libx264")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    if interval is None:
        return _encode_chunk(frames)

    frames_per_chunk = max(1, int(interval * fps))
    chunks = [
        frames[start : start + frames_per_chunk]
        for start in range(0, frames.shape[0], frames_per_chunk)
    ]
    print(
        f"Frames per chunk: {frames_per_chunk}, total chunks: {len(chunks)}, total frames: {frames.shape[0]}"
    )
    return [_encode_chunk(chunk) for chunk in chunks]


def to_jpeg(
    frames: "torch.Tensor",
    quality: int = 75,
) -> "list[bytes]":
    """Encode a frame tensor to a list of JPEG byte strings, one per frame.

    Args:
        frames:  UInt8 tensor of shape (T, H, W, C) with values in [0, 255].
        quality: JPEG quality, 1–100 (default: 75).

    Returns:
        List of ``bytes`` objects, one JPEG-encoded image per frame.

    Raises:
        ImportError: If torchvision is not installed.
        ValueError:  If *frames* is empty.
    """
    try:
        from torchvision.io import encode_jpeg
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for JPEG encoding. "
            "Install with: pip install torchvision"
        ) from exc

    if frames.shape[0] == 0:
        raise ValueError("frames tensor is empty")

    result = []
    for frame in frames:
        # encode_jpeg expects (C, H, W) uint8
        chw = frame.permute(2, 0, 1).contiguous()
        jpeg_tensor = encode_jpeg(chw, quality=quality)
        result.append(jpeg_tensor.numpy().tobytes())
    return result


# ── Backend ───────────────────────────────────────────────────────────────────


class VideoStreamBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.video_file_path: str | None = args.video_file_path
        self.output_format: str = args.output_format  # "jpeg" or "mp4"
        self.mp4_interval: float | None = args.mp4_interval

        self.frames: torch.Tensor | None = None
        self.video_fps: float = 0.0
        self.jpeg_chunks: list[bytes] | None = None
        self.mp4_chunks: list[list[bytes]] | None = None

    async def initialize_model(self) -> None:
        """Load model weights and warm up the generation pipeline."""
        self.frames, self.video_fps = await asyncio.to_thread(
            load_video_frames, self.video_file_path
        )

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(NvCreateVideoRequest, NvVideosResponse)
    async def generate(self, request: NvCreateVideoRequest):
        """
        Streaming endpoint — yields one NvVideosResponse per output unit.

        In JPEG mode each response carries a single frame; in MP4 mode each
        response carries one MP4 clip (the whole clip, or one interval-length
        chunk when --mp4-interval is set).  The 'status' field is
        "in_progress" for all but the last response, which uses "completed".
        """
        nvext = request.nvext
        output_format = request.output_format or self.output_format
        if output_format not in ["jpeg", "mp4"]:
            raise ValueError(f"Invalid output format: {output_format}")

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

        fps = nvext.fps if nvext and nvext.fps is not None else self.video_fps
        seconds = request.seconds or DEFAULT_SECONDS
        num_frames = (
            nvext.num_frames
            if nvext and nvext.num_frames is not None
            else int(fps * seconds)
        )
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(time.time())

        available = self.frames.shape[0]
        if num_frames > available:
            logger.warning(
                "[%s] Requested %d frames but only %d available, clamping",
                video_id,
                num_frames,
                available,
            )
            num_frames = available
        frames_slice = self.frames[:num_frames]

        logger.info(
            "[%s] generate: prompt='%s...' size=%s frames=%d fps=%d format=%s",
            video_id,
            request.prompt[:60],
            size,
            num_frames,
            fps,
            self.output_format,
        )

        t = time.perf_counter()

        if output_format == "mp4":
            if self.mp4_chunks is None:
                encoded = await asyncio.to_thread(
                    to_mp4, frames_slice, fps, self.mp4_interval
                )
                self.mp4_chunks = encoded if isinstance(encoded, list) else [encoded]
            # Add artificial generation delay
            chunks = [
                (chunk, max(self.mp4_interval - 0.1, 0)) for chunk in self.mp4_chunks
            ]
        else:
            if self.jpeg_chunks is None:
                encoded = await asyncio.to_thread(to_jpeg, frames_slice)
                self.jpeg_chunks = encoded if isinstance(encoded, list) else [encoded]
            chunks = [(chunk, 0) for chunk in self.jpeg_chunks]

        total = len(chunks)
        for i, (chunk_bytes, interval) in enumerate(chunks):
            await asyncio.sleep(interval)
            elapsed = time.perf_counter() - t
            is_last = i == total - 1
            logger.debug(
                "[%s] chunk %d/%d ready in %.3fs",
                video_id,
                i + 1,
                total,
                elapsed,
            )
            yield NvVideosResponse(
                id=video_id,
                model=request.model,
                status="completed" if is_last else "in_progress",
                progress=int((i + 1) / total * 100),
                created=created_ts,
                data=[
                    VideoData(
                        output_format=output_format,
                        b64_json=base64.b64encode(chunk_bytes).decode(),
                    )
                ],
                inference_time_s=elapsed,
            ).model_dump()

        logger.info(
            "[%s] Streaming request finished (%d chunks, format=%s)",
            video_id,
            total,
            output_format,
        )


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
    parser.add_argument(
        "--output-format",
        choices=["jpeg", "mp4"],
        default="jpeg",
        dest="output_format",
        help="Output format: 'jpeg' streams one frame per response; "
        "'mp4' streams one MP4 clip per response (default: jpeg)",
    )
    parser.add_argument(
        "--mp4-interval",
        type=float,
        default=1,
        dest="mp4_interval",
        metavar="SECONDS",
        help="Only used with --output-format mp4. Split the clip into chunks "
        "of at most this many seconds. Omit to return the whole clip as one response.",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND", "etcd")
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
