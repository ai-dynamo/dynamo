#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import imageio.v2 as iio
import numpy as np
import uvloop

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.common.utils.video_utils import encode_to_mp4_bytes
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "synthetic-video-stream"
DEFAULT_FRAGMENT_SECONDS = 2.0
DEFAULT_EMIT_CADENCE_MS = 750
DEFAULT_COMPONENT = "backend"
DEFAULT_ENDPOINT = "generate"


def _get_worker_namespace() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


class SyntheticVideoStreamBackend:
    def __init__(
        self,
        *,
        source_mp4: Path,
        fragment_seconds: float,
        emit_cadence_ms: int,
        model_name: str,
    ) -> None:
        self.source_mp4 = source_mp4
        self.fragment_seconds = fragment_seconds
        self.emit_cadence_ms = emit_cadence_ms
        self.model_name = model_name
        self._clips: list[bytes] = []
        self._fps: int = 0

    async def initialize(self) -> None:
        clips, fps = await asyncio.to_thread(self._prepare_replayable_clips)
        self._clips = clips
        self._fps = fps
        logger.info(
            "Prepared %d replayable MP4 clips from %s at %d fps",
            len(self._clips),
            self.source_mp4,
            self._fps,
        )

    def _prepare_replayable_clips(self) -> tuple[list[bytes], int]:
        if not self.source_mp4.exists():
            raise FileNotFoundError(f"Source MP4 not found: {self.source_mp4}")

        reader = iio.get_reader(str(self.source_mp4), format="ffmpeg")
        try:
            metadata = reader.get_meta_data()
            fps = int(round(float(metadata.get("fps") or 0)))
            if fps <= 0:
                raise ValueError(
                    f"Unable to determine FPS for source video: {self.source_mp4}"
                )

            frames = [np.asarray(frame) for frame in reader]
        finally:
            reader.close()

        if not frames:
            raise ValueError(f"Source video contains no frames: {self.source_mp4}")

        frames_np = np.stack(frames, axis=0)
        frames_per_clip = max(1, int(round(self.fragment_seconds * fps)))

        clips: list[bytes] = []
        for start in range(0, len(frames_np), frames_per_clip):
            clip_frames = frames_np[start : start + frames_per_clip]
            if len(clip_frames) == 0:
                continue

            # POC shortcut: pre-encode standalone MP4 clips at startup instead of
            # building a low-latency fragmented MP4 muxer. This intentionally
            # validates the RFC's "replayable unit" requirement first: every
            # streamed chunk is independently playable as its own MP4 clip.
            clips.append(encode_to_mp4_bytes(clip_frames, fps=fps))

        if not clips:
            raise ValueError(
                f"Unable to create replayable clips from {self.source_mp4}"
            )

        return clips, fps

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = NvCreateVideoRequest(**request)
        request_id = context.id() or f"video-{uuid.uuid4().hex}"
        created = int(time.time())
        total = len(self._clips)

        logger.info(
            "Synthetic stream request %s: model=%s prompt=%r clips=%d cadence_ms=%d",
            request_id,
            req.model,
            req.prompt,
            total,
            self.emit_cadence_ms,
        )

        for index, clip_bytes in enumerate(self._clips):
            if context.is_stopped() or context.is_killed():
                logger.info("Request %s cancelled before clip %d", request_id, index)
                raise asyncio.CancelledError

            progress = int(((index + 1) / total) * 100)
            is_last = index == total - 1
            response = NvVideosResponse(
                id=request_id,
                object="video",
                model=req.model,
                status="completed" if is_last else "in_progress",
                progress=progress,
                created=created,
                data=[
                    VideoData(
                        b64_json=base64.b64encode(clip_bytes).decode("ascii"),
                    )
                ],
            )

            logger.info(
                "Request %s emitting clip %d/%d (%d bytes)",
                request_id,
                index + 1,
                total,
                len(clip_bytes),
            )
            yield response.model_dump()

            if not is_last:
                await asyncio.sleep(self.emit_cadence_ms / 1000.0)


async def _register_model(endpoint, model_name: str) -> None:
    await register_llm(
        ModelInput.Text,  # type: ignore[attr-defined]
        ModelType.Videos,
        endpoint,
        model_name,
        model_name,
    )
    logger.info("Registered synthetic streaming model: %s", model_name)


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    endpoint = runtime.endpoint(
        f"{namespace_name}.{DEFAULT_COMPONENT}.{DEFAULT_ENDPOINT}"
    )
    logger.info(
        "Serving synthetic streaming endpoint %s/%s/%s",
        namespace_name,
        DEFAULT_COMPONENT,
        DEFAULT_ENDPOINT,
    )

    backend = SyntheticVideoStreamBackend(
        source_mp4=args.source_mp4,
        fragment_seconds=args.fragment_seconds,
        emit_cadence_ms=args.emit_cadence_ms,
        model_name=args.model,
    )
    await backend.initialize()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.generate),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic MP4 chunk-source harness for /v1/videos/stream"
    )
    parser.add_argument(
        "--source-mp4",
        type=Path,
        required=True,
        help="Path to the prerecorded MP4 used as the synthetic source stream",
    )
    parser.add_argument(
        "--fragment-seconds",
        type=float,
        default=DEFAULT_FRAGMENT_SECONDS,
        help=f"Duration of each replayable MP4 clip (default: {DEFAULT_FRAGMENT_SECONDS})",
    )
    parser.add_argument(
        "--emit-cadence-ms",
        type=int,
        default=DEFAULT_EMIT_CADENCE_MS,
        help=f"Delay between emitted clips in milliseconds (default: {DEFAULT_EMIT_CADENCE_MS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Served model name for registration (default: {DEFAULT_MODEL_NAME})",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(_parse_args()))
