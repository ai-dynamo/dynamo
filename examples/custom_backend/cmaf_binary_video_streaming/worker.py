#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import uvloop

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "synthetic-binary-cmaf-video-stream"
DEFAULT_COMPONENT = "backend"
DEFAULT_ENDPOINT = "generate"
DEFAULT_SEGMENT_SECONDS = 2.0
DEFAULT_EMIT_CADENCE_MS = 750
PACKAGING_FPS = 30
PACKAGING_VIDEO_CODEC = "avc1.4d401f"
PACKAGING_AUDIO_CODEC = "mp4a.40.2"
VIDEO_STREAM_BINARY_CMAF_ANNOTATION = "experimental_binary_cmaf"
VIDEO_STREAM_BINARY_CMAF_METADATA_TAG = "cmaf:metadata"
VIDEO_STREAM_BINARY_CMAF_INIT_TAG = "cmaf:init"
VIDEO_STREAM_BINARY_CMAF_SEGMENT_PREFIX = "cmaf:segment:"


def _get_worker_namespace() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


@dataclass(frozen=True)
class PackagedSegment:
    segment_id: int
    duration_seconds: float
    filename: str
    bytes: bytes


@dataclass(frozen=True)
class PackagedStream:
    target_duration_seconds: int
    has_audio: bool
    source_buffer_mime_type: str
    metadata_bytes: bytes
    init_bytes: bytes
    segments: list[PackagedSegment]


class SyntheticBinaryCmafVideoBackend:
    def __init__(
        self,
        *,
        source_mp4: Path,
        segment_seconds: float,
        emit_cadence_ms: int,
        model_name: str,
    ) -> None:
        self.source_mp4 = source_mp4
        self.segment_seconds = segment_seconds
        self.emit_cadence_ms = emit_cadence_ms
        self.model_name = model_name
        self._packaged_stream: PackagedStream | None = None
        self._workspace: tempfile.TemporaryDirectory[str] | None = None

    @property
    def packaged_stream(self) -> PackagedStream:
        if self._packaged_stream is None:
            raise RuntimeError("Synthetic CMAF binary backend is not initialized")
        return self._packaged_stream

    async def initialize(self) -> None:
        self._packaged_stream = await asyncio.to_thread(self._package_source_into_cmaf)
        logger.info(
            "Prepared continuous CMAF binary stream from %s with %d segments, target_duration=%ds, has_audio=%s",
            self.source_mp4,
            len(self.packaged_stream.segments),
            self.packaged_stream.target_duration_seconds,
            self.packaged_stream.has_audio,
        )

    def _detect_audio_stream(self) -> bool:
        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe is required to inspect the source MP4 audio track for the CMAF binary POC"
            )

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(self.source_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or "ffprobe exited with a non-zero status"
            raise RuntimeError(f"ffprobe failed: {stderr}")
        return bool(result.stdout.strip())

    def _package_source_into_cmaf(self) -> PackagedStream:
        if not self.source_mp4.exists():
            raise FileNotFoundError(f"Source MP4 not found: {self.source_mp4}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to package CMAF fragments")

        has_audio = self._detect_audio_stream()

        self._workspace = tempfile.TemporaryDirectory(prefix="dynamo-binary-cmaf-")
        workspace = Path(self._workspace.name)
        manifest_path = workspace / "manifest.m3u8"
        segment_template = workspace / "segment_%05d.m4s"
        gop = max(1, int(round(self.segment_seconds * PACKAGING_FPS)))

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(self.source_mp4),
            "-map",
            "0:v:0",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-r",
            str(PACKAGING_FPS),
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-level:v",
            "3.1",
            "-preset",
            "veryfast",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{self.segment_seconds})",
        ]

        if has_audio:
            cmd.extend(
                [
                    "-map",
                    "0:a:0?",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                ]
            )

        cmd.extend(
            [
                "-f",
                "hls",
                "-hls_time",
                str(self.segment_seconds),
                "-hls_playlist_type",
                "vod",
                "-hls_list_size",
                "0",
                "-hls_segment_type",
                "fmp4",
                "-hls_fmp4_init_filename",
                "init.mp4",
                "-hls_flags",
                "independent_segments",
                "-hls_segment_filename",
                str(segment_template),
                str(manifest_path),
            ]
        )

        logger.info(
            "Packaging %s into continuous CMAF fragments with ffmpeg (has_audio=%s)",
            self.source_mp4,
            has_audio,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or "ffmpeg exited with a non-zero status"
            raise RuntimeError(f"ffmpeg packaging failed: {stderr}")

        manifest = manifest_path.read_text(encoding="utf-8")
        target_duration_seconds = 0
        init_filename: str | None = None
        segments: list[PackagedSegment] = []
        current_duration: float | None = None

        for raw_line in manifest.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-TARGETDURATION:"):
                target_duration_seconds = int(line.split(":", 1)[1])
                continue
            if line.startswith("#EXT-X-MAP:"):
                match = re.search(r'URI="([^"]+)"', line)
                if match:
                    init_filename = match.group(1)
                continue
            if line.startswith("#EXTINF:"):
                current_duration = float(line.split(":", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("#"):
                continue
            if current_duration is None:
                raise RuntimeError(
                    f"Found segment line without preceding EXTINF in {manifest_path}"
                )

            segment_path = workspace / line
            segments.append(
                PackagedSegment(
                    segment_id=len(segments),
                    duration_seconds=current_duration,
                    filename=line,
                    bytes=segment_path.read_bytes(),
                )
            )
            current_duration = None

        if init_filename is None:
            raise RuntimeError("ffmpeg manifest did not contain EXT-X-MAP")
        if not segments:
            raise RuntimeError("ffmpeg did not produce any CMAF segments")
        if target_duration_seconds <= 0:
            target_duration_seconds = int(
                math.ceil(max(segment.duration_seconds for segment in segments))
            )

        init_bytes = (workspace / init_filename).read_bytes()
        if not init_bytes:
            raise RuntimeError("ffmpeg produced an empty init.mp4 fragment")

        codec_list = [PACKAGING_VIDEO_CODEC]
        if has_audio:
            codec_list.append(PACKAGING_AUDIO_CODEC)
        source_buffer_mime_type = f'video/mp4; codecs="{",".join(codec_list)}"'
        metadata_bytes = json.dumps(
            {
                "mime_type": "video/mp4",
                "source_buffer_mime_type": source_buffer_mime_type,
                "video_codec": PACKAGING_VIDEO_CODEC,
                "audio_codec": PACKAGING_AUDIO_CODEC if has_audio else None,
                "has_audio": has_audio,
                "target_duration_seconds": target_duration_seconds,
                "segment_count": len(segments),
            },
            separators=(",", ":"),
        ).encode("utf-8")

        return PackagedStream(
            target_duration_seconds=target_duration_seconds,
            has_audio=has_audio,
            source_buffer_mime_type=source_buffer_mime_type,
            metadata_bytes=metadata_bytes,
            init_bytes=init_bytes,
            segments=segments,
        )

    def _require_cmaf_request(self, request: NvCreateVideoRequest) -> None:
        annotations = request.nvext.annotations if request.nvext else None
        if not annotations or VIDEO_STREAM_BINARY_CMAF_ANNOTATION not in annotations:
            raise ValueError(
                "synthetic CMAF binary worker requires the experimental_binary_cmaf annotation"
            )

    @staticmethod
    def _encode_bytes(payload: bytes) -> str:
        return base64.b64encode(payload).decode("ascii")

    @staticmethod
    def _build_response(
        *,
        request_id: str,
        model: str,
        created: int,
        status: str,
        progress: int,
        tag: str,
        payload: bytes,
    ) -> NvVideosResponse:
        return NvVideosResponse(
            id=request_id,
            object="video",
            model=model,
            status=status,
            progress=progress,
            created=created,
            data=[
                VideoData(
                    url=tag,
                    b64_json=SyntheticBinaryCmafVideoBackend._encode_bytes(payload),
                )
            ],
            error=None,
            inference_time_s=None,
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = NvCreateVideoRequest(**request)
        self._require_cmaf_request(req)

        request_id = context.id() or f"video-{uuid.uuid4().hex}"
        created = int(time.time())
        total_parts = 2 + len(self.packaged_stream.segments)

        logger.info(
            "Synthetic CMAF binary request %s: model=%s prompt=%r segments=%d cadence_ms=%d has_audio=%s",
            request_id,
            req.model,
            req.prompt,
            len(self.packaged_stream.segments),
            self.emit_cadence_ms,
            self.packaged_stream.has_audio,
        )

        emissions: list[tuple[str, bytes]] = [
            (
                VIDEO_STREAM_BINARY_CMAF_METADATA_TAG,
                self.packaged_stream.metadata_bytes,
            ),
            (VIDEO_STREAM_BINARY_CMAF_INIT_TAG, self.packaged_stream.init_bytes),
        ]
        emissions.extend(
            (
                f"{VIDEO_STREAM_BINARY_CMAF_SEGMENT_PREFIX}{segment.segment_id}",
                segment.bytes,
            )
            for segment in self.packaged_stream.segments
        )

        for index, (tag, payload) in enumerate(emissions):
            if context.is_stopped() or context.is_killed():
                logger.info("Request %s cancelled before payload %d", request_id, index)
                raise asyncio.CancelledError

            progress = int(round(((index + 1) / total_parts) * 100))
            is_last = index == total_parts - 1
            response = self._build_response(
                request_id=request_id,
                model=req.model,
                created=created,
                status="completed" if is_last else "in_progress",
                progress=progress,
                tag=tag,
                payload=payload,
            )

            logger.info(
                "Request %s emitting %s payload %d/%d (%d bytes)",
                request_id,
                tag,
                index + 1,
                total_parts,
                len(payload),
            )
            yield response.model_dump()

            if tag.startswith(VIDEO_STREAM_BINARY_CMAF_SEGMENT_PREFIX) and not is_last:
                await asyncio.sleep(self.emit_cadence_ms / 1000.0)


async def _register_model(endpoint, model_name: str) -> None:
    await register_llm(
        ModelInput.Text,  # type: ignore[attr-defined]
        ModelType.Videos,
        endpoint,
        model_name,
        model_name,
    )
    logger.info("Registered synthetic binary CMAF model: %s", model_name)


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    endpoint = runtime.endpoint(
        f"{namespace_name}.{DEFAULT_COMPONENT}.{DEFAULT_ENDPOINT}"
    )
    logger.info(
        "Serving synthetic binary CMAF endpoint %s/%s/%s",
        namespace_name,
        DEFAULT_COMPONENT,
        DEFAULT_ENDPOINT,
    )

    backend = SyntheticBinaryCmafVideoBackend(
        source_mp4=args.source_mp4,
        segment_seconds=args.segment_seconds,
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
        description="Synthetic CMAF-over-binary harness for /v1/videos/stream/binary/cmaf"
    )
    parser.add_argument(
        "--source-mp4",
        type=Path,
        required=True,
        help="Path to the prerecorded MP4 used as the synthetic source stream",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help=f"Duration of each CMAF media fragment in seconds (default: {DEFAULT_SEGMENT_SECONDS})",
    )
    parser.add_argument(
        "--emit-cadence-ms",
        type=int,
        default=DEFAULT_EMIT_CADENCE_MS,
        help=f"Delay between emitted media fragments in milliseconds (default: {DEFAULT_EMIT_CADENCE_MS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Served model name for registration (default: {DEFAULT_MODEL_NAME})",
    )
    args = parser.parse_args()

    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be positive")
    if args.emit_cadence_ms <= 0:
        parser.error("--emit-cadence-ms must be positive")

    return args


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
