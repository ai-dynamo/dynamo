# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import uvloop

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="backend")

CMAF_ANNOTATION = "experimental_binary_cmaf"
CMAF_METADATA_TAG = "cmaf:metadata"
CMAF_INIT_TAG = "cmaf:init"
CMAF_SEGMENT_PREFIX = "cmaf:segment:"
DEFAULT_NAMESPACE = "dynamo"
VIDEO_CODEC = "avc1.4d401f"
AUDIO_CODEC = "mp4a.40.2"


@dataclass(frozen=True)
class CmafSegment:
    index: int
    duration_s: float
    payload: bytes


@dataclass(frozen=True)
class PackagedCmafAsset:
    init_payload: bytes
    segments: list[CmafSegment]
    source_buffer_mime_type: str
    has_audio: bool
    target_duration_seconds: int


class SyntheticBinaryCmafBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.source_mp4 = Path(args.source_mp4).expanduser().resolve()
        self.segment_seconds: int = args.segment_seconds
        self.emit_cadence_s = args.emit_cadence_ms / 1000.0
        self._packaged_asset: PackagedCmafAsset | None = None
        self._workdir = Path(tempfile.mkdtemp(prefix="dynamo_cmaf_binary_"))

    async def initialize(self) -> None:
        await asyncio.to_thread(self._package_source)

    def _package_source(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required for the CMAF worker but was not found in PATH"
            )
        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe is required for the CMAF worker but was not found in PATH"
            )
        if not self.source_mp4.exists():
            raise FileNotFoundError(f"Source MP4 not found: {self.source_mp4}")

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(self.source_mp4),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe_json = json.loads(probe.stdout or "{}")
        has_audio = any(
            stream.get("codec_type") == "audio"
            for stream in probe_json.get("streams", [])
        )

        manifest_path = self._workdir / "manifest.m3u8"
        init_path = self._workdir / "init.mp4"
        segment_pattern = self._workdir / "segment_%03d.m4s"

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.source_mp4),
            "-map",
            "0:v:0",
        ]
        if has_audio:
            cmd.extend(["-map", "0:a:0?"])

        gop = max(1, self.segment_seconds * 24)
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-level:v",
                "3.1",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-r",
                "24",
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-sc_threshold",
                "0",
                "-force_key_frames",
                f"expr:gte(t,n_forced*{self.segment_seconds})",
            ]
        )
        if has_audio:
            cmd.extend(
                [
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
        else:
            cmd.append("-an")

        cmd.extend(
            [
                "-f",
                "hls",
                "-hls_time",
                str(self.segment_seconds),
                "-hls_playlist_type",
                "vod",
                "-hls_flags",
                "independent_segments",
                "-hls_segment_type",
                "fmp4",
                "-hls_fmp4_init_filename",
                init_path.name,
                "-hls_segment_filename",
                str(segment_pattern),
                str(manifest_path),
            ]
        )

        subprocess.run(cmd, check=True)

        if not init_path.exists():
            raise RuntimeError(
                f"ffmpeg did not create expected init segment: {init_path}"
            )

        segments: list[CmafSegment] = []
        lines = manifest_path.read_text().splitlines()
        pending_duration: float | None = None
        for line in lines:
            if line.startswith("#EXTINF:"):
                pending_duration = float(line.split(":", 1)[1].split(",", 1)[0])
                continue
            if not line or line.startswith("#"):
                continue
            if pending_duration is None:
                raise RuntimeError(
                    f"Manifest referenced segment without EXTINF: {line}"
                )
            segment_path = self._workdir / line
            if not segment_path.exists():
                raise RuntimeError(
                    f"Manifest referenced missing segment: {segment_path}"
                )
            segments.append(
                CmafSegment(
                    index=len(segments),
                    duration_s=pending_duration,
                    payload=segment_path.read_bytes(),
                )
            )
            pending_duration = None

        if not segments:
            raise RuntimeError("ffmpeg packaging produced no media segments")

        codecs = [VIDEO_CODEC]
        if has_audio:
            codecs.append(AUDIO_CODEC)
        source_buffer_mime_type = f'video/mp4; codecs="{", ".join(codecs)}"'

        self._packaged_asset = PackagedCmafAsset(
            init_payload=init_path.read_bytes(),
            segments=segments,
            source_buffer_mime_type=source_buffer_mime_type,
            has_audio=has_audio,
            target_duration_seconds=max(
                self.segment_seconds,
                math.ceil(max(segment.duration_s for segment in segments)),
            ),
        )
        logger.info(
            "Prepared CMAF asset: source=%s segments=%d has_audio=%s mime=%s",
            self.source_mp4,
            len(segments),
            has_audio,
            source_buffer_mime_type,
        )

    def _select_segments(self, requested_seconds: int | None) -> list[CmafSegment]:
        assert self._packaged_asset is not None
        if requested_seconds is None or requested_seconds <= 0:
            return self._packaged_asset.segments
        segment_count = max(1, math.ceil(requested_seconds / self.segment_seconds))
        return self._packaged_asset.segments[:segment_count]

    def _metadata_bytes(self, segments: list[CmafSegment]) -> bytes:
        assert self._packaged_asset is not None
        payload = {
            "protocol": "dynamo-video-binary-cmaf-v1",
            "mime_type": "video/mp4",
            "source_buffer_mime_type": self._packaged_asset.source_buffer_mime_type,
            "video_codec": VIDEO_CODEC,
            "audio_codec": AUDIO_CODEC if self._packaged_asset.has_audio else None,
            "has_audio": self._packaged_asset.has_audio,
            "target_duration_seconds": self._packaged_asset.target_duration_seconds,
            "segment_count": len(segments),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @dynamo_endpoint(NvCreateVideoRequest, NvVideosResponse)
    async def create_video(self, request: NvCreateVideoRequest):
        if self._packaged_asset is None:
            raise RuntimeError("CMAF asset was not initialized")

        annotations = (
            request.nvext.annotations
            if request.nvext and request.nvext.annotations
            else []
        )
        if CMAF_ANNOTATION not in annotations:
            raise ValueError(
                "This synthetic backend only serves requests annotated with experimental_binary_cmaf"
            )

        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(time.time())
        segments = self._select_segments(request.seconds)
        metadata_b64 = base64.b64encode(self._metadata_bytes(segments)).decode("ascii")
        init_b64 = base64.b64encode(self._packaged_asset.init_payload).decode("ascii")

        logger.info(
            "[%s] Starting binary CMAF stream for prompt='%s' segment_count=%d",
            video_id,
            request.prompt[:60],
            len(segments),
        )

        yield NvVideosResponse(
            id=video_id,
            model=request.model,
            created=created_ts,
            status="in_progress",
            progress=0,
            data=[VideoData(url=CMAF_METADATA_TAG, b64_json=metadata_b64)],
        ).model_dump()

        yield NvVideosResponse(
            id=video_id,
            model=request.model,
            created=created_ts,
            status="in_progress",
            progress=1,
            data=[VideoData(url=CMAF_INIT_TAG, b64_json=init_b64)],
        ).model_dump()

        for index, segment in enumerate(segments):
            await asyncio.sleep(self.emit_cadence_s)
            progress = min(99, int(((index + 1) / max(1, len(segments))) * 100))
            payload_b64 = base64.b64encode(segment.payload).decode("ascii")
            yield NvVideosResponse(
                id=video_id,
                model=request.model,
                created=created_ts,
                status="in_progress",
                progress=progress,
                data=[
                    VideoData(
                        url=f"{CMAF_SEGMENT_PREFIX}{segment.index}",
                        b64_json=payload_b64,
                    )
                ],
            ).model_dump()

        logger.info("[%s] Binary CMAF stream completed", video_id)


async def _register_model(endpoint, model_name: str) -> None:
    await register_llm(
        ModelInput.Text,  # type: ignore[attr-defined]
        ModelType.Videos,
        endpoint,
        model_name,
        model_name,
    )
    logger.info("Registered synthetic binary CMAF model: %s", model_name)


def _worker_namespace() -> str:
    return os.environ.get("DYN_NAMESPACE", DEFAULT_NAMESPACE)


async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace = _worker_namespace()
    component_name = "backend"
    endpoint_name = "generate"
    endpoint = runtime.endpoint(f"{namespace}.{component_name}.{endpoint_name}")

    backend = SyntheticBinaryCmafBackend(args)
    await backend.initialize()

    logger.info("Serving endpoint %s/%s/%s", namespace, component_name, endpoint_name)
    await asyncio.gather(
        endpoint.serve_endpoint(backend.create_video),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic Dynamo worker for single-POST binary CMAF video streaming"
    )
    parser.add_argument("--source-mp4", required=True, help="Path to a source MP4 file")
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=2,
        help="Target CMAF fragment duration in seconds",
    )
    parser.add_argument(
        "--emit-cadence-ms",
        type=int,
        default=750,
        help="Delay between emitted media fragments in milliseconds",
    )
    parser.add_argument(
        "--model",
        default="synthetic-binary-cmaf-video-stream",
        help="Model name to register with the Dynamo frontend",
    )
    return parser


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    request_plane = os.environ.get("DYN_REQUEST_PLANE", "tcp")
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Using request plane: %s", request_plane)
    logger.info("Resolved worker namespace: %s", _worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, request_plane)
    await worker(runtime, args)


if __name__ == "__main__":
    uvloop.install()
    parser = create_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
