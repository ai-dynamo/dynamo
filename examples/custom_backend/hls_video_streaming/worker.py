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

import uvloop
from pydantic import BaseModel

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "synthetic-hls-cmaf-video-stream"
DEFAULT_COMPONENT = "backend"
DEFAULT_GENERATE_ENDPOINT = "generate"
DEFAULT_PLAYLIST_ENDPOINT = "playlist"
DEFAULT_INIT_ENDPOINT = "init_fragment"
DEFAULT_SEGMENT_ENDPOINT = "segment"
DEFAULT_SEGMENT_SECONDS = 2.0
DEFAULT_EMIT_CADENCE_MS = 1000
DEFAULT_PLAYLIST_WINDOW_SEGMENTS = 4
DEFAULT_SESSION_TTL_SECONDS = 600
PACKAGING_FPS = 30
HLS_CMAF_ANNOTATION = "experimental_hls_cmaf"


def _get_worker_namespace() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


def _base64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class PackagedSegment:
    segment_id: int
    duration_seconds: float
    filename: str
    bytes: bytes


@dataclass(frozen=True)
class PackagedStream:
    target_duration_seconds: int
    init_filename: str
    init_bytes: bytes
    segments: list[PackagedSegment]


@dataclass
class StreamSession:
    session_id: str
    created: int
    started_at_monotonic: float
    last_access_monotonic: float
    expires_at: int
    visible_segments: int
    completed: bool


class HlsPlaylistRequest(BaseModel):
    session_id: str


class HlsPlaylistResponse(BaseModel):
    playlist: str


class HlsInitRequest(BaseModel):
    session_id: str


class HlsSegmentRequest(BaseModel):
    session_id: str
    segment_id: int


class HlsFragmentResponse(BaseModel):
    bytes: bytes


class SyntheticHlsVideoBackend:
    def __init__(
        self,
        *,
        source_mp4: Path,
        segment_seconds: float,
        emit_cadence_ms: int,
        playlist_window_segments: int,
        session_ttl_seconds: int,
        model_name: str,
        namespace: str,
        component: str,
        worker_id: int,
    ) -> None:
        self.source_mp4 = source_mp4
        self.segment_seconds = segment_seconds
        self.emit_cadence_ms = emit_cadence_ms
        self.playlist_window_segments = max(1, playlist_window_segments)
        self.session_ttl_seconds = session_ttl_seconds
        self.model_name = model_name
        self.namespace = namespace
        self.component = component
        self.worker_id = worker_id
        self._packaged_stream: PackagedStream | None = None
        self._sessions: dict[str, StreamSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._workspace: tempfile.TemporaryDirectory[str] | None = None

    async def initialize(self) -> None:
        self._packaged_stream = await asyncio.to_thread(self._package_source_into_cmaf)
        logger.info(
            "Prepared CMAF stream from %s with %d segments, target_duration=%ds",
            self.source_mp4,
            len(self.packaged_stream.segments),
            self.packaged_stream.target_duration_seconds,
        )

    @property
    def packaged_stream(self) -> PackagedStream:
        if self._packaged_stream is None:
            raise RuntimeError("Synthetic HLS backend is not initialized")
        return self._packaged_stream

    def _package_source_into_cmaf(self) -> PackagedStream:
        if not self.source_mp4.exists():
            raise FileNotFoundError(f"Source MP4 not found: {self.source_mp4}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to package CMAF fragments")

        self._workspace = tempfile.TemporaryDirectory(prefix="dynamo-hls-cmaf-")
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
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-r",
            str(PACKAGING_FPS),
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
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
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
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

        logger.info("Packaging %s into CMAF fragments with ffmpeg", self.source_mp4)
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

        return PackagedStream(
            target_duration_seconds=target_duration_seconds,
            init_filename=init_filename,
            init_bytes=init_bytes,
            segments=segments,
        )

    def _require_hls_request(self, request: NvCreateVideoRequest) -> None:
        annotations = request.nvext.annotations if request.nvext else None
        if not annotations or HLS_CMAF_ANNOTATION not in annotations:
            raise ValueError(
                "synthetic HLS worker requires the experimental_hls_cmaf annotation"
            )

    def _encode_stream_id(self, session: StreamSession) -> str:
        payload = {
            "session_id": session.session_id,
            "worker_id": self.worker_id,
            "namespace": self.namespace,
            "component": self.component,
            "created": session.created,
            "expires_at": session.expires_at,
            "target_duration_seconds": self.packaged_stream.target_duration_seconds,
        }
        return _base64url_encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )

    async def _create_session(self) -> StreamSession:
        created = int(time.time())
        now = time.monotonic()
        session = StreamSession(
            session_id=uuid.uuid4().hex,
            created=created,
            started_at_monotonic=now,
            last_access_monotonic=now,
            expires_at=created + self.session_ttl_seconds,
            visible_segments=min(1, len(self.packaged_stream.segments)),
            completed=len(self.packaged_stream.segments) <= 1,
        )
        async with self._sessions_lock:
            self._purge_expired_locked()
            self._sessions[session.session_id] = session
        return session

    def _purge_expired_locked(self) -> None:
        now_epoch = int(time.time())
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now_epoch
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _refresh_session_locked(self, session: StreamSession) -> tuple[int, bool]:
        if session.expires_at <= int(time.time()):
            self._sessions.pop(session.session_id, None)
            raise FileNotFoundError(f"session expired: {session.session_id}")

        elapsed_ms = max(
            0.0, (time.monotonic() - session.started_at_monotonic) * 1000.0
        )
        total_segments = len(self.packaged_stream.segments)
        visible_segments = min(
            total_segments,
            int(elapsed_ms // self.emit_cadence_ms) + 1,
        )
        completed = visible_segments >= total_segments
        session.visible_segments = visible_segments
        session.completed = completed
        session.last_access_monotonic = time.monotonic()
        return visible_segments, completed

    async def _playlist_state(self, session_id: str) -> tuple[int, bool]:
        async with self._sessions_lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise FileNotFoundError(f"unknown session: {session_id}")
            return self._refresh_session_locked(session)

    def _render_playlist(self, visible_segments: int, completed: bool) -> str:
        window_start = max(0, visible_segments - self.playlist_window_segments)
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            f"#EXT-X-TARGETDURATION:{self.packaged_stream.target_duration_seconds}",
            f"#EXT-X-MEDIA-SEQUENCE:{window_start}",
            "#EXT-X-INDEPENDENT-SEGMENTS",
            '#EXT-X-MAP:URI="init.mp4"',
        ]

        for segment in self.packaged_stream.segments[window_start:visible_segments]:
            lines.append(f"#EXTINF:{segment.duration_seconds:.3f},")
            lines.append(f"segment/{segment.segment_id}.m4s")

        if completed:
            lines.append("#EXT-X-ENDLIST")

        return "\n".join(lines) + "\n"

    @dynamo_endpoint(NvCreateVideoRequest, NvVideosResponse)
    async def generate(self, request: NvCreateVideoRequest):
        self._require_hls_request(request)
        session = await self._create_session()
        stream_id = self._encode_stream_id(session)

        logger.info(
            "Created synthetic HLS session %s for model=%s prompt=%r",
            session.session_id,
            request.model,
            request.prompt,
        )

        yield NvVideosResponse(
            id=stream_id,
            object="video",
            model=request.model,
            status="completed",
            progress=100,
            created=session.created,
            data=[],
            error=None,
            inference_time_s=None,
        ).model_dump()

        logger.info(
            "Synthetic HLS kickoff completed for session %s", session.session_id
        )

    @dynamo_endpoint(HlsPlaylistRequest, HlsPlaylistResponse)
    async def playlist(self, request: HlsPlaylistRequest):
        visible_segments, completed = await self._playlist_state(request.session_id)
        yield {
            "playlist": self._render_playlist(visible_segments, completed),
        }

    @dynamo_endpoint(HlsInitRequest, HlsFragmentResponse)
    async def init_fragment(self, request: HlsInitRequest):
        await self._playlist_state(request.session_id)
        # Python endpoint responses cross the worker/runtime boundary as JSON values.
        # Emit fragment payloads as a u8 array so the Rust frontend can deserialize
        # them back into Vec<u8> without falling back to base64.
        yield {
            "bytes": list(self.packaged_stream.init_bytes),
        }

    @dynamo_endpoint(HlsSegmentRequest, HlsFragmentResponse)
    async def segment(self, request: HlsSegmentRequest):
        visible_segments, _ = await self._playlist_state(request.session_id)
        if request.segment_id < 0:
            raise FileNotFoundError(f"segment not found: {request.segment_id}")
        if request.segment_id >= len(self.packaged_stream.segments):
            raise FileNotFoundError(f"segment not found: {request.segment_id}")
        if request.segment_id >= visible_segments:
            raise FileNotFoundError(f"segment not available yet: {request.segment_id}")

        yield {
            "bytes": list(self.packaged_stream.segments[request.segment_id].bytes),
        }


async def _register_model(endpoint, model_name: str) -> None:
    await register_llm(
        ModelInput.Text,  # type: ignore[attr-defined]
        ModelType.Videos,
        endpoint,
        model_name,
        model_name,
    )
    logger.info("Registered synthetic HLS/CMAF model: %s", model_name)


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace = _get_worker_namespace()
    generate_endpoint = runtime.endpoint(
        f"{namespace}.{DEFAULT_COMPONENT}.{DEFAULT_GENERATE_ENDPOINT}"
    )
    playlist_endpoint = runtime.endpoint(
        f"{namespace}.{DEFAULT_COMPONENT}.{DEFAULT_PLAYLIST_ENDPOINT}"
    )
    init_endpoint = runtime.endpoint(
        f"{namespace}.{DEFAULT_COMPONENT}.{DEFAULT_INIT_ENDPOINT}"
    )
    segment_endpoint = runtime.endpoint(
        f"{namespace}.{DEFAULT_COMPONENT}.{DEFAULT_SEGMENT_ENDPOINT}"
    )

    backend = SyntheticHlsVideoBackend(
        source_mp4=args.source_mp4,
        segment_seconds=args.segment_seconds,
        emit_cadence_ms=args.emit_cadence_ms,
        playlist_window_segments=args.playlist_window_segments,
        session_ttl_seconds=args.session_ttl_seconds,
        model_name=args.model,
        namespace=namespace,
        component=DEFAULT_COMPONENT,
        worker_id=generate_endpoint.connection_id(),
    )
    await backend.initialize()

    logger.info(
        "Serving synthetic HLS endpoints under %s/%s",
        namespace,
        DEFAULT_COMPONENT,
    )
    await asyncio.gather(
        generate_endpoint.serve_endpoint(backend.generate),
        playlist_endpoint.serve_endpoint(backend.playlist),
        init_endpoint.serve_endpoint(backend.init_fragment),
        segment_endpoint.serve_endpoint(backend.segment),
        _register_model(generate_endpoint, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic HLS/CMAF harness for /v1/videos/stream/hls"
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
        help=f"HLS segment duration in seconds (default: {DEFAULT_SEGMENT_SECONDS})",
    )
    parser.add_argument(
        "--emit-cadence-ms",
        type=int,
        default=DEFAULT_EMIT_CADENCE_MS,
        help=f"Delay between newly visible segments in milliseconds (default: {DEFAULT_EMIT_CADENCE_MS})",
    )
    parser.add_argument(
        "--playlist-window-segments",
        type=int,
        default=DEFAULT_PLAYLIST_WINDOW_SEGMENTS,
        help=f"Sliding live playlist window size (default: {DEFAULT_PLAYLIST_WINDOW_SEGMENTS})",
    )
    parser.add_argument(
        "--session-ttl-seconds",
        type=int,
        default=DEFAULT_SESSION_TTL_SECONDS,
        help=f"Time-to-live for synthetic stream sessions in seconds (default: {DEFAULT_SESSION_TTL_SECONDS})",
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
    if args.playlist_window_segments <= 0:
        parser.error("--playlist-window-segments must be positive")
    if args.session_ttl_seconds <= 0:
        parser.error("--session-ttl-seconds must be positive")

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
