# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary CMAF packaging for the vLLM-Omni video path.

The Dynamo HTTP frontend exposes ``POST /v1/videos/stream/binary/cmaf``. That
route injects the ``experimental_binary_cmaf`` annotation into the request and
re-frames a stream of :class:`NvVideosResponse` items into a single binary CMAF
response body. Each response item carries a *tag* in ``VideoData.url`` and a
base64 payload in ``VideoData.b64_json``:

    cmaf:metadata   -> JSON describing the asset (protocol, codecs, segment_count)
    cmaf:init       -> the fMP4 initialization segment
    cmaf:segment:N  -> the Nth fMP4 media segment

Wan2.1-T2V (and diffusion video generation generally) produces the *whole* clip
in one call. The pipeline is therefore ``diffuser -> encode_video -> split ->
stream``: the shared unified encoder
(:func:`dynamo.common.utils.video_utils.encode_video`) does the hardware
encoding, asked for a fragmented/CMAF-compatible MP4, and the encoder-agnostic
:func:`split_fragmented_mp4` chops that single stream into an init segment plus
media segments. This module only owns the *protocol* (tags, metadata, request
opt-in); it contains no encoder- or hardware-specific logic.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from dynamo.common.utils.video_utils import (
    encode_video,
    h264_codec_string_from_init,
    split_fragmented_mp4,
)

logger = logging.getLogger(__name__)

# Wire protocol constants -- must match the Rust frontend
# (lib/llm/src/http/service/openai.rs) and the reference worker
# (examples/custom_backend/cmaf_binary_video_streaming/worker.py).
CMAF_ANNOTATION = "experimental_binary_cmaf"
CMAF_PROTOCOL = "dynamo-video-binary-cmaf-v1"
CMAF_METADATA_TAG = "cmaf:metadata"
CMAF_INIT_TAG = "cmaf:init"
CMAF_SEGMENT_PREFIX = "cmaf:segment:"

# Fallback H.264 codec string (Main profile, level 3.1) advertised only when the
# real profile/level cannot be parsed from the encoded init segment.
CMAF_FALLBACK_VIDEO_CODEC = "avc1.4d401f"

_DEFAULT_SEGMENT_SECONDS = 2
_DEFAULT_EMIT_CADENCE_MS = 0


def has_cmaf_annotation(nvext) -> bool:
    """Return True when the request opts in to binary CMAF streaming.

    ``nvext`` is a ``VideoNvExt`` (or None). The frontend's binary-CMAF route
    appends :data:`CMAF_ANNOTATION` to ``nvext.annotations`` before dispatch.
    """
    return bool(
        nvext is not None
        and nvext.annotations
        and CMAF_ANNOTATION in nvext.annotations
    )


def cmaf_segment_seconds() -> int:
    """Target CMAF fragment duration (env: ``DYN_CMAF_SEGMENT_SECONDS``)."""
    raw = os.environ.get("DYN_CMAF_SEGMENT_SECONDS")
    if not raw:
        return _DEFAULT_SEGMENT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid DYN_CMAF_SEGMENT_SECONDS=%r; using %d",
            raw,
            _DEFAULT_SEGMENT_SECONDS,
        )
        return _DEFAULT_SEGMENT_SECONDS
    return max(1, value)


def cmaf_emit_cadence_s() -> float:
    """Optional delay between emitted media segments (env: ``DYN_CMAF_EMIT_CADENCE_MS``).

    Defaults to 0 (emit as fast as the client can consume). A positive value
    paces the segment stream, which can smooth out client-side buffering during
    demos.
    """
    raw = os.environ.get("DYN_CMAF_EMIT_CADENCE_MS")
    if not raw:
        return _DEFAULT_EMIT_CADENCE_MS / 1000.0
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid DYN_CMAF_EMIT_CADENCE_MS=%r; using 0", raw)
        return 0.0
    return max(0, value) / 1000.0


def source_buffer_mime_type(video_codec: str) -> str:
    """MSE ``SourceBuffer`` mime type for the packaged video-only asset."""
    return f'video/mp4; codecs="{video_codec}"'


def metadata_bytes(
    segment_count: int, target_duration_seconds: int, video_codec: str
) -> bytes:
    """Serialize the ``cmaf:metadata`` payload (JSON, UTF-8)."""
    payload = {
        "protocol": CMAF_PROTOCOL,
        "mime_type": "video/mp4",
        "source_buffer_mime_type": source_buffer_mime_type(video_codec),
        "video_codec": video_codec,
        "audio_codec": None,
        "has_audio": False,
        "target_duration_seconds": target_duration_seconds,
        "segment_count": segment_count,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def package_frames_to_cmaf(
    frames: np.ndarray, fps: int, segment_seconds: int
) -> tuple[bytes, list[bytes], int, str]:
    """Encode canonical frames and split them into CMAF init + media segments.

    Thin orchestration over the shared unified encoder: ask
    :func:`encode_video` for a fragmented (CMAF-compatible) H.264/MP4 stream,
    then split it with :func:`split_fragmented_mp4`. No encoder- or
    hardware-specific logic lives here.

    Args:
        frames: Canonical ``np.ndarray (T, H, W, 3)`` uint8 RGB frames.
        fps: Frames per second of the generated clip.
        segment_seconds: Target duration of each CMAF media fragment.

    Returns:
        ``(init_bytes, segment_bytes_list, target_duration_seconds, video_codec)``.
        ``video_codec`` is parsed from the encoded bitstream so the advertised
        codec matches reality.

    Raises:
        RuntimeError: If ffmpeg is missing or encoding/fragmentation fails.
        ValueError: If ``frames`` is not canonical ``(T, H, W, 3)`` uint8.
    """
    mp4_bytes = encode_video(
        frames,
        fps,
        container="mp4",
        codec="h264",
        gop_seconds=segment_seconds,
        fragmented=True,
    )
    init_bytes, segments = split_fragmented_mp4(mp4_bytes)

    video_codec = h264_codec_string_from_init(init_bytes) or CMAF_FALLBACK_VIDEO_CODEC

    # TODO: parse the real per-fragment duration from each moof (tfhd default
    # sample duration / trun sample durations, scaled by the mdhd timescale in
    # the init segment) instead of approximating from the requested segment
    # length. HW encoders may not honor the requested GOP exactly, so the actual
    # fragment durations can differ.
    target_duration = segment_seconds

    logger.info(
        "CMAF packaging produced init (%d bytes) + %d segments, codec=%s",
        len(init_bytes),
        len(segments),
        video_codec,
    )
    return init_bytes, segments, target_duration, video_codec
