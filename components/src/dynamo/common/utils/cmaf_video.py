# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CMAF video streaming protocol: tags, frame metadata, and chunking.

This module owns the wire vocabulary shared by the Python worker and the Rust
frontend route, and nothing else -- it deliberately has no encoder dependency so
that ``video_utils`` can import it.

A CMAF response stream is a sequence of tagged frames:

    cmaf:metadata     once, first: JSON describing the stream
    cmaf:init         once: the fragmented-MP4 init segment (ftyp + moov)
    cmaf:segment:{n}  per chunk: one media fragment (moof + mdat)

The frontend turns each tag into a one-byte kind and emits
``[kind][len][payload]``; ``error`` and ``end`` frames are produced by the route
itself, not by the worker.
"""

import json
import os
from typing import Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Wire tags. Must stay in sync with the frame-kind mapping in the Rust route
# (lib/llm/src/http/service/openai.rs).
# ---------------------------------------------------------------------------

CMAF_METADATA_TAG = "cmaf:metadata"
CMAF_INIT_TAG = "cmaf:init"
CMAF_SEGMENT_TAG_PREFIX = "cmaf:segment:"

# Annotation that selects CMAF output on a video request. The streaming route
# sets it; nothing else in the request switches output shape.
CMAF_ANNOTATION = "cmaf"

# Segment length in frames. Must be a multiple of four: once frames arrive
# incrementally the Wan2.1 causal VAE emits them 1, 4, 4, 4, ..., and a
# multiple-of-four segment keeps the carry into the next segment at a constant
# one frame instead of oscillating. See DEP 0017.
ENV_CMAF_GOP_FRAMES = "DYN_CMAF_GOP_FRAMES"
DEFAULT_CMAF_GOP_FRAMES = 8


def segment_tag(index: int) -> str:
    """Tag for media segment ``index`` (1-based)."""
    return f"{CMAF_SEGMENT_TAG_PREFIX}{index}"


def cmaf_gop_frames() -> int:
    """Frames per CMAF segment (env: ``DYN_CMAF_GOP_FRAMES``).

    Raises:
        ValueError: If the value is not a positive multiple of four.
    """
    raw = (os.environ.get(ENV_CMAF_GOP_FRAMES) or "").strip()
    if not raw:
        return DEFAULT_CMAF_GOP_FRAMES
    try:
        gop = int(raw)
    except ValueError as err:
        raise ValueError(f"{ENV_CMAF_GOP_FRAMES}={raw!r} is not an integer") from err
    if gop <= 0 or gop % 4 != 0:
        raise ValueError(
            f"{ENV_CMAF_GOP_FRAMES}={gop} must be a positive multiple of 4; "
            f"default is {DEFAULT_CMAF_GOP_FRAMES}."
        )
    return gop


def cmaf_segment_seconds(fps: int, gop_frames: Optional[int] = None) -> float:
    """Nominal segment duration in seconds."""
    gop = cmaf_gop_frames() if gop_frames is None else gop_frames
    return gop / float(fps)


def iter_cmaf_chunks(frames: np.ndarray, gop_frames: int) -> Iterator[np.ndarray]:
    """Slice canonical frames into fixed-size segments.

    The final slice carries the remainder, which is routinely shorter than
    ``gop_frames`` -- 97 frames at 8 gives twelve full segments and a 1-frame
    tail -- so callers must not assume a uniform chunk length.
    """
    if gop_frames <= 0:
        raise ValueError(f"gop_frames must be positive; got {gop_frames}")
    total = len(frames)
    for start in range(0, total, gop_frames):
        yield frames[start : min(start + gop_frames, total)]


def metadata_bytes(
    *,
    video_codec: str,
    width: int,
    height: int,
    fps: int,
    target_duration: float,
    segment_count: Optional[int] = None,
) -> bytes:
    """Build the ``cmaf:metadata`` payload.

    ``video_codec`` is the RFC 6381 codec string read from the init segment.
    ``segment_count`` is ``None`` when the total is not known in advance -- the
    client keys end-of-stream off the terminal ``end`` frame either way.
    """
    return json.dumps(
        {
            "mime_type": f'video/mp4; codecs="{video_codec}"',
            "video_codec": video_codec,
            "width": width,
            "height": height,
            "fps": fps,
            "target_duration": target_duration,
            "segment_count": segment_count,
        }
    ).encode("utf-8")
