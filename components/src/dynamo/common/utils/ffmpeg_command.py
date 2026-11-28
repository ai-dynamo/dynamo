# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder selection and ffmpeg command construction for generated video.

Generated video is encoded by one of two royalty-free encoders:

* ``libvpx-vp9`` in software -- the default, and the only video encoder the
  in-tree LGPL ffmpeg carries.
* ``av1_vaapi`` in hardware, on an Intel GPU over VA-API -- used only when the
  operator points ``DYN_XPU_FFMPEG_PATH`` at an ffmpeg built with VA-API.

No H.264 or H.265 in any form: those are the codecs the shipped images
deliberately exclude. Intel media engines can encode them, but AV1 gives
hardware encoding without putting a royalty-bearing codec back into an image.
VP9 has no hardware path: the Arc Pro B60 this was validated on exposes VP9
for decode only.

Selection is not auto-detected. ``ffmpeg -encoders`` lists VA-API wrappers the
driver cannot run -- ``vp9_vaapi`` lists cleanly and then fails with "No usable
encoding entrypoint found" -- so probing gives false positives. The operator
declares hardware support by setting the path. There is no fallback: a path
that is set but unusable, or a hardware encode that fails, is an error.

Two callers build their argv here, so codec identity, the VA-API upload and
the quality setting are decided in one place: the batch hardware encode in
:func:`dynamo.common.utils.video_utils.encode_to_video_bytes`, and the CMAF
streaming session in :mod:`dynamo.common.utils.cmaf_encoder`, for both
encoders. The batch software encode goes through imageio and does not use this
module.
"""

import os
from typing import List, Optional

import numpy as np

# An ffmpeg built with VA-API support and the av1_vaapi encoder. Its presence is
# the only encoder switch: unset means software VP9.
ENV_XPU_FFMPEG_PATH = "DYN_XPU_FFMPEG_PATH"

# DRM render node of the GPU that encodes. Read only when hardware is selected.
ENV_XPU_VIDEO_DEVICE = "DYN_XPU_VIDEO_DEVICE"
DEFAULT_XPU_VIDEO_DEVICE = "/dev/dri/renderD128"

SW_VIDEO_ENCODER = "libvpx-vp9"
HW_VIDEO_ENCODER = "av1_vaapi"

# VA-API quality level, set explicitly so the output does not depend on the
# driver's built-in default (the iHD driver logs "No quality level set; using
# default (25)"). 25 matches that default.
HW_VIDEO_GLOBAL_QUALITY = 25

# Low-latency arguments for the software encoder on the streaming path:
# libvpx's defaults encode 832x480 slower than real time and look ahead 25
# frames, either of which defeats streaming. `yuv420p` is pinned for
# playability, not quality -- from rgb24 input libvpx picks 4:4:4, which is VP9
# profile 1, and browser MSE support for `vp09.01` is not dependable.
SW_VIDEO_STREAMING_ARGS = (
    "-deadline",
    "realtime",
    "-cpu-used",
    "8",
    "-lag-in-frames",
    "0",
    "-auto-alt-ref",
    "0",
    "-row-mt",
    "1",
    "-pix_fmt",
    "yuv420p",
)

# The VA-API encoder queues frames before returning packets; a depth of one
# gives up a little throughput so a finished fragment is not held back.
HW_VIDEO_STREAMING_ARGS = ("-async_depth", "1")

# Fragmented-MP4 muxer flags: `empty_moov` makes the leading ftyp+moov a
# self-contained init segment, `frag_keyframe` cuts a fragment at each keyframe,
# `default_base_moof` sets the tfhd flag CMAF requires. No `+faststart` -- it
# rewrites the file afterwards and needs a seekable output.
FRAGMENTED_MOVFLAGS = "+frag_keyframe+empty_moov+default_base_moof"


def hw_ffmpeg_path() -> Optional[str]:
    """Resolve the hardware-encode ffmpeg, or ``None`` to encode in software.

    Returns the path from ``DYN_XPU_FFMPEG_PATH``, or ``None`` when the variable
    is unset or blank.

    Raises:
        RuntimeError: If the variable is set but does not name an executable
            file. The operator asked for hardware encoding and should hear that
            they did not get it, so this does not fall back to software.
    """
    path = (os.environ.get(ENV_XPU_FFMPEG_PATH) or "").strip()
    if not path:
        return None
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"{ENV_XPU_FFMPEG_PATH}={path!r} is not an executable file. Point it "
            f"at an ffmpeg built with VA-API support and the {HW_VIDEO_ENCODER} "
            f"encoder, or unset it to encode with software {SW_VIDEO_ENCODER}."
        )
    return path


def xpu_video_device() -> str:
    """DRM render node for VA-API encoding (env: ``DYN_XPU_VIDEO_DEVICE``)."""
    return (
        os.environ.get(ENV_XPU_VIDEO_DEVICE) or ""
    ).strip() or DEFAULT_XPU_VIDEO_DEVICE


def validate_raw_rgb_frames(frames, *, what: str) -> None:
    """Validate frames that are written to ffmpeg as raw ``rgb24`` bytes.

    The bytes carry no shape or type, so anything else would encode as garbage
    rather than fail.

    Args:
        frames: The frames to check.
        what: Name of the consumer, for the error message.

    Raises:
        ValueError: Unless ``frames`` is an ``np.ndarray`` of shape
            ``(T, H, W, 3)`` with dtype ``uint8``.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError(
            f"{what} expects frames as np.ndarray (T, H, W, 3) uint8; "
            f"got {type(frames).__name__}"
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{what} expects shape (T, H, W, 3); got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"{what} expects dtype uint8; got {frames.dtype}")


def build_ffmpeg_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: int,
    output: str,
    hw_device: Optional[str] = None,
    gop: Optional[int] = None,
    streaming: bool = False,
) -> List[str]:
    """Build the ffmpeg argv for one raw-RGB-in, mp4-out encode.

    Args:
        ffmpeg: Binary to invoke.
        width: Frame width, for the rawvideo input.
        height: Frame height, for the rawvideo input.
        fps: Input frame rate.
        output: Output target -- a file path, or ``pipe:1`` when streaming.
        hw_device: DRM render node. Its presence selects the hardware encoder.
        gop: Keyframe interval in frames; ``None`` leaves the encoder default.
        streaming: Emit fragmented MP4 with low-latency encoder settings, for a
            non-seekable output that is read while encoding.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if hw_device:
        cmd += ["-vaapi_device", hw_device]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if hw_device:
        # hwupload moves frames onto the VA-API surface; nv12 is 4:2:0, which is
        # what the encoder consumes and what players expect.
        cmd += ["-vf", "format=nv12,hwupload"]
    cmd += ["-c:v", HW_VIDEO_ENCODER if hw_device else SW_VIDEO_ENCODER]
    if hw_device:
        cmd += ["-global_quality", str(HW_VIDEO_GLOBAL_QUALITY)]
    if gop:
        # Every fragment must start with a keyframe, so the GOP is the segment
        # length rather than an encoder heuristic.
        cmd += ["-g", str(gop), "-keyint_min", str(gop)]
    if streaming:
        cmd += list(HW_VIDEO_STREAMING_ARGS if hw_device else SW_VIDEO_STREAMING_ARGS)
        cmd += ["-movflags", FRAGMENTED_MOVFLAGS, "-flush_packets", "1"]
    else:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-f", "mp4", output]
    return cmd
