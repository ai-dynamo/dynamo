# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import asyncio
import contextlib
import io
import logging
import os
import subprocess
import tempfile
from typing import AsyncIterator, Iterator, List, Optional, Tuple

import numpy as np

from dynamo.common.utils.cmaf_video import (
    CMAF_INIT_TAG,
    DEFAULT_CMAF_GOP_FRAMES,
    segment_tag,
)

logger = logging.getLogger(__name__)


DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_NUM_FRAMES = 97


def parse_size(
    size: str | None,
    default_w: int = DEFAULT_VIDEO_WIDTH,
    default_h: int = DEFAULT_VIDEO_HEIGHT,
) -> Tuple[int, int]:
    """Parse a 'WxH' string into (width, height).

    Falls back to default_w x default_h when size is None or malformed.
    """
    if not size:
        return default_w, default_h
    try:
        w, h = size.split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        logger.warning("Invalid size format: %s, using defaults", size)
        return default_w, default_h


def compute_num_frames(
    num_frames: int | None = None,
    seconds: int | None = None,
    fps: int | None = None,
    default_fps: int = DEFAULT_VIDEO_FPS,
    default_num_frames: int = DEFAULT_VIDEO_NUM_FRAMES,
) -> int:
    """Compute the number of video frames.

    Priority: num_frames > seconds x fps > default_num_frames.
    """
    if num_frames is not None:
        return num_frames
    if seconds is not None or fps is not None:
        _seconds = seconds if seconds is not None else 4
        _fps = fps if fps is not None else default_fps
        return _seconds * _fps
    return default_num_frames


def normalize_video_frames(images: list) -> list:
    """Normalize stage_output.images into a frame list for export_to_video.

    Args:
        images: stage_output.images -- a list that may contain a single
            torch.Tensor or np.ndarray representing the full video.

    Returns:
        List of frames suitable for diffusers export_to_video.
    """
    frames = images[0] if len(images) == 1 else images

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        return list(frames)

    return list(frames)


def frames_to_numpy(images: list) -> np.ndarray:
    """Convert a list of video frames to a numpy array suitable for encoding.

    Accepts either PIL Images or numpy arrays. Diffusion video pipelines (e.g.
    Wan2.1 T2V) emit numpy frames — float in ``[0, 1]`` by default — while other
    stages hand back PIL Images; this normalizes both to ``uint8`` RGB, matching
    the conversion ``diffusers.export_to_video`` performs internally.

    Args:
        images: List of PIL Image objects or ``np.ndarray`` frames (H, W, 3).

    Returns:
        Numpy array of shape ``(num_frames, height, width, 3)`` with dtype
        ``uint8`` and values in ``[0, 255]``.

    Raises:
        ValueError: If no images are provided or images have inconsistent sizes.
    """
    if not images:
        raise ValueError("No images provided for video encoding")

    frames = []
    for img in images:
        if isinstance(img, np.ndarray):
            arr = img
            if arr.dtype != np.uint8:
                # Diffusers convention: numpy frames are float in [0, 1].
                arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        else:
            # PIL Image.
            arr = np.array(img.convert("RGB"))
        frames.append(arr)

    # Validate consistent sizes
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent frame sizes detected: {shapes}. "
            "All frames must have the same dimensions."
        )

    return np.stack(frames, axis=0)


def encode_to_mp4(
    frames: np.ndarray,
    output_dir: str,
    request_id: str,
    fps: int = 16,
) -> str:
    """Encode numpy frames to MP4 file.

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        output_dir: Directory to save the output video.
        request_id: Unique identifier for the request (used in filename).
        fps: Frames per second for the output video.

    Returns:
        Path to the saved MP4 file.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{request_id}.mp4")

    logger.info(f"Encoding {len(frames)} frames to {output_path} at {fps} fps")

    try:
        # Encode with VP9 (libvpx-vp9). The in-tree ffmpeg build is LGPL-only and
        # royalty-free: it carries no H.264 codec (not even the h264_nvenc HW
        # encoder), so VP9 is the video encoder we ship. VP9-in-mp4 is valid and
        # decodes with our VP8/VP9 decoder allowlist; see
        # container/templates/wheel_builder.Dockerfile.
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="libvpx-vp9")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="libvpx-vp9")  # type: ignore[attr-defined]
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        logger.info(f"Video saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to encode video: {e}")
        raise RuntimeError(f"Video encoding failed: {e}") from e


def encode_to_video_bytes(
    frames: np.ndarray,
    fps: int = 16,
    output_format: str = "mp4",
) -> bytes:
    """Encode numpy frames to video bytes (in-memory).

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        fps: Frames per second for the output video.
        output_format: Container format — "mp4", "webm".

    Returns:
        Encoded video as bytes.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    logger.info(f"Encoding {len(frames)} frames to {output_format} bytes at {fps} fps")

    try:
        buffer = io.BytesIO()

        # VP9 (libvpx-vp9) for both containers: the in-tree ffmpeg is royalty-free
        # and carries no H.264 encoder. VP9-in-mp4 and VP9-in-webm are both valid.
        kwargs: dict = {"fps": fps}
        if output_format in ("webm", "mp4"):
            kwargs["codec"] = "libvpx-vp9"
        else:
            raise ValueError(f"No codec specified for response format: {output_format}")

        if hasattr(iio, "imwrite"):
            # v3 API
            iio.imwrite(buffer, frames, extension=f".{output_format}", **kwargs)
        else:
            # v2 API
            writer = iio.get_writer(  # type: ignore[attr-defined]
                buffer, format="FFMPEG", mode="I", **kwargs
            )
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        video_bytes = buffer.getvalue()
        logger.info(f"Encoded video to {len(video_bytes)} bytes")
        return video_bytes

    except Exception as e:
        logger.error(f"Failed to encode video to bytes: {e}")
        raise RuntimeError(f"Video encoding to bytes failed: {e}") from e


# ---------------------------------------------------------------------------
# Unified video encoding
#
# A single shared entry point (``encode_video``) that all backends call with a
# canonical frame array -- ``np.ndarray (T, H, W, 3) uint8`` RGB. Each backend
# owns a ``to_canonical()`` converter (next to its handler) that maps its native
# output into this format, composed from the canonical-domain primitives below
# (``ensure_uint8_rgb`` / ``pil_frames_to_array`` / ``drop_alpha``).
#
# There are exactly two encoders, and both are royalty-free:
#
#   * ``libvpx-vp9`` in software, via imageio -- the default, and the only video
#     encoder the in-tree LGPL ffmpeg carries. Matches what the runtime images
#     already ship.
#   * ``av1_vaapi`` in hardware, via the ffmpeg CLI -- used only when the
#     operator points ``DYN_XPU_FFMPEG_PATH`` at an ffmpeg built with VA-API.
#
# No H.264 or H.265 in any form. Those are the codecs the shipped images
# deliberately exclude (see tests/dependencies/test_no_software_video_codecs.py).
# Intel media engines can encode them, but doing so would put a royalty-bearing
# codec surface back into a distributed image; AV1 gives us hardware encode
# without that problem. VP9 is decode-only on current Intel silicon, so it has
# no hardware path.
#
# Selection is deliberately not auto-detected. ``ffmpeg -encoders`` advertises
# wrappers the driver cannot actually run -- ``vp9_vaapi`` lists cleanly and then
# fails at runtime with "No usable encoding entrypoint found" -- so capability
# probing yields false positives. The operator declares hardware support by
# setting the path, and we take them at their word.
# ---------------------------------------------------------------------------

# Path to an ffmpeg built with VA-API and the AV1 encoder. Its presence is the
# only switch that selects hardware encoding.
ENV_XPU_FFMPEG_PATH = "DYN_XPU_FFMPEG_PATH"

# DRM render node used for VA-API hardware encoding.
ENV_XPU_VIDEO_DEVICE = "DYN_XPU_VIDEO_DEVICE"
DEFAULT_XPU_VIDEO_DEVICE = "/dev/dri/renderD128"

# Software ffmpeg, needed only by the CMAF path: imageio returns nothing until
# encoding completes, so it cannot hand back fragments. The default is the LGPL
# build the runtime images ship, the only location the codec policy permits.
ENV_FFMPEG_PATH = "DYN_FFMPEG_PATH"
DEFAULT_FFMPEG_PATH = "/usr/local/bin/ffmpeg"

# The only container we emit. Every backend already rejects anything else during
# request validation, so this is not a narrowing of behaviour.
VIDEO_CONTAINER = "mp4"

# The two encoders.
SW_VIDEO_ENCODER = "libvpx-vp9"
HW_VIDEO_ENCODER = "av1_vaapi"

# VA-API quality level, set explicitly so output does not depend on the driver's
# built-in default (the iHD driver logs "No quality level set; using default
# (25)"). 25 matches that default, so this pins current behaviour rather than
# changing it.
HW_VIDEO_GLOBAL_QUALITY = 25

# Low-latency arguments for the software encoder on the streaming path: libvpx's
# defaults encode 832x480 slower than real time and look ahead 25 frames, either
# of which defeats streaming. `yuv420p` is pinned for playability, not quality --
# from rgb24 input libvpx picks 4:4:4, which is VP9 profile 1, and browser MSE
# support for `vp09.01` is not dependable. The batch encoder is untouched.
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
# trades a little throughput for not holding a finished fragment back.
HW_VIDEO_STREAMING_ARGS = ("-async_depth", "1")

# Fragmented-MP4 muxer flags: `empty_moov` makes the leading ftyp+moov a
# self-contained init segment, `frag_keyframe` cuts a fragment at each keyframe,
# `default_base_moof` sets the tfhd flag CMAF requires. No `+faststart` -- it
# rewrites the file afterwards and needs a seekable output.
FRAGMENTED_MOVFLAGS = "+frag_keyframe+empty_moov+default_base_moof"


def xpu_video_device() -> str:
    """DRM render node for VA-API encoding (env: ``DYN_XPU_VIDEO_DEVICE``)."""
    return (
        os.environ.get(ENV_XPU_VIDEO_DEVICE) or ""
    ).strip() or DEFAULT_XPU_VIDEO_DEVICE


def hw_ffmpeg_path() -> str | None:
    """Resolve the hardware-encode ffmpeg, or ``None`` to encode in software.

    Returns the path from ``DYN_XPU_FFMPEG_PATH`` when it is set and usable, or
    ``None`` when the variable is unset -- meaning software VP9.

    Raises:
        RuntimeError: If the variable is set but does not name an executable
            file. A misconfigured path is a deployment error, not a reason to
            silently fall back to software: the operator asked for hardware
            encoding and should hear that they did not get it.
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


def sw_ffmpeg_path() -> str:
    """Path to the software ffmpeg binary (env: ``DYN_FFMPEG_PATH``)."""
    return (os.environ.get(ENV_FFMPEG_PATH) or "").strip() or DEFAULT_FFMPEG_PATH


def resolve_streaming_ffmpeg() -> Tuple[str, Optional[str]]:
    """Resolve the ffmpeg binary and VA-API device for the CMAF path.

    Returns ``(ffmpeg, device)``, where ``device`` is the DRM render node for
    hardware AV1 or ``None`` for software VP9. Selection is ``encode_video``'s --
    ``DYN_XPU_FFMPEG_PATH`` is the only switch -- but this path drives the
    software encoder through the CLI too, so it needs a binary, not imageio.

    Raises:
        RuntimeError: If the selected binary is not executable. There is no
            fallback to the other encoder.
    """
    hw = hw_ffmpeg_path()
    if hw is not None:
        return hw, xpu_video_device()

    sw = sw_ffmpeg_path()
    if not os.path.isfile(sw) or not os.access(sw, os.X_OK):
        raise RuntimeError(
            f"CMAF streaming needs an ffmpeg binary, but {sw!r} is not an "
            f"executable file. Point {ENV_FFMPEG_PATH} at an ffmpeg carrying "
            f"{SW_VIDEO_ENCODER}, or set {ENV_XPU_FFMPEG_PATH} to encode "
            f"{HW_VIDEO_ENCODER} in hardware."
        )
    return sw, None


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

    Shared by the batch hardware encoder and the CMAF streaming session, so
    codec identity, VA-API upload, and quality live in exactly one place.

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
    encoder = HW_VIDEO_ENCODER if hw_device else SW_VIDEO_ENCODER

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
    cmd += ["-c:v", encoder]
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
    cmd += ["-f", VIDEO_CONTAINER, output]
    return cmd


def validate_video_encoder_config() -> None:
    """Resolve and log the video encoder configuration.

    Safe to call at worker startup so a misconfigured ``DYN_XPU_FFMPEG_PATH`` is
    reported before the first request rather than on it. ``encode_video`` performs
    the same resolution, so calling this is optional.

    Raises:
        RuntimeError: If ``DYN_XPU_FFMPEG_PATH`` is set but unusable.
    """
    path = hw_ffmpeg_path()
    if path is None:
        logger.info(
            "Video encoding: software %s (%s not set)",
            SW_VIDEO_ENCODER,
            ENV_XPU_FFMPEG_PATH,
        )
    else:
        logger.info(
            "Video encoding: hardware %s via %s on %s",
            HW_VIDEO_ENCODER,
            path,
            xpu_video_device(),
        )


def drop_alpha(frames: np.ndarray) -> np.ndarray:
    """Drop a trailing alpha channel (RGBA -> RGB) when present."""
    if frames.shape[-1] == 4:
        return frames[..., :3]
    return frames


def ensure_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """Normalize an RGB frame array to contiguous ``(T, H, W, 3) uint8``.

    Drops a trailing alpha channel and scales floating-point values in
    ``[0, 1]`` up to ``[0, 255]``. Channel order and axis layout are assumed to
    be RGB / ``(T, H, W, C)`` already; this operates purely in the canonical
    domain and carries no backend-specific knowledge.
    """
    frames = drop_alpha(frames)
    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames * 255.0, 0, 255).round()
    return np.ascontiguousarray(frames, dtype=np.uint8)


def pil_frames_to_array(frames) -> np.ndarray:
    """Stack a list of per-frame images into a single ``(T, H, W, C)`` array.

    Each element may be a ``PIL.Image`` or an ``np.ndarray``; PIL images are
    converted to RGB numpy arrays first.
    """
    per_frame = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            per_frame.append(frame)
        else:
            per_frame.append(np.array(frame.convert("RGB")))
    return np.stack(per_frame, axis=0)


def _validate_canonical_frames(frames) -> None:
    """Validate the canonical encoder input contract.

    Raises ``ValueError`` unless ``frames`` is an ``np.ndarray`` of shape
    ``(T, H, W, 3)`` with dtype ``uint8``.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError(
            "encode_video expects canonical frames as np.ndarray (T, H, W, 3) "
            f"uint8; got {type(frames).__name__}. Convert backend output with "
            "the backend's to_canonical() first."
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"encode_video expects shape (T, H, W, 3); got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"encode_video expects dtype uint8; got {frames.dtype}")


def _encode_vp9_imageio(frames: np.ndarray, fps: int) -> bytes:
    """Encode canonical frames to VP9-in-mp4 with the software encoder.

    Goes through imageio, which resolves the ffmpeg shipped with the runtime
    image -- the royalty-free LGPL build carrying ``libvpx-vp9`` and no H.264.
    This is the default path, and the only one exercised when
    ``DYN_XPU_FFMPEG_PATH`` is unset.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError as err:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            ) from err

    buffer = io.BytesIO()
    if hasattr(iio, "imwrite"):
        iio.imwrite(
            buffer,
            frames,
            extension=f".{VIDEO_CONTAINER}",
            fps=fps,
            codec=SW_VIDEO_ENCODER,
        )
    else:
        writer = iio.get_writer(  # type: ignore[attr-defined]
            buffer, format="FFMPEG", mode="I", fps=fps, codec=SW_VIDEO_ENCODER
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
    return buffer.getvalue()


def _encode_av1_vaapi(frames: np.ndarray, fps: int, ffmpeg: str, device: str) -> bytes:
    """Encode canonical frames to AV1-in-mp4 on the VA-API hardware encoder.

    Pipes raw RGB to the operator-supplied ffmpeg. mp4 needs a seekable output
    for the moov atom, so we encode to a temp file and read the bytes back.

    Raises:
        RuntimeError: If ffmpeg exits non-zero. Its stderr is included, and no
            software fallback is attempted -- see ``hw_ffmpeg_path``.
    """
    num_frames, height, width, _ = frames.shape

    with tempfile.NamedTemporaryFile(suffix=f".{VIDEO_CONTAINER}", delete=False) as tmp:
        output_path = tmp.name
    try:
        cmd = build_ffmpeg_command(
            ffmpeg,
            width=width,
            height=height,
            fps=fps,
            output=output_path,
            hw_device=device,
        )

        logger.info(
            "Encoding %d frames (%dx%d @ %d fps) via %s on %s",
            num_frames,
            width,
            height,
            fps,
            HW_VIDEO_ENCODER,
            device,
        )
        proc = subprocess.run(
            cmd,
            input=frames.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg {HW_VIDEO_ENCODER} encode failed "
                f"(exit {proc.returncode}): {stderr}"
            )
        with open(output_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def encode_video(frames: np.ndarray, fps: int = DEFAULT_VIDEO_FPS) -> bytes:
    """Unified video encoder: encode canonical frames to mp4 bytes.

    ``frames`` must already be canonical -- an ``np.ndarray`` of shape
    ``(T, H, W, 3)``, dtype ``uint8``, RGB. Each backend converts its native
    output with its own ``to_canonical()`` before calling this.

    The codec is not a caller choice. Hardware AV1 is used when
    ``DYN_XPU_FFMPEG_PATH`` names a VA-API-capable ffmpeg; otherwise frames are
    encoded with software VP9. Both are royalty-free, and the container is
    always mp4.

    Args:
        frames: Canonical ``np.ndarray (T, H, W, 3) uint8`` RGB frames.
        fps: Frames per second for the output video.

    Returns:
        Encoded mp4 bytes.

    Raises:
        ValueError: If ``frames`` is not the canonical ``(T, H, W, 3) uint8`` array.
        RuntimeError: If ``DYN_XPU_FFMPEG_PATH`` is set but unusable, or if a
            hardware encode fails. Neither case falls back to software.
    """
    _validate_canonical_frames(frames)

    ffmpeg = hw_ffmpeg_path()
    if ffmpeg is None:
        logger.info(
            "Encoding %d frames -> %s (%s) at %d fps",
            len(frames),
            VIDEO_CONTAINER,
            SW_VIDEO_ENCODER,
            fps,
        )
        return _encode_vp9_imageio(frames, fps)

    return _encode_av1_vaapi(frames, fps, ffmpeg, xpu_video_device())


# ---------------------------------------------------------------------------
# CMAF streaming
#
# One long-lived ffmpeg per request, raw RGB in on stdin, fragmented MP4 out on
# stdout. One muxer instance sees every frame, so mfhd.sequence_number and
# tfdt.baseMediaDecodeTime are continuous by construction and our only bitstream
# work is reading box headers to find segment boundaries. See DEP 0017.
# ---------------------------------------------------------------------------

MP4_BOX_HEADER_SIZE = 8

# Bytes of VisualSampleEntry body that precede the codec configuration box.
_VISUAL_SAMPLE_ENTRY_BODY = 78

_CODEC_CONFIG_BOXES = (b"av1C", b"vpcC")

# Used when the init segment cannot be parsed, so the client still gets a codec
# string it can hand to MediaSource.isTypeSupported().
CMAF_FALLBACK_VIDEO_CODEC = {
    HW_VIDEO_ENCODER: "av01.0.05M.08",
    SW_VIDEO_ENCODER: "vp09.00.10.08",
}

# How long to wait for one read from a live ffmpeg before giving up on it.
CMAF_READ_TIMEOUT_S = 120.0


def _box_header(buf, pos: int, limit: int) -> Optional[Tuple[int, int, bytes]]:
    """Parse the box header at ``pos``.

    Returns ``(header_size, box_size, box_type)``, or ``None`` when fewer than
    the header's bytes are available yet.

    Raises:
        ValueError: If the box declares a size smaller than its own header.
    """
    if limit - pos < MP4_BOX_HEADER_SIZE:
        return None
    size = int.from_bytes(buf[pos : pos + 4], "big")
    btype = bytes(buf[pos + 4 : pos + 8])
    header = MP4_BOX_HEADER_SIZE
    if size == 1:
        if limit - pos < 16:
            return None
        size = int.from_bytes(buf[pos + 8 : pos + 16], "big")
        header = 16
    if size == 0:
        # "Extends to end of stream" -- legal only for the last top-level box,
        # and never completable from a growing buffer.
        return None
    if size < header:
        raise ValueError(f"invalid MP4 box size {size} for {btype!r} at offset {pos}")
    return header, size, btype


def _iter_child_boxes(
    buf: bytes, start: int, end: int
) -> Iterator[Tuple[bytes, int, int]]:
    """Yield ``(box_type, payload_start, box_end)`` for boxes in ``[start, end)``."""
    pos = start
    while pos < end:
        header = _box_header(buf, pos, end)
        if header is None:
            return
        header_size, box_size, btype = header
        box_end = min(pos + box_size, end)
        yield btype, pos + header_size, box_end
        pos += box_size


def _find_child(
    buf: bytes, start: int, end: int, btype: bytes
) -> Optional[Tuple[int, int]]:
    """Return ``(payload_start, box_end)`` of the first ``btype`` box, or ``None``."""
    for found, payload_start, box_end in _iter_child_boxes(buf, start, end):
        if found == btype:
            return payload_start, box_end
    return None


def _av1_codec_string(cfg: bytes) -> Optional[str]:
    """RFC 6381 ``av01.P.LLT.DD`` from an ``av1C`` payload."""
    if len(cfg) < 3:
        return None
    profile = (cfg[1] >> 5) & 0x07
    level = cfg[1] & 0x1F
    tier = "H" if (cfg[2] >> 7) & 0x01 else "M"
    high_bitdepth = (cfg[2] >> 6) & 0x01
    twelve_bit = (cfg[2] >> 5) & 0x01
    depth = 12 if twelve_bit else (10 if high_bitdepth else 8)
    return f"av01.{profile}.{level:02d}{tier}.{depth:02d}"


def _vp9_codec_string(cfg: bytes) -> Optional[str]:
    """RFC 6381 ``vp09.PP.LL.DD`` from a ``vpcC`` payload (a FullBox)."""
    if len(cfg) < 7:
        return None
    profile, level = cfg[4], cfg[5]
    depth = cfg[6] >> 4
    return f"vp09.{profile:02d}.{level:02d}.{depth:02d}"


def codec_string_from_init(init: bytes) -> Optional[str]:
    """Read the RFC 6381 codec string out of a CMAF init segment.

    It cannot be a constant: profile and level depend on the encode the
    deployment actually ran, and the client passes the string to
    ``MediaSource.isTypeSupported()`` before appending anything.

    Returns ``None`` if the sample entry cannot be located.
    """
    try:
        scope = _find_child(init, 0, len(init), b"moov")
        for name in (b"trak", b"mdia", b"minf", b"stbl", b"stsd"):
            if scope is None:
                return None
            scope = _find_child(init, scope[0], scope[1], name)
        if scope is None:
            return None

        # stsd is a FullBox: 4 bytes of version/flags, then a 4-byte entry count.
        entries_start = scope[0] + 8
        entry = _box_header(init, entries_start, scope[1])
        if entry is None:
            return None
        header_size, box_size, _ = entry
        body = entries_start + header_size + _VISUAL_SAMPLE_ENTRY_BODY
        entry_end = entries_start + box_size

        for btype, payload_start, box_end in _iter_child_boxes(init, body, entry_end):
            if btype not in _CODEC_CONFIG_BOXES:
                continue
            cfg = init[payload_start:box_end]
            if btype == b"av1C":
                return _av1_codec_string(cfg)
            return _vp9_codec_string(cfg)
    except (ValueError, IndexError) as err:
        logger.warning("Could not parse CMAF init segment: %s", err)
    return None


class FragmentedMp4Cutter:
    """Cut a growing fragmented-MP4 byte stream into CMAF segments.

    Feed bytes as they arrive; each call returns the segments that just became
    complete, as ``(kind, payload)`` with ``kind`` in ``{"init", "segment"}``.
    The init segment is ``ftyp`` through the end of ``moov``; each later segment
    is one ``moof`` plus its ``mdat``. A boundary landing mid-read costs only a
    wait for the rest of the box.

    Byte ranges are emitted verbatim -- no box is modified, because the muxer
    owns the timeline.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pos = 0  # offset of the next unparsed box header
        self._init_done = False
        self._in_fragment = False

    def feed(self, data: bytes) -> List[Tuple[str, bytes]]:
        """Add bytes and return whatever segments they completed."""
        self._buf += data
        return self._cut()

    def flush(self) -> List[Tuple[str, bytes]]:
        """Emit what remains at end of stream.

        A trailing ``mdat`` whose size was declared as "to end of stream" can
        only be closed here.
        """
        out = self._cut()
        if self._in_fragment and self._buf:
            out.append(("segment", bytes(self._buf)))
            self._reset_fragment()
        elif self._buf:
            logger.debug(
                "Discarding %d trailing byte(s) after the last fragment", len(self._buf)
            )
            self._reset_fragment()
        return out

    def _reset_fragment(self) -> None:
        self._buf.clear()
        self._pos = 0
        self._in_fragment = False

    def _take(self, end: int, kind: str) -> Tuple[str, bytes]:
        payload = bytes(self._buf[:end])
        del self._buf[:end]
        self._pos = 0
        return kind, payload

    def _cut(self) -> List[Tuple[str, bytes]]:
        out: List[Tuple[str, bytes]] = []
        while True:
            header = _box_header(self._buf, self._pos, len(self._buf))
            if header is None:
                break
            _, box_size, btype = header
            end = self._pos + box_size
            if len(self._buf) < end:
                break  # box still arriving

            if not self._init_done:
                self._pos = end
                if btype == b"moov":
                    out.append(self._take(end, "init"))
                    self._init_done = True
                continue

            if btype == b"moof":
                if self._pos:
                    # A previous fragment never saw its mdat; close it so the
                    # stream stays in step rather than merging two fragments.
                    out.append(self._take(self._pos, "segment"))
                    end = box_size  # the moof now sits at offset 0
                self._pos = end
                self._in_fragment = True
                continue

            if not self._in_fragment:
                # Not part of any fragment (the mfra the muxer writes after the
                # last one, say): the SourceBuffer wants fragments only.
                logger.debug("Skipping top-level %r box outside a fragment", btype)
                del self._buf[:end]
                self._pos = 0
                continue

            if btype == b"mdat":
                out.append(self._take(end, "segment"))
                self._in_fragment = False
                continue

            self._pos = end  # some other box inside the fragment
        return out


class StreamingCmafEncoder:
    """One long-lived ffmpeg session producing a CMAF stream for one request.

    Frames go in a chunk at a time; completed fragments come out as
    ``(tag, payload)`` pairs using the tags in :mod:`dynamo.common.utils.cmaf_video`.
    Encoder selection is DEP 0016's -- ``DYN_XPU_FFMPEG_PATH`` set means hardware
    AV1, unset means software VP9 -- and there is no fallback either way.

    Three pipes are pumped concurrently: writing stdin while ffmpeg blocks on a
    full stdout pipe would deadlock, so stdout and stderr are drained by
    background tasks for the life of the process.

    A fragmented muxer cannot finalize fragment *k* until it sees the first frame
    of *k+1* or end of input, so the last fragment only appears once
    :meth:`finish` closes stdin.

    Usage::

        enc = StreamingCmafEncoder(fps, width, height)
        await enc.start()
        for chunk in iter_cmaf_chunks(frames, enc.gop_frames):
            async for tag, payload in enc.push(chunk):
                ...
        async for tag, payload in enc.finish():
            ...
    """

    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        *,
        gop_frames: Optional[int] = None,
        read_timeout_s: float = CMAF_READ_TIMEOUT_S,
    ) -> None:
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.gop_frames = int(
            gop_frames if gop_frames is not None else DEFAULT_CMAF_GOP_FRAMES
        )
        self._read_timeout_s = read_timeout_s

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._encoder: Optional[str] = None
        self._cutter = FragmentedMp4Cutter()
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stderr = bytearray()
        self._stdout_eof = False
        self._stdin_closed = False
        self._segments = 0
        self._frames_in = 0
        self._init_segment: Optional[bytes] = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn ffmpeg and start draining its output pipes."""
        if self._proc is not None:
            raise RuntimeError("StreamingCmafEncoder.start() called twice")

        ffmpeg, device = resolve_streaming_ffmpeg()
        self._encoder = HW_VIDEO_ENCODER if device else SW_VIDEO_ENCODER
        cmd = build_ffmpeg_command(
            ffmpeg,
            width=self.width,
            height=self.height,
            fps=self.fps,
            output="pipe:1",
            hw_device=device,
            gop=self.gop_frames,
            streaming=True,
        )
        logger.info("CMAF encode session: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stdout_task = asyncio.create_task(self._pump_stdout())
        self._stderr_task = asyncio.create_task(self._pump_stderr())

    async def aclose(self) -> None:
        """Tear the session down without waiting for a clean end of stream.

        Safe to call on a client disconnect or from an exception path, and safe
        to call twice. Always reaps the process.
        """
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._stdout_task = self._stderr_task = None

    async def __aenter__(self) -> "StreamingCmafEncoder":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # -- streaming ---------------------------------------------------------

    async def push(self, frames: np.ndarray) -> AsyncIterator[Tuple[str, bytes]]:
        """Encode one chunk of canonical frames, yielding completed fragments.

        Yields nothing for a chunk whose fragment is not finished yet, which is
        normal: the muxer needs the next chunk's keyframe first.

        Raises:
            ValueError: If ``frames`` is not canonical, or its geometry does not
                match the session's.
            RuntimeError: If ffmpeg has exited.
        """
        proc = self._require_started()
        _validate_canonical_frames(frames)
        _, height, width, _ = frames.shape
        if (width, height) != (self.width, self.height):
            raise ValueError(
                f"chunk geometry {width}x{height} does not match the CMAF session's "
                f"{self.width}x{self.height}"
            )
        if proc.returncode is not None:
            raise await self._failure()

        assert proc.stdin is not None
        try:
            proc.stdin.write(np.ascontiguousarray(frames).tobytes())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            raise await self._failure() from None
        self._frames_in += len(frames)

        for item in self._drain_ready():
            yield item
        if self._stdout_eof:
            # ffmpeg closed stdout while we still have frames to give it.
            raise await self._failure()

    async def finish(self) -> AsyncIterator[Tuple[str, bytes]]:
        """Close the input and yield the fragments that remain.

        Raises:
            RuntimeError: If ffmpeg exits non-zero. Its stderr is included.
        """
        proc = self._require_started()
        await self._close_stdin()

        while not self._stdout_eof:
            try:
                item = await asyncio.wait_for(
                    self._out_q.get(), timeout=self._read_timeout_s
                )
            except asyncio.TimeoutError:
                await self.aclose()
                raise RuntimeError(
                    f"ffmpeg {self._encoder} produced no output for "
                    f"{self._read_timeout_s:.0f}s while draining the CMAF stream"
                ) from None
            if item is None:
                self._stdout_eof = True
                break
            for kind, payload in self._cutter.feed(item):
                yield self._tag(kind, payload)

        for kind, payload in self._cutter.flush():
            yield self._tag(kind, payload)

        returncode = await proc.wait()
        if self._stderr_task is not None:
            with contextlib.suppress(Exception):
                await self._stderr_task
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg {self._encoder} CMAF encode failed (exit {returncode}): "
                f"{self._stderr_text()}"
            )
        logger.info(
            "CMAF session done: %d frame(s) in, %d segment(s) out via %s",
            self._frames_in,
            self._segments,
            self._encoder,
        )

    # -- introspection -----------------------------------------------------

    @property
    def encoder(self) -> Optional[str]:
        """The resolved ffmpeg encoder name, once :meth:`start` has run."""
        return self._encoder

    @property
    def init_segment(self) -> Optional[bytes]:
        """The init segment, once it has been drained."""
        return self._init_segment

    def codec_string(self) -> str:
        """RFC 6381 codec string for this stream.

        Read from the init segment, so it is only exact once the first fragment
        has been drained; before that, and if the parse fails, it falls back to
        the resolved encoder's typical configuration.
        """
        if self._init_segment is not None:
            parsed = codec_string_from_init(self._init_segment)
            if parsed:
                return parsed
        return CMAF_FALLBACK_VIDEO_CODEC.get(
            self._encoder or SW_VIDEO_ENCODER,
            CMAF_FALLBACK_VIDEO_CODEC[SW_VIDEO_ENCODER],
        )

    # -- internals ---------------------------------------------------------

    def _require_started(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            raise RuntimeError("StreamingCmafEncoder.start() has not been called")
        return self._proc

    def _tag(self, kind: str, payload: bytes) -> Tuple[str, bytes]:
        if kind == "init":
            self._init_segment = payload
            return CMAF_INIT_TAG, payload
        self._segments += 1
        return segment_tag(self._segments), payload

    def _drain_ready(self) -> List[Tuple[str, bytes]]:
        """Cut whatever stdout bytes have already arrived, without waiting."""
        data = bytearray()
        while True:
            try:
                item = self._out_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._stdout_eof = True
                break
            data += item
        if not data:
            return []
        return [self._tag(kind, payload) for kind, payload in self._cutter.feed(data)]

    async def _pump_stdout(self) -> None:
        """Move stdout into a queue so a full pipe can never block our writes."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                data = await proc.stdout.read(65536)
                if not data:
                    break
                await self._out_q.put(data)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - reported through the queue's EOF
            logger.warning("CMAF stdout reader failed: %s", err)
        finally:
            self._out_q.put_nowait(None)

    async def _pump_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        with contextlib.suppress(asyncio.CancelledError, Exception):
            while True:
                data = await proc.stderr.read(4096)
                if not data:
                    break
                # Keep the tail: ffmpeg's useful diagnostics are the last lines.
                self._stderr += data
                if len(self._stderr) > 65536:
                    del self._stderr[:-65536]

    def _stderr_text(self) -> str:
        return bytes(self._stderr).decode("utf-8", errors="replace").strip()

    async def _close_stdin(self) -> None:
        proc = self._require_started()
        if self._stdin_closed or proc.stdin is None:
            return
        self._stdin_closed = True
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, Exception):
            proc.stdin.close()
            await proc.stdin.wait_closed()

    async def _failure(self) -> RuntimeError:
        """Reap the process and build the error describing why it died."""
        proc = self._require_started()
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        if self._stderr_task is not None:
            with contextlib.suppress(Exception):
                await self._stderr_task
        return RuntimeError(
            f"ffmpeg {self._encoder} CMAF encode failed (exit {proc.returncode}) "
            f"after {self._frames_in} frame(s) and {self._segments} segment(s): "
            f"{self._stderr_text()}"
        )
