# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import io
import logging
import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np

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
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-vaapi_device",
            device,
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
            # hwupload moves frames onto the VA-API surface; nv12 is 4:2:0, which
            # is what the encoder consumes and what players expect.
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            HW_VIDEO_ENCODER,
            "-global_quality",
            str(HW_VIDEO_GLOBAL_QUALITY),
            "-movflags",
            "+faststart",
            "-f",
            VIDEO_CONTAINER,
            output_path,
        ]

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
