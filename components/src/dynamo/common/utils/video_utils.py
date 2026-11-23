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

from dynamo.common.utils.ffmpeg_command import (
    HW_VIDEO_ENCODER,
    build_ffmpeg_command,
    hw_ffmpeg_path,
    validate_raw_rgb_frames,
    xpu_video_device,
)

logger = logging.getLogger(__name__)

# How long one hardware encode of a whole clip may take before it is abandoned.
HW_ENCODE_TIMEOUT_S = 600.0


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
    fps: float | None = None,
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
        return round(_seconds * _fps)
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


def _encode_av1_vaapi(
    frames: np.ndarray, fps: int, output_format: str, ffmpeg: str, device: str
) -> bytes:
    """Encode frames to AV1-in-mp4 on the VA-API hardware encoder.

    Pipes raw RGB to the operator-supplied ffmpeg. mp4 needs a seekable output
    for the moov atom, so the clip is encoded to a temporary file and read back.

    Raises:
        ValueError: If ``frames`` is not ``(T, H, W, 3) uint8``, or
            ``output_format`` is not "mp4".
        RuntimeError: If ffmpeg cannot be run, times out, or exits non-zero.
            Its stderr is included, and software encoding is not attempted.
    """
    if output_format != "mp4":
        raise ValueError(
            f"Hardware {HW_VIDEO_ENCODER} encoding supports only 'mp4'; "
            f"got {output_format!r}"
        )
    validate_raw_rgb_frames(frames, what="Hardware video encoder")
    num_frames, height, width, _ = frames.shape

    with tempfile.TemporaryDirectory(prefix="dynamo-av1-") as tmp_dir:
        output_path = os.path.join(tmp_dir, "out.mp4")
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
        try:
            proc = subprocess.run(
                cmd,
                input=np.ascontiguousarray(frames).tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=HW_ENCODE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg {HW_VIDEO_ENCODER} encode did not finish within "
                f"{HW_ENCODE_TIMEOUT_S:.0f}s"
            ) from None
        except OSError as err:
            raise RuntimeError(
                f"ffmpeg {HW_VIDEO_ENCODER} encode could not be started: {err}"
            ) from err
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg {HW_VIDEO_ENCODER} encode failed "
                f"(exit {proc.returncode}): {stderr}"
            )
        try:
            with open(output_path, "rb") as fh:
                video_bytes = fh.read()
        except OSError:
            video_bytes = b""
    if not video_bytes:
        raise RuntimeError(
            f"ffmpeg {HW_VIDEO_ENCODER} encode exited 0 but produced no output"
        )
    logger.info("Encoded video to %d bytes", len(video_bytes))
    return video_bytes


def encode_to_video_bytes(
    frames: np.ndarray,
    fps: int = 16,
    output_format: str = "mp4",
) -> bytes:
    """Encode numpy frames to video bytes (in-memory).

    Software VP9 (``libvpx-vp9``, through imageio) by default. When
    ``DYN_XPU_FFMPEG_PATH`` names a VA-API-capable ffmpeg, the clip is encoded
    as AV1 on the Intel GPU instead; see
    :mod:`dynamo.common.utils.ffmpeg_command`. The caller does not choose the
    codec, and a hardware encode that fails does not fall back to software.

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        fps: Frames per second for the output video.
        output_format: Container format — "mp4", "webm". Hardware encoding
            supports "mp4" only.

    Returns:
        Encoded video as bytes.

    Raises:
        ImportError: If imageio is not available (software encoding).
        ValueError: If hardware encoding is selected and ``frames`` is not
            ``(T, H, W, 3) uint8`` or ``output_format`` is not "mp4".
        RuntimeError: If encoding fails, or ``DYN_XPU_FFMPEG_PATH`` is set but
            unusable.
    """
    hw_ffmpeg = hw_ffmpeg_path()
    if hw_ffmpeg is not None:
        return _encode_av1_vaapi(
            frames, fps, output_format, hw_ffmpeg, xpu_video_device()
        )

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
