# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import io
import logging
import os
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


def normalize_video_frames(images) -> list:
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
    """Convert a list of PIL Images to a numpy array suitable for video encoding.

    Args:
        images: List of PIL Image objects (video frames).

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
            import imageio as iio
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
        # Use imageio to write MP4
        # imageio.v3 API
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="libx264")
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


def encode_to_mp4_bytes(
    frames: np.ndarray,
    fps: int = 16,
) -> bytes:
    """Encode numpy frames to MP4 bytes (in-memory).

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        fps: Frames per second for the output video.

    Returns:
        MP4 video as bytes.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    logger.info(f"Encoding {len(frames)} frames to bytes at {fps} fps")

    try:
        # Use in-memory buffer
        buffer = io.BytesIO()

        # imageio can write to BytesIO with format hint
        if hasattr(iio, "imwrite"):
            # v3 API - write to buffer
            iio.imwrite(buffer, frames, extension=".mp4", fps=fps, codec="libx264")
        else:
            # v2 API
            writer = iio.get_writer(
                buffer, format="FFMPEG", mode="I", fps=fps, codec="libx264"
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
