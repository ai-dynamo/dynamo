# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encoding utilities for TensorRT-LLM video diffusion.

This module provides utilities for encoding numpy video frames to MP4 format.
"""

import io
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


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
            for frame in frames:
                writer.append_data(frame)
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
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        video_bytes = buffer.getvalue()
        logger.info(f"Encoded video to {len(video_bytes)} bytes")
        return video_bytes

    except Exception as e:
        logger.error(f"Failed to encode video to bytes: {e}")
        raise RuntimeError(f"Video encoding to bytes failed: {e}") from e
