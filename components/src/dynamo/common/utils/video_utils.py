# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

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


def normalize_image_frames(images: list) -> list:
    """Normalize stage_output.images into a flat list of PIL Images.

    Image diffusion pipelines usually return PIL Images, but some (e.g. the
    Cosmos3 native pipeline) return numpy arrays shaped ``[batch, frames, H, W,
    C]`` even for single images. Collapse leading batch/frame dims and convert
    each frame to a PIL Image; PIL inputs pass through unchanged.
    """
    from PIL import Image

    out: list = []
    for item in images:
        if isinstance(item, Image.Image):
            out.append(item)
            continue
        arr = np.asarray(item)
        while arr.ndim > 4:  # [batch, frames, H, W, C] -> [frames, H, W, C]
            arr = arr[0]
        if arr.dtype != np.uint8:  # frames share a dtype/range; convert once
            arr = ((arr.clip(0, 1) * 255).round() if arr.max() <= 1.0 else arr).astype(
                np.uint8
            )
        frames = arr if arr.ndim == 4 else arr[None]  # -> [N, H, W, C]
        for frame in frames:
            out.append(Image.fromarray(frame))
    return out


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
        # Use imageio to write MP4. We use h264_nvenc (NVIDIA HW encoder) instead
        # of libx264 because the in-tree ffmpeg build is LGPL-only and libx264
        # is GPL-licensed; see container/templates/wheel_builder.Dockerfile.
        # Requires a CUDA-capable GPU at runtime.
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="h264_nvenc")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="h264_nvenc")  # type: ignore[attr-defined]
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


def _rgb_to_yuv420p(frames: np.ndarray) -> bytes:
    """Convert RGB frames (N, H, W, 3) uint8 to planar YUV420p bytes.

    Done in numpy (BT.601, full range) so ffmpeg never performs the RGB->YUV
    conversion itself: the in-tree LGPL ffmpeg's libswscale RGB->YUV path is
    broken and collapses chroma (greens render as magenta). H and W must be even.
    """
    rgb = frames.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    n, h, w = y.shape
    y = y.round().clip(0, 255).astype(np.uint8)
    # 4:2:0 -- box-average each 2x2 chroma block
    u = u.reshape(n, h // 2, 2, w // 2, 2).mean((2, 4)).round().clip(0, 255).astype(np.uint8)
    v = v.reshape(n, h // 2, 2, w // 2, 2).mean((2, 4)).round().clip(0, 255).astype(np.uint8)
    out = bytearray()
    for i in range(n):
        out += y[i].tobytes() + u[i].tobytes() + v[i].tobytes()
    return bytes(out)


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
        RuntimeError: If encoding fails.
    """
    import subprocess
    import tempfile

    codec = {"mp4": "h264_nvenc", "webm": "libvpx-vp9"}.get(output_format)
    if codec is None:
        raise ValueError(f"No codec specified for response format: {output_format}")

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames of shape (N, H, W, 3), got {frames.shape}")
    n, h, w, _ = frames.shape
    h, w = h & ~1, w & ~1  # yuv420p needs even dimensions
    frames = frames[:, :h, :w, :]

    logger.info(f"Encoding {n} frames to {output_format} bytes at {fps} fps")

    # Pre-convert RGB->YUV420p in numpy and feed planar YUV directly, bypassing
    # the in-tree ffmpeg's broken libswscale RGB->YUV path.
    yuv = _rgb_to_yuv420p(frames)
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
        "-r", str(fps), "-color_range", "pc", "-i", "-",
        "-c:v", codec, "-pix_fmt", "yuv420p", "-color_range", "pc",
    ]
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as tmp:
        try:
            subprocess.run(cmd + [tmp.name], input=yuv, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Video encoding to bytes failed: {e.stderr.decode(errors='replace')}"
            ) from e
        tmp.seek(0)
        video_bytes = tmp.read()

    logger.info(f"Encoded video to {len(video_bytes)} bytes")
    return video_bytes
