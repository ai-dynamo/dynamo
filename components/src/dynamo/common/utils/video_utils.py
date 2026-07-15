# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import io
import logging
import os
import shutil
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
        # Use imageio to write MP4
        # imageio.v3 API
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="libx264")  # type: ignore[attr-defined]
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

        kwargs: dict = {"fps": fps}
        if output_format == "webm":
            kwargs["codec"] = "libvpx-vp9"
        elif output_format == "mp4":
            kwargs["codec"] = "libx264"
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
# (``ensure_uint8_rgb`` / ``pil_frames_to_array`` / ``drop_alpha``). The encoder
# validates the canonical contract on entry and dispatches to one of two paths:
#
#   * imageio -> ffmpeg (NVENC)  -- preserves the existing NVIDIA behavior.
#   * ffmpeg CLI (raw pipe)      -- new hardware (XPU) / codecs that imageio's
#                                   ffmpeg plugin cannot reach.
#
# Encoding controls are read from DYN_VIDEO_* environment variables for now.
# ---------------------------------------------------------------------------


# Logical codec name -> ffmpeg encoder, per hardware path.
_FFMPEG_ENCODERS = {
    "nvenc": {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "vp9": "libvpx-vp9"},
    "xpu": {"h264": "h264_vaapi", "hevc": "hevc_vaapi", "vp9": "vp9_vaapi"},
    "cpu": {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9"},
}

# Default logical codec per container.
_DEFAULT_CODEC = {"mp4": "h264", "webm": "vp9"}

# Logical codecs each container can legally mux. Used to reject incompatible
# (container, codec) combinations before invoking the encoder.
_CONTAINER_CODECS = {
    "mp4": ("h264", "hevc"),
    "webm": ("vp9",),
}


def _video_codec() -> str | None:
    """Codec override from ``DYN_VIDEO_CODEC`` (e.g. ``h264`` / ``hevc`` / ``vp9``)."""
    val = os.environ.get("DYN_VIDEO_CODEC")
    return val.strip().lower() if val else None


def _video_container() -> str:
    """Container from ``DYN_VIDEO_CONTAINER`` (default ``mp4``)."""
    return os.environ.get("DYN_VIDEO_CONTAINER", "mp4").strip().lower()


def _video_hw_accel() -> str:
    """HW accelerator from ``DYN_VIDEO_HW_ACCEL`` (``auto``/``nvenc``/``xpu``/``cpu``)."""
    return os.environ.get("DYN_VIDEO_HW_ACCEL", "auto").strip().lower()


def _video_device() -> str:
    """DRM render node / device for HW encoding.

    Read from ``DYN_VIDEO_DEVICE``, falling back to the legacy
    ``DYNAMO_VAAPI_DEVICE`` and finally ``/dev/dri/renderD128``.
    """
    return (
        os.environ.get("DYN_VIDEO_DEVICE")
        or os.environ.get("DYNAMO_VAAPI_DEVICE")
        or "/dev/dri/renderD128"
    )


def _running_on_xpu() -> bool:
    """Return True when torch reports an available XPU backend."""
    try:
        import torch

        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False


def _resolve_ffmpeg_encoder(container: str, codec: str | None, hw_accel: str) -> str:
    """Map (container, logical codec, hw path) to an ffmpeg encoder name.

    ``container`` / ``codec`` / ``hw_accel`` are assumed already validated by
    ``encode_video``. Raises ``ValueError`` if no encoder entry exists for the
    resolved (hw_accel, codec) pair -- i.e. the lookup tables are inconsistent.
    """
    codec = codec or _DEFAULT_CODEC[container]
    table = _FFMPEG_ENCODERS[hw_accel]
    encoder = table.get(codec)
    if encoder is None:
        raise ValueError(
            f"No ffmpeg encoder for codec {codec!r} on {hw_accel!r}; "
            f"available: {sorted(table)}"
        )
    return encoder


def _validate_container_codec(container: str, codec: str | None) -> None:
    """Reject container/codec pairs that are not muxing-compatible.

    A ``codec`` of ``None`` means "use the container's default", which is
    compatible by construction. Raises ``ValueError`` for an unknown container
    or a codec the container cannot carry (e.g. H.264 in webm).
    """
    allowed = _CONTAINER_CODECS.get(container)
    if allowed is None:
        raise ValueError(
            f"Unsupported container {container!r}; "
            f"supported: {sorted(_CONTAINER_CODECS)}"
        )
    effective = codec or _DEFAULT_CODEC[container]
    if effective not in allowed:
        raise ValueError(
            f"Codec {effective!r} is not compatible with container "
            f"{container!r}; supported: {list(allowed)}"
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


def _encode_video_imageio(
    frames: np.ndarray, fps: int, container: str, codec: str | None
) -> bytes:
    """Encode via imageio -> ffmpeg (NVENC path). Preserves existing behavior."""
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

    encoder = _resolve_ffmpeg_encoder(container, codec, "nvenc")
    buffer = io.BytesIO()
    if hasattr(iio, "imwrite"):
        iio.imwrite(buffer, frames, extension=f".{container}", fps=fps, codec=encoder)
    else:
        writer = iio.get_writer(  # type: ignore[attr-defined]
            buffer, format="FFMPEG", mode="I", fps=fps, codec=encoder
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
    return buffer.getvalue()


def _encode_video_ffmpeg_cli(
    frames: np.ndarray,
    fps: int,
    container: str,
    codec: str | None,
    hw_accel: str,
    device: str,
) -> bytes:
    """Encode by piping raw RGB frames to the ffmpeg CLI (XPU / new codecs).

    mp4 needs a seekable output for the moov atom, so we encode to a temp file
    and read the bytes back.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH; required for video encoding")

    num_frames, height, width, _ = frames.shape
    encoder = _resolve_ffmpeg_encoder(container, codec, hw_accel)

    with tempfile.NamedTemporaryFile(suffix=f".{container}", delete=False) as tmp:
        output_path = tmp.name
    try:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if hw_accel == "xpu":
            cmd += ["-vaapi_device", device]
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
        if hw_accel == "xpu":
            cmd += ["-vf", "format=nv12,hwupload"]
        cmd += ["-c:v", encoder]
        if hw_accel == "cpu":
            # Software encoders default to yuv444p from rgb24 input, which most
            # players (browsers, mobile) cannot decode; force widely-compatible
            # 4:2:0. The XPU path already gets 4:2:0 via the nv12 hwupload above.
            cmd += ["-pix_fmt", "yuv420p"]
        if container == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += ["-f", container, output_path]

        logger.info(
            "Encoding %d frames (%dx%d @ %d fps) via %s on %s",
            num_frames,
            width,
            height,
            fps,
            encoder,
            device if hw_accel == "xpu" else hw_accel,
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
                f"ffmpeg {encoder} encode failed (exit {proc.returncode}): {stderr}"
            )
        with open(output_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def encode_video(
    frames: np.ndarray,
    fps: int = DEFAULT_VIDEO_FPS,
    *,
    container: str | None = None,
    codec: str | None = None,
    hw_accel: str | None = None,
    device: str | None = None,
) -> bytes:
    """Unified video encoder: encode canonical frames to video bytes.

    ``frames`` must already be in the canonical format -- an ``np.ndarray`` of
    shape ``(T, H, W, 3)``, dtype ``uint8``, RGB. Each backend converts its
    native output with its own ``to_canonical()`` before calling this.

    Any explicit argument overrides its ``DYN_VIDEO_*`` environment variable,
    which in turn overrides platform auto-detection. Dispatches to the imageio
    NVENC path on NVIDIA, or the ffmpeg-CLI path for XPU / CPU.

    Args:
        frames: Canonical ``np.ndarray (T, H, W, 3) uint8`` RGB frames.
        fps: Frames per second for the output video.
        container: ``mp4`` / ``webm`` (env: ``DYN_VIDEO_CONTAINER``).
        codec: Logical codec ``h264`` / ``hevc`` / ``vp9`` (env: ``DYN_VIDEO_CODEC``).
        hw_accel: ``auto`` / ``nvenc`` / ``xpu`` / ``cpu`` (env: ``DYN_VIDEO_HW_ACCEL``).
            ``auto`` picks a hardware encoder (XPU if present, else NVENC) and
            never selects CPU -- request ``cpu`` explicitly for software encoding.
        device: HW device / DRM render node (env: ``DYN_VIDEO_DEVICE``).

    Returns:
        Encoded video as bytes.

    Raises:
        ValueError: If ``frames`` is not the canonical ``(T, H, W, 3) uint8`` array.
    """
    _validate_canonical_frames(frames)

    container = (container or _video_container()).lower()
    codec = codec.lower() if codec else _video_codec()
    _validate_container_codec(container, codec)

    hw_accel = (hw_accel or _video_hw_accel()).lower()
    if hw_accel == "auto":
        # "auto" selects a hardware encoder only: XPU when present, otherwise
        # NVENC. It never resolves to CPU -- software encoding is slow and must
        # be requested explicitly (hw_accel="cpu" / DYN_VIDEO_HW_ACCEL=cpu). A
        # deployment with neither NVIDIA nor XPU is unsupported by design.
        hw_accel = "xpu" if _running_on_xpu() else "nvenc"
    if hw_accel not in _FFMPEG_ENCODERS:
        raise ValueError(
            f"Unsupported hw_accel {hw_accel!r}; "
            f"supported: {sorted(_FFMPEG_ENCODERS)} (or 'auto')"
        )

    logger.info(
        "Encoding %d frames -> %s (hw=%s codec=%s) at %d fps",
        len(frames),
        container,
        hw_accel,
        codec or "default",
        fps,
    )

    if hw_accel == "nvenc":
        return _encode_video_imageio(frames, fps, container, codec)

    device = device or _video_device()
    return _encode_video_ffmpeg_cli(frames, fps, container, codec, hw_accel, device)
