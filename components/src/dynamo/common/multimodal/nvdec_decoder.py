# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU (NVDEC) video decode for H.264/H.265 via PyNvVideoCodec.

Dual-path companion to the VP8/VP9 in-tree FFmpeg decoders. Royalty-free codecs
(VP8/VP9/AV1) stay on the existing CPU path; H.264/H.265 decode on the GPU
through NVDEC, which links ``libnvcuvid`` at runtime and carries no bundled
software codec. NVDEC does not use SMs beyond a small YUV->RGB conversion, so the
inference-time impact is minimal.

This module is backend-agnostic. Each backend's decode site (vLLM ``VideoLoader``,
TRT-LLM ``multimodal_processor``, SGLang encode worker) routes on the probed codec:

    codec = probe_video_codec(video_bytes)
    if should_use_nvdec(codec):
        frames, meta = decode_video_nvdec(video_bytes, num_frames)
    else:
        frames, meta = <existing decoder>

The returned frames match the existing ``VideoLoader`` contract: a host numpy
array of shape ``(T, H, W, 3)``, dtype ``uint8``, RGB, C-contiguous, plus a
metadata dict ``{"fps", "frames_indices", "total_num_frames"}``.

Gating: NVDEC is used when PyNvVideoCodec is importable and ``DYN_DISABLE_NVDEC``
is not set. When it is unavailable (CPU image, unsupported profile) the caller
raises the actionable unsupported-codec error built by
``common.multimodal.codec_errors``.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import tempfile
import warnings

import numpy as np

from dynamo.common.utils.env import env_bool

logger = logging.getLogger(__name__)

DISABLE_ENV = "DYN_DISABLE_NVDEC"
GPU_ID_ENV = "DYN_NVDEC_GPU_ID"

# Used only when the decoder cannot report a source frame rate (see
# _source_fps). Consumers divide by this to build frame timestamps, so it must
# stay non-zero; 30 fps is a common capture rate and keeps timestamps in a
# plausible range rather than failing the request outright.
FALLBACK_FPS = 30.0

# Codecs routed to NVDEC. VP8/VP9/AV1 stay on the existing royalty-free path.
HW_ROUTED_CODECS = frozenset({"h264", "hevc"})

# Container byte markers -> codec id. A video file carries exactly one video
# codec, so the first match wins. Covers ISO-BMFF (mp4/mov) sample-entry fourccs
# and Matroska/WebM CodecID strings. HEVC/AV1 checked before H.264 so a more
# specific marker is not shadowed. This is a routing hint, not a full demux --
# PyNvVideoCodec re-parses the stream authoritatively at decode time.
_CODEC_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"hev1", "hevc"),
    (b"hvc1", "hevc"),
    (b"V_MPEGH/ISO/HEVC", "hevc"),
    (b"av01", "av1"),
    (b"V_AV1", "av1"),
    (b"avc1", "h264"),
    (b"avc3", "h264"),
    (b"V_MPEG4/ISO/AVC", "h264"),
    (b"vp09", "vp9"),
    (b"V_VP9", "vp9"),
    (b"vp08", "vp8"),
    (b"V_VP8", "vp8"),
)


def probe_video_codec(data: bytes) -> str | None:
    """Best-effort codec identification from container bytes.

    Returns a codec id (``"h264"``/``"hevc"``/``"vp9"``/``"vp8"``/``"av1"``) or
    ``None`` when it can't tell. Scans the whole buffer because an mp4 ``moov``
    (which holds the codec fourcc) may sit at the end of a non-faststart file;
    multimodal clips are small and already in memory.
    """
    if not data:
        return None
    for marker, codec in _CODEC_MARKERS:
        if marker in data:
            return codec
    return None


def should_use_nvdec(codec: str | None) -> bool:
    """True if `codec` is one we route to NVDEC and NVDEC is available."""
    return codec in HW_ROUTED_CODECS and nvdec_available()


@functools.lru_cache(maxsize=1)
def nvdec_available() -> bool:
    """True if NVDEC decode can run in this process.

    Cached: PyNvVideoCodec is only installed in GPU images, and the import is the
    reliable signal. ``DYN_DISABLE_NVDEC`` forces the software path. A GPU that is
    absent at decode time still raises and is caught by the caller's fallback.
    """
    if env_bool(DISABLE_ENV):
        return False
    try:
        with warnings.catch_warnings():
            # A third-party deprecation must not decide whether the capability
            # exists. PyNvVideoCodec 2.2.0's __init__ does `from ast import Str`,
            # which warns on Python 3.12; under pytest's filterwarnings=error
            # that warning is *raised*, the import "fails", and this lru_cache
            # then pins False for the whole process -- silently disabling NVDEC
            # for every test in the session (a second import would have
            # succeeded, since the warning is only raised once).
            warnings.simplefilter("ignore")
            import PyNvVideoCodec  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        # Two very different situations reach here, and they need different
        # log levels. The wheel being absent is normal on CPU-only images, so
        # it stays at debug. The wheel being *installed* but unimportable is a
        # misconfiguration -- almost always a container that did not request
        # the "video" driver capability, so libnvcuvid is not mounted -- and it
        # silently costs H.264/H.265 decode, which the codec-compliant images
        # have no software fallback for. That deserves a warning; keeping it at
        # debug is how this class of failure goes unnoticed.
        if importlib.util.find_spec("PyNvVideoCodec") is None:
            logger.debug(
                "PyNvVideoCodec is not installed; NVDEC decode disabled (%s)", exc
            )
        else:
            logger.warning(
                "PyNvVideoCodec is installed but failed to import (%s); NVDEC "
                "hardware decode is DISABLED. This usually means the container "
                "was not given the 'video' driver capability, so libnvcuvid is "
                "unavailable -- set NVIDIA_DRIVER_CAPABILITIES to include "
                "'video'. H.264/H.265 video input has no software decoder in "
                "the codec-compliant images and will fail until this is fixed.",
                exc,
            )
        return False
    return True


def _gpu_id() -> int:
    raw = os.environ.get(GPU_ID_ENV, "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using GPU 0", GPU_ID_ENV, raw)
        return 0


def _frame_to_rgb_hwc(frame) -> np.ndarray:
    """Copy a decoded RGB frame into an owned host ``(H, W, 3)`` uint8 array.

    A 2.x ``DecodedFrame`` (``output_color_type=RGB``) supports the DLPack
    protocol, so torch wraps its buffer zero-copy. The decoder is constructed
    with ``use_device_memory=False``, so that buffer is already host memory:
    ``.cpu()`` is a no-op and ``.numpy()`` would keep aliasing decoder-owned
    memory, which the decoder is free to recycle for the next indexed read.
    The explicit ``np.array(..., copy=True)`` is what guarantees every
    collected frame owns its pixels (and normalizes dtype in the same step).
    Validated on PyNvVideoCodec 2.1.1 for H.264/H.265.
    """
    import torch

    try:
        tensor = torch.from_dlpack(frame)
    except Exception:  # noqa: BLE001 - fall back to the CUDA-array-interface path
        tensor = torch.as_tensor(frame, device="cuda")
    return np.array(tensor.cpu().numpy(), dtype=np.uint8, copy=True)


def _source_fps(decoder) -> float:
    """Source frame rate for the metadata dict. Never returns 0.

    This value is not cosmetic. ``load_video`` hands ``(frames, metadata)``
    straight to vLLM as ``mm_data["video"]``, and Qwen3-VL turns sampled frame
    indices into timestamps with ``idx / fps`` -- a zero raises
    ZeroDivisionError and the request 500s.

    PyNvVideoCodec 2.2.0 exposes the rate as
    ``get_stream_metadata().average_fps``; older probes are kept for other
    versions, then ``num_frames / duration``, then a documented fallback so a
    future API change degrades timestamp accuracy instead of failing requests.
    """
    meta = getattr(decoder, "get_stream_metadata", None)
    if callable(meta):
        try:
            info = meta()
            val = getattr(info, "average_fps", None)
            if val and float(val) > 0:
                return float(val)
            # Derive it when the rate is absent but the span is known.
            frames = float(getattr(info, "num_frames", 0) or 0)
            duration = float(getattr(info, "duration", 0) or 0)
            if frames > 0 and duration > 0:
                return frames / duration
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            pass

    for attr in ("get_fps", "fps", "GetFPS"):
        val = getattr(decoder, attr, None)
        try:
            val = val() if callable(val) else val
            if val and float(val) > 0:
                return float(val)
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            continue

    logger.warning(
        "NVDEC could not determine source fps; falling back to %.1f. Frame "
        "timestamps handed to the model will be approximate.",
        FALLBACK_FPS,
    )
    return FALLBACK_FPS


def decode_video_nvdec(
    data: bytes, num_frames: int, gpu_id: int | None = None
) -> tuple[np.ndarray, dict]:
    """Decode H.264/H.265 (or any NVDEC-supported codec) bytes to sampled frames.

    Returns ``(frames, metadata)`` where ``frames`` is host numpy
    ``(num_frames, H, W, 3)`` uint8 RGB C-contiguous, uniformly sampled across the
    clip, and ``metadata`` has ``fps``/``frames_indices``/``total_num_frames`` --
    the same contract as ``VideoLoader.load_video``. Raises on decode failure so
    the caller can fall back.
    """
    import PyNvVideoCodec as nvc

    if gpu_id is None:
        gpu_id = _gpu_id()
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    # SimpleDecoder takes a file PATH (not bytes) and reads frames lazily, so the
    # temp file must stay alive for the whole decode -- keep it inside the context.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        decoder = nvc.SimpleDecoder(
            tmp.name,
            gpu_id=gpu_id,
            output_color_type=nvc.OutputColorType.RGB,
            use_device_memory=False,
        )
        total = len(decoder)
        if total <= 0:
            raise RuntimeError("NVDEC decode produced no frames")
        n = min(num_frames, total)
        indices = np.unique(np.linspace(0, total - 1, n).astype(int))
        frames = [_frame_to_rgb_hwc(decoder[int(i)]) for i in indices]
        fps = _source_fps(decoder)

    stacked = np.ascontiguousarray(np.stack(frames)).astype(np.uint8, copy=False)
    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise RuntimeError(
            f"NVDEC frames have unexpected shape {stacked.shape}; expected (T,H,W,3)"
        )
    metadata = {
        "fps": fps,
        "frames_indices": indices.tolist(),
        "total_num_frames": int(total),
        # Whether the consumer may still sample these frames. vLLM's own video
        # loader reports `len(frames_indices) == total_num_frames`: true when the
        # loader handed over every source frame, false when it already picked a
        # subset. Qwen3-VL treats a MISSING flag as false, so omitting it here
        # made short clips -- the case where we return everything -- look
        # pre-sampled, and the model's own fps policy never ran. Reported with a
        # reproduction against vLLM 0.26.0's Qwen3-VL processor: the 10-frame
        # fixture yielded video_grid_thw [[5,16,20]] instead of [[2,16,20]].
        #
        # Reported by @Chokoyo on #11836. Note the E2E video tests cannot see
        # this: they assert on what the model says about the clip, which does not
        # change when the frame count does. The unit test is the check that can.
        "do_sample_frames": len(indices) == total,
    }
    return stacked, metadata
