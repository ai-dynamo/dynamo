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
import math
import os
import tempfile
import warnings
from collections.abc import Iterator
from typing import TypeVar

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

# Read only codec fields in video tracks. Metadata, brands and compressed payloads
# can contain these same byte strings and must never influence decoder routing.
_CODEC_ROUTING_PRIORITY = ("hevc", "av1", "h264", "vp9", "vp8")
_MP4_SAMPLE_ENTRY_CODECS = {
    b"hev1": "hevc",
    b"hvc1": "hevc",
    b"av01": "av1",
    b"avc1": "h264",
    b"avc3": "h264",
    b"vp09": "vp9",
    b"vp08": "vp8",
}
_MATROSKA_CODEC_IDS = {
    b"V_MPEGH/ISO/HEVC": "hevc",
    b"V_AV1": "av1",
    b"V_MPEG4/ISO/AVC": "h264",
    b"V_VP9": "vp9",
    b"V_VP8": "vp8",
}

# Include one byte beyond the longest known ID to reject longer non-null IDs.
_MATROSKA_CODEC_ID_PREFIX_SIZE = max(map(len, _MATROSKA_CODEC_IDS)) + 1


# Direct Cluster children, plus the global Void and CRC-32 elements. Their
# payloads must be skipped even when the Cluster itself has no declared size.
_MATROSKA_CLUSTER_CHILD_IDS = frozenset(
    {
        0xE7,  # Timestamp
        0x5854,  # SilentTracks
        0xA7,  # Position
        0xAB,  # PrevSize
        0xA3,  # SimpleBlock
        0xA0,  # BlockGroup
        0xAF,  # EncryptedBlock
        0xEC,  # Void
        0xBF,  # CRC-32
    }
)


def _iter_mp4_boxes(
    data: bytes, start: int, end: int
) -> Iterator[tuple[bytes, int, int]]:
    """Yield (type, payload start, end) without copying or scanning payloads."""
    while start < end:
        if end - start < 8:
            raise ValueError("Truncated MP4 box")
        size = int.from_bytes(data[start : start + 4], "big")
        kind = data[start + 4 : start + 8]
        header_size = 8
        if size == 1:
            if end - start < 16:
                raise ValueError("Truncated extended MP4 box")
            size = int.from_bytes(data[start + 8 : start + 16], "big")
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size or size > end - start:
            raise ValueError("Invalid MP4 box size")
        yield kind, start + header_size, start + size
        start += size


_Element = TypeVar("_Element")


def _iter_complete_elements(elements: Iterator[_Element]) -> Iterator[_Element]:
    """Stop at an invalid outer boundary without discarding earlier elements.

    Use only outside track metadata. Errors while parsing a yielded element's
    contents must still propagate to reject incomplete or invalid tracks.
    """
    while True:
        try:
            element = next(elements)
        except (StopIteration, ValueError):
            return
        yield element


def _iter_mp4_path(
    data: bytes, start: int, end: int, path: tuple[bytes, ...]
) -> Iterator[tuple[int, int]]:
    # Recursion is bounded by the fixed container path, not by untrusted input.
    boxes = _iter_mp4_boxes(data, start, end)
    if path[0] == b"moov":
        # Only the file boundary is tolerant; moov and its children are strict.
        boxes = _iter_complete_elements(boxes)
    for kind, payload_start, box_end in boxes:
        if kind == path[0]:
            if len(path) == 1:
                yield payload_start, box_end
            else:
                yield from _iter_mp4_path(data, payload_start, box_end, path[1:])


def _probe_mp4_codecs(data: bytes) -> set[str]:
    codecs: set[str] = set()
    for start, end in _iter_mp4_path(data, 0, len(data), (b"moov", b"trak", b"mdia")):
        is_video = any(
            data[payload_start + 8 : payload_start + 12] == b"vide"
            for payload_start, box_end in _iter_mp4_path(data, start, end, (b"hdlr",))
            if box_end - payload_start >= 12
        )
        if not is_video:
            continue
        for payload_start, box_end in _iter_mp4_path(
            data, start, end, (b"minf", b"stbl", b"stsd")
        ):
            if box_end - payload_start < 8:
                raise ValueError("Truncated MP4 sample description")
            # stsd is a FullBox followed by entry_count and sample entries.
            entry_count = int.from_bytes(
                data[payload_start + 4 : payload_start + 8], "big"
            )
            entries_read = 0
            for kind, entry_start, entry_end in _iter_mp4_boxes(
                data, payload_start + 8, box_end
            ):
                # A VisualSampleEntry has a 78-byte fixed payload.
                if entry_end - entry_start < 78:
                    raise ValueError("Truncated MP4 visual sample entry")
                codec = _MP4_SAMPLE_ENTRY_CODECS.get(kind)
                if codec is not None:
                    codecs.add(codec)
                entries_read += 1
            if entries_read != entry_count:
                raise ValueError("Invalid MP4 sample entry count")
    return codecs


def _read_ebml_vint(
    data: bytes, start: int, end: int, *, element_id: bool = False
) -> tuple[int, int]:
    """Read an EBML ID or size and return its value and the next offset.

    Unknown sizes are returned as -1; IDs retain their VINT marker bit.
    """
    if start >= end or data[start] == 0:
        raise ValueError("Invalid EBML integer")
    width = 9 - data[start].bit_length()
    if width > (4 if element_id else 8) or start + width > end:
        raise ValueError("Truncated EBML integer")
    value = int.from_bytes(data[start : start + width], "big")
    if not element_id:
        value &= (1 << (7 * width)) - 1
        if value == (1 << (7 * width)) - 1:
            value = -1  # unknown-size master element
    return value, start + width


def _unknown_cluster_end(data: bytes, start: int, end: int) -> int:
    """Skip Cluster children up to the next sibling element or end of segment."""
    while start < end:
        kind, size_start = _read_ebml_vint(data, start, end, element_id=True)
        if kind not in _MATROSKA_CLUSTER_CHILD_IDS:
            return start
        size, payload_start = _read_ebml_vint(data, size_start, end)
        if size < 0 or size > end - payload_start:
            raise ValueError("Invalid Matroska Cluster child size")
        start = payload_start + size
    return end


def _iter_ebml_elements(
    data: bytes, start: int, end: int
) -> Iterator[tuple[int, int, int]]:
    while start < end:
        kind, payload_start = _read_ebml_vint(data, start, end, element_id=True)
        size, payload_start = _read_ebml_vint(data, payload_start, end)
        if size == -1:
            if kind == 0x18538067:  # Segment
                size = end - payload_start
            elif kind == 0x1F43B675:  # Cluster
                size = _unknown_cluster_end(data, payload_start, end) - payload_start
            else:
                raise ValueError("Unsupported unknown-size EBML element")
        if size > end - payload_start:
            raise ValueError("Truncated EBML element")
        yield kind, payload_start, payload_start + size
        start = payload_start + size


def _iter_ebml_path(
    data: bytes, start: int, end: int, path: tuple[int, ...]
) -> Iterator[tuple[int, int]]:
    elements = _iter_ebml_elements(data, start, end)
    if path[0] in (0x18538067, 0x1654AE6B):  # file or Segment boundary
        elements = _iter_complete_elements(elements)
    for kind, payload_start, element_end in elements:
        if kind == path[0]:
            if len(path) == 1:
                yield payload_start, element_end
            else:
                yield from _iter_ebml_path(data, payload_start, element_end, path[1:])


def _probe_matroska_codecs(data: bytes) -> set[str]:
    codecs: set[str] = set()
    # Segment / Tracks / TrackEntry. Only TrackType=1 is video; CodecID in an
    # audio track, attachment, tag or CodecPrivate blob is not a video codec.
    for start, end in _iter_ebml_path(
        data, 0, len(data), (0x18538067, 0x1654AE6B, 0xAE)
    ):
        track_type = None
        codec_id = None
        for kind, payload_start, element_end in _iter_ebml_elements(data, start, end):
            if kind == 0x83 and 0 < element_end - payload_start <= 8:
                track_type = int.from_bytes(data[payload_start:element_end], "big")
            elif kind == 0x86:
                # EBML strings end at the first null and may have arbitrary padding.
                prefix_end = min(
                    element_end, payload_start + _MATROSKA_CODEC_ID_PREFIX_SIZE
                )
                codec_id = data[payload_start:prefix_end].split(b"\0", 1)[0]
        if track_type == 1 and codec_id is not None:
            codec = _MATROSKA_CODEC_IDS.get(codec_id)
            if codec is not None:
                codecs.add(codec)
    return codecs


def probe_video_codec(data: bytes) -> str | None:
    """Identify a routing codec from MP4/MOV or Matroska/WebM video tracks.

    Skip payloads by their declared sizes, including an mdat before moov in a
    non-faststart MP4. Ignore invalid outer tails after complete track metadata.
    Return None for unrecognized or incomplete track metadata;
    never fall back to searching arbitrary bytes for codec markers. This is a
    routing hint; the selected decoder still selects and parses the stream.
    Container traversal is synchronous; async callers must offload it.
    """
    try:
        if data.startswith(b"\x1a\x45\xdf\xa3"):
            codecs = _probe_matroska_codecs(data)
        else:
            codecs = _probe_mp4_codecs(data)
    except ValueError:
        return None
    # Preserve the existing routing preference for multi-track files. The first
    # track need not be the stream selected by the decoder (e.g. a later track
    # may be marked as default). Only actual video codec fields are candidates.
    return next((codec for codec in _CODEC_ROUTING_PRIORITY if codec in codecs), None)


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
            # exists. PyNvVideoCodec 2.2.0's __init__ did `from ast import Str`,
            # which warns on Python 3.12; under pytest's filterwarnings=error
            # that warning is *raised*, the import "fails", and this lru_cache
            # then pins False for the whole process -- silently disabling NVDEC
            # for every test in the session (a second import would have
            # succeeded, since the warning is only raised once).
            #
            # 2.2.2 removed that import, so the shipped floor no longer trips
            # this. The suppression is kept because it is not specific to that
            # import: any third-party warning raised here would disable NVDEC
            # for the session, and the only signal would be a debug log.
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
    """Copy a decoded (device) RGB frame to a host ``(H, W, 3)`` uint8 array.

    A 2.x ``DecodedFrame`` (``output_color_type=RGB``) holds a CUDA buffer and
    supports the DLPack protocol, so torch wraps it zero-copy on the GPU and
    ``.cpu()`` copies to host. Returns a single frame: ``decode_video_nvdec`` stacks
    these into ``(num_frames, H, W, 3)``, so a batched replacement here has to drop
    that stack too.
    """
    import torch

    try:
        tensor = torch.from_dlpack(frame)
    except Exception:  # noqa: BLE001 - fall back to the CUDA-array-interface path
        tensor = torch.as_tensor(frame, device="cuda")
    arr = tensor.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _source_fps(decoder) -> float:
    """Source frame rate for the metadata dict. Never returns 0.

    This value is not cosmetic. ``load_video`` hands ``(frames, metadata)``
    straight to vLLM as ``mm_data["video"]``, and Qwen3-VL turns sampled frame
    indices into timestamps with ``idx / fps`` -- a zero raises
    ZeroDivisionError and the request 500s.

    PyNvVideoCodec exposes the rate as ``get_stream_metadata().average_fps``; older
    probes are kept for other versions, then ``num_frames / duration``, then a
    documented fallback so a future API change degrades timestamp accuracy instead
    of failing requests.
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


def _compute_num_frames(
    num_frames: int, target_fps: float, total_frames: int, source_fps: float
) -> int:
    """How many frames to sample: the lower of the two caps, never below 1.

    `target_fps` caps by duration (`total_frames / source_fps`), which is why
    this runs against an already-open clip rather than in the caller.
    """
    n = min(num_frames, total_frames)
    if target_fps > 0:
        n = max(1, min(n, math.floor((total_frames / source_fps) * target_fps)))
    return n


def decode_video_nvdec(
    data: bytes, num_frames: int, gpu_id: int | None = None, fps: float = -1
) -> tuple[np.ndarray, dict]:
    """Decode H.264/H.265 (or any NVDEC-supported codec) bytes to sampled frames.

    Returns ``(frames, metadata)`` where ``frames`` is host numpy
    ``(num_frames, H, W, 3)`` uint8 RGB C-contiguous, uniformly sampled across the
    clip, and ``metadata`` has ``fps``/``frames_indices``/``total_num_frames`` --
    the same contract as ``VideoLoader.load_video``. Raises on decode failure so
    the caller can fall back.

    ``num_frames`` is a cap, and ``fps`` an optional second one: sample at most
    that many frames per second of source, i.e. ``floor(duration * fps)``.
    ``fps <= 0`` (the default) leaves sampling to ``num_frames`` alone.
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
        source_fps = _source_fps(decoder)
        n = _compute_num_frames(num_frames, fps, total, source_fps)
        indices = np.unique(np.linspace(0, total - 1, n).astype(int))
        frames = [_frame_to_rgb_hwc(decoder[int(i)]) for i in indices]

    stacked = np.ascontiguousarray(np.stack(frames)).astype(np.uint8, copy=False)
    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise RuntimeError(
            f"NVDEC frames have unexpected shape {stacked.shape}; expected (T,H,W,3)"
        )
    metadata = {
        "fps": source_fps,
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
