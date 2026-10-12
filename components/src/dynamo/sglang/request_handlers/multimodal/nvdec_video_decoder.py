# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An NVDEC-backed decoder that satisfies SGLang's ``VideoDecoderWrapper`` contract.

SGLang chooses which frames a video model sees. ``preprocess_video`` reads the
*source* frame count and frame rate, applies the model's ``vision_config.video``
policy (``fps`` / ``nframes`` / ``min_frames`` / ``max_frames``) via
``smart_nframes``, and only then asks the decoder for those specific frames::

    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        return vr, None
    total_frames, video_fps = len(vr), vr.avg_fps
    nframes = smart_nframes(video_config, total_frames=total_frames, video_fps=video_fps)
    idx = np.unique(np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64))
    video = vr.get_frames_as_tensor(idx.tolist())

Handing SGLang a pre-sampled ndarray takes the ``return vr, None`` branch, which
discards that policy: the model gets a backend-global frame count instead of the
count its own config asks for, with no true source metadata. Handing it this
class instead keeps frame selection where it belongs and reports real values.

The gate is ``isinstance``, so subclassing is required -- duck typing does not
pass. Three properties of the upstream class make that safe (verified against
``v0.5.16:python/sglang/srt/utils/video_decoder.py``):

* The module imports cleanly with neither torchcodec nor decord installed --
  ``_BACKEND`` falls back to ``"decord"`` and the import of decord is deferred to
  construction. So the base class is importable in the codec-compliant image even
  though both software decoders are stripped from it.
* ``close()`` touches only ``self._tmp_path`` and ``__del__`` just calls
  ``close()``. ``SimpleDecoder`` needs a file path and reads lazily, so the temp
  file has to outlive construction; recording it in ``_tmp_path`` makes the
  inherited cleanup exactly right.
* Only ``__len__``, ``avg_fps`` and ``get_frames_as_tensor`` are on the
  ``preprocess_video`` path.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np

from dynamo.common.multimodal.nvdec_decoder import (
    _frame_to_rgb_hwc,
    _gpu_id,
    _source_fps,
)

logger = logging.getLogger(__name__)

try:
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    SGLANG_VIDEO_DECODER_AVAILABLE = True
except Exception:  # noqa: BLE001 - SGLang absent (unit tests) or interface moved
    VideoDecoderWrapper = object  # type: ignore[assignment,misc]
    SGLANG_VIDEO_DECODER_AVAILABLE = False

# Latched so the pinned-memory regression warning is emitted once, not per request.
_PIN_MEMORY_WARNED = False


class NvdecVideoDecoder(VideoDecoderWrapper):  # type: ignore[misc,valid-type]
    """Random-access NVDEC decoder over in-memory video bytes.

    Construct from the bytes Dynamo has already fetched and policy-validated, so
    the SSRF checks in ``validate_media_url`` / ``fetch_bytes`` still apply.
    SGLang's own ``_normalize_video_input`` would fetch http(s) with no policy
    check at all, which is why the bytes are passed in rather than the URL.
    """

    def __init__(self, data: bytes, gpu_id: int | None = None) -> None:
        import PyNvVideoCodec as nvc

        if gpu_id is None:
            gpu_id = _gpu_id()

        # SimpleDecoder takes a PATH and reads frames lazily, so the file must
        # outlive __init__. Mirrors what the parent's decord-bytes branch does.
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

        # Deliberately not calling super().__init__(): it would construct a
        # decord/torchcodec decoder, which this image does not ship. Set the
        # attributes it would have set so inherited members still work --
        # close()/__del__ need _tmp_path, and source_bytes reads the rest.
        self._source = tmp_path
        self._source_bytes = data
        self._source_path: str | None = None
        # Annotated because close() clears it; without this mypy infers `str`
        # from the assignment and rejects the reset.
        self._tmp_path: str | None = tmp_path
        self._num_decode_threads = 1
        self._tc_kwargs: dict[str, Any] = {}

        try:
            self._decoder = nvc.SimpleDecoder(
                tmp_path,
                gpu_id=gpu_id,
                output_color_type=nvc.OutputColorType.RGB,
                use_device_memory=False,
            )
            if len(self._decoder) <= 0:
                raise RuntimeError("NVDEC opened the video but reports no frames")
        except Exception:
            # Do not leak the temp file when construction fails -- __del__ never
            # runs for an object whose __init__ raised.
            self.close()
            raise

    # __len__, close and __del__ are defined here rather than inherited. The
    # parent's versions are equivalent, but when SGLang is unavailable the base
    # collapses to `object` (see the import guard above) and inheriting nothing
    # would silently break len() and leak temp files under unit test.

    def __len__(self) -> int:
        return len(self._decoder)

    def close(self) -> None:
        """Remove the backing temp file. Idempotent; safe before _decoder exists."""
        tmp_path = getattr(self, "_tmp_path", None)
        if tmp_path is not None:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self._tmp_path = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def avg_fps(self) -> float:
        """True source frame rate. Never 0 -- see ``_source_fps``."""
        return _source_fps(self._decoder)

    def get_frames_as_tensor(self, indices: list):
        """Frames at ``indices`` as an NHWC uint8 CPU tensor in pinned memory.

        NHWC because ``preprocess_video`` immediately does
        ``video.permute(0, 3, 1, 2)``. Pinned host memory matches the parent's
        decord branch and keeps the tensor off the GPU, which also avoids the
        ``pin_memory()`` failure that a CUDA-resident tensor would raise.
        """
        import torch

        if not indices:
            raise ValueError("get_frames_as_tensor requires at least one index")
        frames = [_frame_to_rgb_hwc(self._decoder[int(i)]) for i in indices]
        arr = np.ascontiguousarray(np.stack(frames)).astype(np.uint8, copy=False)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"NVDEC frames have unexpected shape {arr.shape}; expected (N,H,W,3)"
            )
        tensor = torch.from_numpy(arr)
        try:
            return tensor.pin_memory()
        except RuntimeError as exc:
            # Pinned host memory is the point of this return type: it lets the
            # downstream host-to-device copy run async instead of staging through
            # a bounce buffer, and it is what the upstream decord branch does.
            # Losing it is a real throughput regression, so say so loudly -- once,
            # since this sits on the per-request path. The only expected cause is
            # a host whose default accelerator is not CUDA (a non-CUDA dev box);
            # on a deployed encode worker, which requires CUDA for NVDEC anyway,
            # this should never fire. Serving unpinned still beats failing the
            # request outright.
            global _PIN_MEMORY_WARNED
            if not _PIN_MEMORY_WARNED:
                _PIN_MEMORY_WARNED = True
                logger.warning(
                    "pin_memory() failed (%s); serving video frames from unpinned "
                    "host memory. Host-to-device transfer is slower than designed "
                    "-- expect reduced encode throughput. This is not expected on "
                    "a CUDA host.",
                    exc,
                )
            return tensor

    def __getitem__(self, idx):
        """Single frame as numpy NHWC uint8, matching the parent's contract."""
        return _frame_to_rgb_hwc(self._decoder[int(idx)])
