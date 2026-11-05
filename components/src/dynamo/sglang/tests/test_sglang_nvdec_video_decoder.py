# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVDEC-backed SGLang video decoder.

PyNvVideoCodec is mocked throughout -- these never touch a GPU. The point is the
*contract* SGLang's ``preprocess_video`` relies on: an ``isinstance`` check
against ``VideoDecoderWrapper``, then ``len()``, ``avg_fps`` and
``get_frames_as_tensor(indices)``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dynamo.sglang.request_handlers.multimodal import nvdec_video_decoder as mod
from dynamo.sglang.request_handlers.multimodal.nvdec_video_decoder import (
    NvdecVideoDecoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_WIDTH, _HEIGHT, _TOTAL, _FPS = 8, 6, 50, 10.0


class _FakeSimpleDecoder:
    """Stands in for nvc.SimpleDecoder: len() plus random access by index."""

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.requested: list[int] = []

    def __len__(self) -> int:
        return _TOTAL

    def __getitem__(self, idx):
        self.requested.append(idx)
        # Encode the index into the pixels so tests can prove which frames came back.
        return np.full((_HEIGHT, _WIDTH, 3), idx % 256, dtype=np.uint8)

    def get_stream_metadata(self):
        return SimpleNamespace(average_fps=_FPS, num_frames=_TOTAL, duration=5.0)


@pytest.fixture
def fake_nvc(monkeypatch):
    """Install a fake PyNvVideoCodec module and a passthrough frame converter."""
    nvc = MagicMock()
    nvc.SimpleDecoder = _FakeSimpleDecoder
    nvc.OutputColorType.RGB = "RGB"
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", nvc)
    # The real converter expects a CUDA DecodedFrame; our fake yields numpy already.
    monkeypatch.setattr(mod, "_frame_to_rgb_hwc", lambda frame: frame)
    return nvc


@pytest.fixture
def decoder(fake_nvc):
    d = NvdecVideoDecoder(b"fake-mp4-bytes")
    yield d
    d.close()


def test_reports_true_source_length_and_fps(decoder) -> None:
    """SGLang derives its frame policy from these, so they must be the source values."""
    assert len(decoder) == _TOTAL
    assert decoder.avg_fps == _FPS


def test_get_frames_as_tensor_returns_nhwc_uint8_cpu(decoder) -> None:
    """preprocess_video does video.permute(0, 3, 1, 2), so NHWC is required."""
    out = decoder.get_frames_as_tensor([0, 10, 20])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, _HEIGHT, _WIDTH, 3)  # NHWC
    assert out.dtype == torch.uint8
    assert out.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned host memory requires CUDA"
)
# The one test here that touches CUDA. It stays consistent with the file's
# gpu_0 mark because gpu_0 means "no GPU required": on the CPU-only lane it
# skips itself, and on the GPU job's gpu_0 stage it runs and guards pinning.
def test_frames_are_pinned_on_cuda(decoder) -> None:
    """Pinned memory is the performance contract of this return type.

    It lets the downstream host-to-device copy overlap instead of staging through
    a bounce buffer. The implementation degrades to unpinned rather than failing
    a request, so without this assertion a regression would show up only as lost
    throughput -- silently, and never in CI.
    """
    out = decoder.get_frames_as_tensor([0, 1, 2])
    assert out.is_pinned(), "video frames must be returned in pinned host memory"


def test_get_frames_as_tensor_returns_exactly_the_requested_frames(decoder) -> None:
    """Frame selection belongs to SGLang: return its indices, not a resample."""
    indices = [1, 17, 33, 49]
    out = decoder.get_frames_as_tensor(indices)
    assert out.shape[0] == len(indices)
    assert decoder._decoder.requested == indices
    # Pixel value encodes the source index, proving identity rather than count.
    assert [int(out[i, 0, 0, 0]) for i in range(len(indices))] == indices


def test_get_frames_as_tensor_rejects_empty_indices(decoder) -> None:
    with pytest.raises(ValueError):
        decoder.get_frames_as_tensor([])


def test_close_removes_the_backing_temp_file(fake_nvc) -> None:
    """SimpleDecoder needs a real path, so the temp file must be cleaned up."""
    d = NvdecVideoDecoder(b"fake-mp4-bytes")
    tmp_path = d._tmp_path
    assert tmp_path is not None and os.path.exists(tmp_path)
    d.close()
    assert not os.path.exists(tmp_path)
    d.close()  # idempotent


def test_temp_file_not_leaked_when_construction_fails(fake_nvc, monkeypatch) -> None:
    """__del__ never runs for an object whose __init__ raised."""
    seen: dict = {}

    class _Boom:
        def __init__(self, path, **kwargs):
            seen["path"] = path
            raise RuntimeError("decoder init failed")

    fake_nvc.SimpleDecoder = _Boom
    with pytest.raises(RuntimeError):
        NvdecVideoDecoder(b"fake-mp4-bytes")
    assert not os.path.exists(seen["path"])


def test_rejects_a_video_with_no_frames(fake_nvc) -> None:
    class _Empty(_FakeSimpleDecoder):
        def __len__(self) -> int:
            return 0

    fake_nvc.SimpleDecoder = _Empty
    with pytest.raises(RuntimeError, match="no frames"):
        NvdecVideoDecoder(b"fake-mp4-bytes")


@pytest.mark.skipif(
    not mod.SGLANG_VIDEO_DECODER_AVAILABLE, reason="SGLang not installed"
)
def test_satisfies_sglangs_isinstance_gate(decoder) -> None:
    """preprocess_video gates on isinstance -- duck typing would silently no-op.

    Without this, preprocess_video takes its ``return vr, None`` branch and
    SGLang's frame-selection policy is skipped entirely.
    """
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    assert isinstance(decoder, VideoDecoderWrapper)
