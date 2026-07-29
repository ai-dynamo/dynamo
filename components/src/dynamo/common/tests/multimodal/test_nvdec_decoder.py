# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVDEC video decoder (PyNvVideoCodec mocked, no GPU)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from dynamo.common.multimodal import nvdec_decoder as nd

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def _clear_cache():
    nd.nvdec_available.cache_clear()
    yield
    nd.nvdec_available.cache_clear()


def _fake_pynv(num_frames: int = 10, h: int = 4, w: int = 6):
    """A stand-in PyNvVideoCodec module returning deterministic HWC uint8 frames."""
    mod = types.ModuleType("PyNvVideoCodec")

    class OutputColorType:
        RGB = "RGB"

    class SimpleDecoder:
        def __init__(
            self, src, gpu_id=0, output_color_type=None, use_device_memory=False
        ):
            self._n = num_frames

        def __len__(self):
            return self._n

        def __getitem__(self, i):
            return np.full((h, w, 3), i % 256, dtype=np.uint8)

        def get_fps(self):
            return 30.0

    mod.OutputColorType = OutputColorType
    mod.SimpleDecoder = SimpleDecoder
    return mod


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"\x00\x00\x00\x18ftypisom....avc1", "h264"),
        (b"....hev1....", "hevc"),
        (b"....hvc1....", "hevc"),
        (b"....vp09....", "vp9"),
        (b"....av01....", "av1"),
        (b"\x1aE\xdf\xa3....V_VP9....", "vp9"),
        (b"....V_MPEG4/ISO/AVC....", "h264"),
        (b"....V_MPEGH/ISO/HEVC....", "hevc"),
        (b"random bytes, no codec marker", None),
        (b"", None),
    ],
)
def test_probe_video_codec(data, expected):
    assert nd.probe_video_codec(data) == expected


def test_nvdec_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv(nd.DISABLE_ENV, "1")
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv())
    assert nd.nvdec_available() is False


def test_nvdec_available_false_when_not_importable(monkeypatch):
    monkeypatch.delenv(nd.DISABLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)  # -> ImportError
    assert nd.nvdec_available() is False


def test_nvdec_available_true_when_importable(monkeypatch):
    monkeypatch.delenv(nd.DISABLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv())
    assert nd.nvdec_available() is True


def test_nvdec_available_false_when_import_raises_runtime(monkeypatch):
    # PyNvVideoCodec raises RuntimeError (not ImportError) at import when the
    # NVDEC/NVENC driver libs aren't exposed (no NVIDIA_DRIVER_CAPABILITIES=video).
    monkeypatch.delenv(nd.DISABLE_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "PyNvVideoCodec", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PyNvVideoCodec":
            raise RuntimeError("Failed to load NVENC library: libnvidia-encode.so.1")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert nd.nvdec_available() is False


def test_should_use_nvdec_routes_only_h264_hevc(monkeypatch):
    monkeypatch.delenv(nd.DISABLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv())
    assert nd.should_use_nvdec("h264") is True
    assert nd.should_use_nvdec("hevc") is True
    assert nd.should_use_nvdec("vp9") is False
    assert nd.should_use_nvdec("av1") is False
    assert nd.should_use_nvdec(None) is False


def test_should_use_nvdec_false_when_unavailable(monkeypatch):
    monkeypatch.setenv(nd.DISABLE_ENV, "1")
    assert nd.should_use_nvdec("h264") is False


def test_decode_matches_frame_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=10, h=4, w=6)
    )
    # The real frame->host conversion is torch/DLPack on the GPU (validated on
    # hardware); stub it so this test runs on CPU CI.
    monkeypatch.setattr(
        nd, "_frame_to_rgb_hwc", lambda f: np.asarray(f, dtype=np.uint8)
    )
    frames, meta = nd.decode_video_nvdec(b"fakebytes", num_frames=5)
    assert frames.shape == (5, 4, 6, 3)  # THWC
    assert frames.dtype == np.uint8
    assert frames.flags["C_CONTIGUOUS"]
    assert meta["total_num_frames"] == 10
    assert len(meta["frames_indices"]) == 5
    assert meta["fps"] == 30.0


def test_decode_samples_uniformly(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=100))
    monkeypatch.setattr(
        nd, "_frame_to_rgb_hwc", lambda f: np.asarray(f, dtype=np.uint8)
    )
    _, meta = nd.decode_video_nvdec(b"x", num_frames=10)
    assert meta["frames_indices"][0] == 0
    assert meta["frames_indices"][-1] == 99


def test_decode_caps_at_total_frames(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=3))
    monkeypatch.setattr(
        nd, "_frame_to_rgb_hwc", lambda f: np.asarray(f, dtype=np.uint8)
    )
    frames, _ = nd.decode_video_nvdec(b"x", num_frames=32)
    assert frames.shape[0] == 3  # cannot sample more frames than exist


def test_decode_raises_on_empty_stream(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=0))
    with pytest.raises(RuntimeError):
        nd.decode_video_nvdec(b"x", num_frames=5)


def test_decode_rejects_bad_num_frames(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv())
    with pytest.raises(ValueError):
        nd.decode_video_nvdec(b"x", num_frames=0)
