# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVDEC video decoder (PyNvVideoCodec mocked, no GPU)."""

from __future__ import annotations

import logging
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


def _force_import_failure(monkeypatch, exc):
    monkeypatch.delenv(nd.DISABLE_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "PyNvVideoCodec", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PyNvVideoCodec":
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_installed_but_unimportable_warns_about_video_capability(monkeypatch, caplog):
    """A misconfigured container must not fail silently.

    The wheel being present but unimportable means libnvcuvid is missing -- the
    container was not granted the 'video' driver capability. That silently costs
    H.264/H.265 decode, which the codec-compliant images cannot do in software,
    so it has to be visible above debug level.
    """
    _force_import_failure(monkeypatch, RuntimeError("libnvcuvid.so.1: not found"))
    monkeypatch.setattr(
        nd.importlib.util, "find_spec", lambda name: object()
    )  # wheel IS installed
    with caplog.at_level(logging.WARNING):
        assert nd.nvdec_available() is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "video" in caplog.text and "NVIDIA_DRIVER_CAPABILITIES" in caplog.text


def test_absent_wheel_stays_quiet(monkeypatch, caplog):
    """CPU-only images legitimately have no wheel; that must not warn."""
    _force_import_failure(monkeypatch, ImportError("No module named PyNvVideoCodec"))
    monkeypatch.setattr(nd.importlib.util, "find_spec", lambda name: None)
    with caplog.at_level(logging.WARNING):
        assert nd.nvdec_available() is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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


def test_frame_to_rgb_hwc_copies_out_of_decoder_memory():
    """The decoder hands over host frames (``use_device_memory=False``) and is
    free to recycle the underlying buffer between indexed reads, so the
    conversion must copy. numpy arrays export DLPack exactly like a
    ``DecodedFrame`` does: without the copy, torch wraps the buffer zero-copy
    and mutating the source would corrupt the already-collected frame.
    """
    source = np.full((4, 6, 3), 7, dtype=np.uint8)

    converted = nd._frame_to_rgb_hwc(source)
    source[:] = 99  # the decoder reuses its buffer for the next frame

    assert converted.dtype == np.uint8
    assert converted.shape == (4, 6, 3)
    np.testing.assert_array_equal(converted, np.full((4, 6, 3), 7, dtype=np.uint8))


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


@pytest.mark.parametrize(
    "total,num_frames,expect_flag,expect_returned",
    [
        (3, 32, True, 3),  # fewer frames than asked for -> every source frame
        (32, 32, True, 32),  # exactly as many -> still every source frame
        (100, 10, False, 10),  # more than asked for -> we already sampled
        (1, 5, True, 1),  # single-frame clip
    ],
)
def test_decode_reports_whether_it_already_sampled(
    monkeypatch, total, num_frames, expect_flag, expect_returned
):
    """`do_sample_frames` must say whether the consumer may still sample.

    vLLM's own loader reports `len(frames_indices) == total_num_frames`, and
    Qwen3-VL reads a MISSING flag as False. Omitting it therefore made short
    clips -- exactly the case where every source frame is returned -- look
    pre-sampled, so the model's fps policy never ran and all frames were consumed
    instead of the four it would have chosen.

    Asserted here rather than end to end on purpose: the serve tests check what
    the model says about the clip, and that answer does not change when the frame
    count does. This is the level that can see it.
    """
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=total))
    monkeypatch.setattr(
        nd, "_frame_to_rgb_hwc", lambda f: np.asarray(f, dtype=np.uint8)
    )
    frames, meta = nd.decode_video_nvdec(b"x", num_frames=num_frames)

    assert meta["do_sample_frames"] is expect_flag
    assert frames.shape[0] == expect_returned
    # The flag is exactly vLLM's rule, restated against what we returned.
    assert meta["do_sample_frames"] == (
        len(meta["frames_indices"]) == meta["total_num_frames"]
    )


def test_decode_raises_on_empty_stream(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=0))
    with pytest.raises(RuntimeError):
        nd.decode_video_nvdec(b"x", num_frames=5)


def test_decode_rejects_bad_num_frames(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv())
    with pytest.raises(ValueError):
        nd.decode_video_nvdec(b"x", num_frames=0)


# ---------------------------------------------------------------------------
# _source_fps -- metadata["fps"] reaches vLLM as mm_data["video"], and Qwen3-VL
# computes timestamps as `idx / fps`, so a 0 here 500s the request.
# ---------------------------------------------------------------------------


class _Meta:
    def __init__(self, average_fps=None, num_frames=None, duration=None):
        if average_fps is not None:
            self.average_fps = average_fps
        if num_frames is not None:
            self.num_frames = num_frames
        if duration is not None:
            self.duration = duration


class _Decoder:
    def __init__(self, meta=None, **attrs):
        if meta is not None:
            self.get_stream_metadata = lambda: meta
        for k, v in attrs.items():
            setattr(self, k, v)


def test_source_fps_prefers_stream_metadata():
    """PyNvVideoCodec 2.2.0 reports the rate via get_stream_metadata()."""
    dec = _Decoder(meta=_Meta(average_fps=10.0), get_fps=lambda: 30.0)
    assert nd._source_fps(dec) == 10.0


def test_source_fps_derives_from_frames_and_duration():
    dec = _Decoder(meta=_Meta(num_frames=24, duration=2.0))
    assert nd._source_fps(dec) == 12.0


def test_source_fps_falls_back_to_legacy_attrs():
    assert nd._source_fps(_Decoder(get_fps=lambda: 25.0)) == 25.0


def test_source_fps_never_returns_zero():
    """Regression: a 0 fps crashed the vLLM NVDEC serve path with
    ZeroDivisionError in Qwen3-VL's _calculate_timestamps."""
    # Nothing usable: no metadata, no legacy attrs, and a zero-valued one.
    for dec in (_Decoder(), _Decoder(get_fps=lambda: 0), _Decoder(meta=_Meta())):
        fps = nd._source_fps(dec)
        assert fps > 0, f"fps must never be 0 (got {fps})"
        assert fps == nd.FALLBACK_FPS


def test_decode_metadata_fps_is_nonzero(monkeypatch):
    """End of the chain: the dict handed to vLLM must carry a usable fps."""
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=10))
    _, metadata = nd.decode_video_nvdec(b"x", num_frames=4)
    assert metadata["fps"] > 0
