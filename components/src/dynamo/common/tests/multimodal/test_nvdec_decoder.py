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


def _mp4_box(
    kind: bytes, payload: bytes = b"", *, extended: bool = False, to_end: bool = False
) -> bytes:
    if extended:
        return b"\0\0\0\1" + kind + (len(payload) + 16).to_bytes(8, "big") + payload
    size = 0 if to_end else len(payload) + 8
    return size.to_bytes(4, "big") + kind + payload


def _mp4_track(
    codec: bytes,
    handler: bytes = b"vide",
    *,
    entry_count: int = 1,
    entry_payload: bytes = bytes(78),
) -> bytes:
    entry = _mp4_box(codec, entry_payload)
    stsd = _mp4_box(b"stsd", bytes(4) + entry_count.to_bytes(4, "big") + entry)
    mdia = _mp4_box(b"hdlr", bytes(8) + handler + bytes(12))
    mdia += _mp4_box(b"minf", _mp4_box(b"stbl", stsd))
    return _mp4_box(b"trak", _mp4_box(b"mdia", mdia))


@pytest.mark.parametrize(
    "fourcc,expected",
    [
        (b"avc1", "h264"),
        (b"avc3", "h264"),
        (b"hev1", "hevc"),
        (b"hvc1", "hevc"),
        (b"av01", "av1"),
        (b"vp09", "vp9"),
        (b"vp08", "vp8"),
    ],
)
def test_probe_mp4_video_track(fourcc, expected):
    moov = _mp4_box(b"moov", _mp4_track(fourcc))
    # Conflicting markers in media data must not override the sample entry.
    mdat = _mp4_box(b"mdat", b"av01 hev1 avc1 V_AV1")
    data = _mp4_box(b"ftyp", b"isom\0\0\0\0isomavc1")
    data += moov + mdat
    assert nd.probe_video_codec(data) == expected


def test_probe_mp4_moov_after_mdat():
    # A non-faststart clip has its track metadata after the media payload.
    fake_moov = _mp4_box(b"moov", _mp4_track(b"av01"))
    data = _mp4_box(b"mdat", fake_moov, extended=True)
    data += _mp4_box(b"moov", _mp4_track(b"avc1"))
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize("location", ["title", "padding"])
def test_probe_mp4_ignores_non_codec_markers(location):
    text = b"ordinary av01 demo"
    if location == "padding":
        data = _mp4_box(b"free", text) + _mp4_box(b"moov", _mp4_track(b"avc1"))
    else:
        value = _mp4_box(b"data", b"\0\0\0\1" + bytes(4) + text)
        ilst = _mp4_box(b"ilst", _mp4_box(b"\xa9nam", value))
        metadata = _mp4_box(b"udta", _mp4_box(b"meta", bytes(4) + ilst))
        data = _mp4_box(b"moov", metadata + _mp4_track(b"avc1"))
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize(
    "entry_count,entry_payload",
    [
        (0, bytes(78)),
        (2, bytes(78)),
        (1, b"av01"),
    ],
)
def test_probe_mp4_rejects_invalid_sample_entries(entry_count, entry_payload):
    track = _mp4_track(b"avc1", entry_count=entry_count, entry_payload=entry_payload)
    assert nd.probe_video_codec(_mp4_box(b"moov", track)) is None


@pytest.mark.parametrize("extended,to_end", [(True, False), (False, True)])
def test_probe_mp4_box_size_variants(extended, to_end):
    data = _mp4_box(b"moov", _mp4_track(b"avc1"), extended=extended, to_end=to_end)
    assert nd.probe_video_codec(data) == "h264"


def test_probe_mp4_ignores_audio_tracks_and_nested_metadata():
    fake_video = _mp4_track(b"av01")
    tracks = _mp4_box(b"udta", fake_video) + _mp4_track(b"av01", b"soun")
    tracks += _mp4_track(b"avc1")
    data = _mp4_box(b"moov", tracks)
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (b"vp09", b"avc1", "h264"),
        (b"avc1", b"vp09", "h264"),
        (b"avc1", b"av01", "av1"),
        (b"avc1", b"hvc1", "hevc"),
        (b"xxxx", b"avc1", "h264"),
    ],
)
def test_probe_mp4_preserves_multi_track_routing(first, second, expected):
    data = _mp4_box(b"moov", _mp4_track(first) + _mp4_track(second))
    assert nd.probe_video_codec(data) == expected


def _ebml_element(
    kind: int, payload: bytes = b"", *, unknown_size: bool = False
) -> bytes:
    element_id = kind.to_bytes((kind.bit_length() + 7) // 8, "big")
    if unknown_size:
        return element_id + b"\xff" + payload
    width = next(n for n in range(1, 9) if len(payload) < (1 << (7 * n)) - 1)
    size = ((1 << (7 * width)) | len(payload)).to_bytes(width, "big")
    return element_id + size + payload


def _matroska_track(codec: bytes, track_type: int = 1, extra: bytes = b"") -> bytes:
    return _ebml_element(
        0xAE,
        _ebml_element(0x83, bytes([track_type])) + _ebml_element(0x86, codec) + extra,
    )


def _matroska(tracks: bytes, *, unknown_size: bool = False) -> bytes:
    return _ebml_element(
        0x1A45DFA3, _ebml_element(0x4282, b"matroska")
    ) + _ebml_element(
        0x18538067, _ebml_element(0x1654AE6B, tracks), unknown_size=unknown_size
    )


@pytest.mark.parametrize(
    "codec,expected",
    [
        (b"V_MPEG4/ISO/AVC", "h264"),
        (b"V_MPEGH/ISO/HEVC", "hevc"),
        (b"V_AV1", "av1"),
        (b"V_VP9", "vp9"),
        (b"V_VP8", "vp8"),
    ],
)
def test_probe_matroska_video_track(codec, expected):
    extra = _ebml_element(0x536E, b"title av01 hev1 V_AV1") + _ebml_element(
        0x63A2, b"V_AV1 avc1"
    )
    data = _matroska(_matroska_track(codec, extra=extra))
    assert nd.probe_video_codec(data) == expected


@pytest.mark.parametrize(
    "codec_id,expected",
    [
        (b"V_MPEG4/ISO/AVC" + bytes(18), "h264"),
        (b"V_MPEGH/ISO/HEVC" + bytes(1024), "hevc"),
    ],
)
def test_probe_matroska_null_padded_codec_id(codec_id, expected):
    assert nd.probe_video_codec(_matroska(_matroska_track(codec_id))) == expected


@pytest.mark.parametrize(
    "codec_id,expected",
    [
        (b"V_AV1\0ignored", "av1"),
        (b"V_MPEGH/ISO/HEVC" + b"X" * 1024, None),
    ],
)
def test_probe_matroska_codec_id_termination(codec_id, expected):
    assert nd.probe_video_codec(_matroska(_matroska_track(codec_id))) == expected


@pytest.mark.parametrize("size_bytes", [b"\xff", b"\x01" + b"\xff" * 7])
def test_probe_matroska_unknown_segment_size(size_bytes):
    tracks = _ebml_element(0x1654AE6B, _matroska_track(b"V_MPEG4/ISO/AVC"))
    data = _ebml_element(0x1A45DFA3) + b"\x18\x53\x80\x67" + size_bytes + tracks
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize("cluster_unknown_size", [False, True])
def test_probe_matroska_tracks_after_cluster(cluster_unknown_size):
    # Track metadata can follow Cluster. Marker-looking bytes inside a block
    # still belong to the compressed payload, not to the container hierarchy.
    block = _ebml_element(0xA3, _ebml_element(0x1654AE6B, _matroska_track(b"V_AV1")))
    cluster = _ebml_element(0x1F43B675, block, unknown_size=cluster_unknown_size)
    tracks = _ebml_element(0x1654AE6B, _matroska_track(b"V_MPEG4/ISO/AVC"))

    # Late Tracks must be indexed by a preceding SeekHead. Fixed-width seek
    # offsets keep its length stable while computing the Tracks position.
    def seek_head(position):
        seek = _ebml_element(0x53AB, b"\x16\x54\xae\x6b")
        seek += _ebml_element(0x53AC, position.to_bytes(8, "big"))
        return _ebml_element(0x114D9B74, _ebml_element(0x4DBB, seek))

    seek = seek_head(len(seek_head(0)) + len(cluster))
    segment = _ebml_element(0x18538067, seek + cluster + tracks, unknown_size=True)
    assert nd.probe_video_codec(_ebml_element(0x1A45DFA3) + segment) == "h264"


def test_probe_matroska_ignores_non_video_tracks_and_fake_nested_entries():
    tracks = _ebml_element(0xEC, _matroska_track(b"V_AV1"))
    tracks += _matroska_track(b"V_AV1", track_type=2) + _matroska_track(
        b"V_MPEG4/ISO/AVC"
    )
    assert nd.probe_video_codec(_matroska(tracks)) == "h264"


@pytest.mark.parametrize("first", [b"V_VP9", b"V_UNKNOWN"])
def test_probe_matroska_considers_later_video_tracks(first):
    data = _matroska(_matroska_track(first) + _matroska_track(b"V_MPEG4/ISO/AVC"))
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"ordinary title av01",
        b"....avc1....",
        b"\0\0\0\7moovavc1",
        b"\0\0\0\xffmoovavc1",
        b"\0\0\0\1moov",
        _mp4_box(b"mdat", b"avc1"),
        _mp4_box(
            b"moov", _mp4_box(b"trak", _mp4_box(b"mdia", _mp4_box(b"hdlr", b"vide")))
        ),
        _ebml_element(0x1A45DFA3) + b"\x18\x53",
        _ebml_element(0x1A45DFA3) + b"\0avc1",
        _ebml_element(0x1A45DFA3) + b"\x18\x53\x80\x67\x90avc1",
        b"".join(
            (
                _ebml_element(0x1A45DFA3),
                _ebml_element(
                    0x18538067,
                    _ebml_element(0x1F43B675, b"V_AV1", unknown_size=True),
                    unknown_size=True,
                ),
            )
        ),
    ],
)
def test_probe_malformed_or_unrecognized_container(data):
    assert nd.probe_video_codec(data) is None


@pytest.mark.parametrize(
    "data",
    [
        _mp4_box(b"moov", _mp4_track(b"avc1")),
        _matroska(_matroska_track(b"V_MPEG4/ISO/AVC")),
    ],
)
def test_probe_truncated_container_does_not_raise(data):
    for end in range(len(data)):
        assert nd.probe_video_codec(data[:end]) is None


@pytest.mark.parametrize("trailer", [b"\0", b"\0\0\0\x20freeav01"])
def test_probe_mp4_preserves_codec_with_invalid_trailer(trailer):
    data = _mp4_box(b"moov", _mp4_track(b"avc1")) + trailer
    assert nd.probe_video_codec(data) == "h264"


@pytest.mark.parametrize("unknown_size", [False, True])
def test_probe_matroska_preserves_codec_with_invalid_trailer(unknown_size):
    tracks = _ebml_element(0x1654AE6B, _matroska_track(b"V_MPEG4/ISO/AVC"))
    segment = _ebml_element(0x18538067, tracks + b"\0V_AV1", unknown_size=unknown_size)
    data = _ebml_element(0x1A45DFA3) + segment + b"\0"
    assert nd.probe_video_codec(data) == "h264"


def test_probe_mp4_rejects_invalid_trailer_inside_moov():
    data = _mp4_box(b"moov", _mp4_track(b"avc1") + b"\0")
    assert nd.probe_video_codec(data) is None


def test_probe_matroska_rejects_invalid_trailer_inside_tracks():
    data = _matroska(_matroska_track(b"V_MPEG4/ISO/AVC") + b"\0")
    assert nd.probe_video_codec(data) is None


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


@pytest.mark.parametrize(
    "num_frames,fps,expected",
    [
        (10, -1, 10),  # no fps: the frame cap alone decides, as before
        (32, 1, 3),  # floor(3s * 1fps), well under the cap
        (2, 10, 2),  # whichever cap is lower wins
        (32, 0.1, 1),  # a sub-frame rate still yields a decodable clip
    ],
)
def test_decode_applies_the_fps_cap_against_clip_duration(
    monkeypatch, num_frames, fps, expected
):
    """`fps` caps the count at floor(duration * fps).

    Resolved here rather than by the caller because duration is only knowable
    once the container is open -- 90 frames at the decoder's 30 fps is 3s.
    """
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", _fake_pynv(num_frames=90))
    monkeypatch.setattr(
        nd, "_frame_to_rgb_hwc", lambda f: np.asarray(f, dtype=np.uint8)
    )
    frames, meta = nd.decode_video_nvdec(b"x", num_frames=num_frames, fps=fps)

    assert frames.shape[0] == expected
    assert meta["fps"] == 30.0  # metadata reports the source rate, not the target


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
    """PyNvVideoCodec reports the rate via get_stream_metadata()."""
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
