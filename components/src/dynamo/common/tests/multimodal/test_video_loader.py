# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import AsyncMock

import numpy as np
import pytest

import dynamo.common.multimodal.codec_errors as codec_errors
import dynamo.common.multimodal.video_loader as video_loader_module
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
async def test_load_video_rejects_http_by_default():
    """Wiring smoke: VideoLoader plumbs ``url_policy`` to the validator.

    Validator behavior is covered in ``test_url_validator.py``;
    per-hop SSRF revalidation in ``http/test_http_backends.py``.
    """
    loader = VideoLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(UrlValidationError, match="not allowed"):
        await loader.load_video("http://example.com/x.mp4")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.mp4"),
    ],
)
async def test_load_video_preserves_client_error(client_error):
    loader = VideoLoader()
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=client_error
    )

    with pytest.raises(type(client_error)) as exc_info:
        await loader.load_video("https://example.com/x.mp4")

    assert exc_info.value is client_error


@pytest.mark.asyncio
async def test_load_video_uses_vllm_media_connector():
    loader = VideoLoader()
    # data: scheme is in the default allowlist regardless of env flags.
    loader._url_policy = UrlValidationPolicy()
    frames = np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3)[:, :, ::-1, :]
    metadata = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        return_value=(frames, metadata)
    )

    loaded_frames, loaded_metadata = await loader.load_video(
        "data:video/webm;base64,Zm9v"
    )

    assert loaded_frames.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(loaded_frames, np.ascontiguousarray(frames))
    assert loaded_metadata == metadata


@pytest.mark.asyncio
async def test_load_video_batch_uses_url_loader():
    loader = VideoLoader()
    first = (
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    second = (
        np.ones((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    loader.load_video = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    videos = await loader.load_video_batch(
        [
            {"Url": "https://example.com/one.mp4"},
            {"Url": "https://example.com/two.mp4"},
        ]
    )

    np.testing.assert_array_equal(videos[0][0], first[0])
    np.testing.assert_array_equal(videos[1][0], second[0])
    assert videos[0][1] == first[1]
    assert videos[1][1] == second[1]


@pytest.mark.asyncio
async def test_load_video_batch_rejects_decoded_variant_without_frontend_decoding():
    loader = VideoLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="enable_frontend_decoding=False"):
        await loader.load_video_batch([{"Decoded": {"shape": [1, 2, 2, 3]}}])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.mp4"),
    ],
)
async def test_load_video_batch_prioritizes_typed_client_error(client_error):
    loader = VideoLoader()
    loader.load_video = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("decode failed"), client_error]
    )

    with pytest.raises(type(client_error)) as exc_info:
        await loader.load_video_batch(
            [
                {"Url": "https://example.com/bad.mp4"},
                {"Url": "https://example.com/unsupported.mp4"},
            ]
        )

    assert exc_info.value is client_error


@pytest.mark.asyncio
async def test_load_video_batch_preserves_value_error():
    loader = VideoLoader()
    client_error = ValueError("invalid num_frames")
    loader.load_video = AsyncMock(side_effect=client_error)  # type: ignore[method-assign]

    with pytest.raises(ValueError) as exc_info:
        await loader.load_video_batch([{"Url": "https://example.com/bad.mp4"}])

    assert exc_info.value is client_error


@pytest.mark.asyncio
async def test_load_video_batch_reads_decoded_variant_with_metadata(monkeypatch):
    loader = VideoLoader(enable_frontend_decoding=False)
    loader._enable_frontend_decoding = True
    loader._nixl_connector = object()

    decoded_item = {
        "shape": [1, 2, 2, 3],
        "metadata": {"fps": 3.0, "frames_indices": [0], "total_num_frames": 1},
    }
    frames = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    read_decoded = AsyncMock(return_value=(frames, decoded_item["metadata"]))
    monkeypatch.setattr(
        video_loader_module, "read_decoded_media_via_nixl", read_decoded
    )

    videos = await loader.load_video_batch([{"Decoded": decoded_item}])

    np.testing.assert_array_equal(videos[0][0], np.ascontiguousarray(frames))
    assert videos[0][1] == decoded_item["metadata"]
    read_decoded.assert_awaited_once_with(
        loader._nixl_connector,
        decoded_item,
        return_metadata=True,
    )


@pytest.mark.asyncio
async def test_load_video_batch_normalizes_rust_frontend_metadata(monkeypatch):
    loader = VideoLoader(enable_frontend_decoding=False)
    loader._enable_frontend_decoding = True
    loader._nixl_connector = object()

    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    rust_metadata = {
        "Video": {
            "source_fps": 24.0,
            "source_duration": 10.0,
            "sampled_timestamps": [0.0, 5.0, 9.0],
        }
    }
    monkeypatch.setattr(
        video_loader_module,
        "read_decoded_media_via_nixl",
        AsyncMock(return_value=(frames, rust_metadata)),
    )

    [(loaded_frames, metadata)] = await loader.load_video_batch(
        [{"Decoded": {"shape": [3, 2, 2, 3]}}]
    )

    np.testing.assert_array_equal(loaded_frames, frames)
    assert metadata == {
        "fps": 24.0,
        "duration": 10.0,
        "frames_indices": [0, 120, 216],
        "total_num_frames": 240,
        "video_backend": "dynamo_frontend",
        "do_sample_frames": False,
    }


@pytest.mark.asyncio
async def test_decode_video_bytes_routes_h264_to_nvdec(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    meta = {"fps": 30.0, "frames_indices": [0, 1], "total_num_frames": 2}
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "h264")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)
    monkeypatch.setattr(
        video_loader_module,
        "decode_video_nvdec",
        lambda content, **kwargs: (frames, meta),
    )
    media_io = _RecordingMediaIO(frames)

    got_frames, got_meta = await loader._decode_video_bytes(b"h264-bytes", media_io)

    np.testing.assert_array_equal(got_frames, frames)
    assert got_meta == meta
    assert media_io.calls == []  # hardware decoded; software untouched


@pytest.mark.asyncio
async def test_decode_video_bytes_routes_royalty_free_to_software(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    called = {"decode": False}

    def _decode(*a, **k):
        called["decode"] = True

    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", _decode)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"vp9-bytes", media_io)

    assert result is media_io.result  # VP9 stays on the software path
    assert called["decode"] is False  # NVDEC not invoked
    assert media_io.calls == [b"vp9-bytes"]


@pytest.mark.asyncio
async def test_decode_video_bytes_falls_back_on_nvdec_failure(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "hevc")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)

    def _boom(*a, **k):
        raise RuntimeError("nvdec session limit")

    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", _boom)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"hevc-bytes", media_io)

    assert result is media_io.result  # NVDEC failure falls back, never raises
    assert media_io.calls == [b"hevc-bytes"]


class _RecordingMediaIO:
    """Stub for vLLM's VideoMediaIO: records load_bytes calls."""

    def __init__(self, frames):
        self.result = (frames, {"fps": 1.0})
        self.calls: list[bytes] = []

    def load_bytes(self, content: bytes):
        self.calls.append(content)
        return self.result


class _ImportErrorMediaIO:
    """Stub reproducing vLLM's lazy cv2 import failure on a stripped image."""

    def load_bytes(self, content: bytes):
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")


@pytest.mark.asyncio
async def test_decode_video_bytes_missing_decoder_is_actionable(monkeypatch):
    """A bare `No module named 'cv2'` must become the actionable codec error.

    Reproduced on a real runtime image before this existed: the user-visible
    text was `Failed to load video from ...: No module named 'cv2'` -- no
    codec, no remedy.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    # cv2 genuinely absent, independent of what the test machine has installed.
    monkeypatch.setattr(codec_errors.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader._decode_video_bytes(b"vp9-bytes", _ImportErrorMediaIO())

    msg = str(exc_info.value)
    assert "'vp9'" in msg  # names the codec
    assert VALIDATED_SPECS["opencv-python-headless"] in msg  # bounded spec
    assert "pip install" in msg
    assert "cv2" in msg


class _SystemErrorMediaIO:
    """What a cv2 with no video backend does: fail inside VideoCapture."""

    def load_bytes(self, content: bytes):
        raise SystemError(
            "<class 'cv2.VideoCapture'> returned a result with an exception set"
        )


@pytest.mark.asyncio
async def test_decode_video_bytes_backendless_cv2_is_actionable(monkeypatch):
    """A cv2 built without a video backend must reach the same error.

    The runtime images rebuild OpenCV from source with WITH_FFMPEG=OFF, so cv2
    imports and resizes but opens no video. vLLM raises SystemError from
    VideoCapture rather than ImportError, which would otherwise escape the
    handler and reach the client with no codec and no remedy.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "_cv2_lacks_video_backend", lambda: True)

    monkeypatch.setattr(codec_errors.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        codec_errors.importlib.metadata, "version", lambda p: "5.0.0.93"
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader._decode_video_bytes(b"vp9-bytes", _SystemErrorMediaIO())

    msg = str(exc_info.value)
    assert "'vp9'" in msg
    assert "opencv-python-headless==5.0.0.93" in msg
    assert "--force-reinstall" in msg
    assert "video backend" in msg


@pytest.mark.asyncio
async def test_backendless_cv2_does_not_pre_empt_a_working_decode(monkeypatch):
    """The check must not short-circuit a decoder that would have succeeded.

    On the shipped image the predicate is always true, so gating the call on it
    would reject every VideoMediaIO -- including a configured non-OpenCV
    video_backend, and the stubbed load_bytes the worker tests rely on.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "_cv2_lacks_video_backend", lambda: True)

    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    class _WorkingMediaIO:
        def load_bytes(self, content: bytes):
            return frames, {"frames_indices": [0, 1]}

    decoded, metadata = await loader._decode_video_bytes(
        b"vp9-bytes", _WorkingMediaIO()
    )
    assert decoded is frames
    assert metadata == {"frames_indices": [0, 1]}


@pytest.mark.asyncio
async def test_other_backend_import_error_is_not_blamed_on_opencv(monkeypatch):
    """A media_io configured for another backend keeps its own ImportError.

    vLLM selects among several video backends and PyAV is absent from the
    shipped images too, so its ImportError would otherwise be rewritten into
    advice to reinstall OpenCV -- which cannot repair a VideoMediaIO still
    configured for PyAV.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)

    class _PyAvMediaIO:
        def load_bytes(self, content: bytes):
            raise ModuleNotFoundError("No module named 'av'", name="av")

    with pytest.raises(ModuleNotFoundError, match="'av'"):
        await loader._decode_video_bytes(b"vp9-bytes", _PyAvMediaIO())


@pytest.mark.asyncio
async def test_other_backend_system_error_is_not_blamed_on_opencv(monkeypatch):
    """A different decoder's SystemError must survive on a backendless image.

    _cv2_lacks_video_backend() describes the installed cv2 build; it does not
    prove this media_io ran OpenCV. On the shipped image it is always true, so
    keying only on it would relabel every backend's SystemError as a missing
    OpenCV and bury the real failure.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "_cv2_lacks_video_backend", lambda: True)

    class _OtherBackendMediaIO:
        def load_bytes(self, content: bytes):
            raise SystemError("torchcodec decoder failed to initialise")

    with pytest.raises(SystemError, match="torchcodec"):
        await loader._decode_video_bytes(b"vp9-bytes", _OtherBackendMediaIO())


@pytest.mark.asyncio
async def test_unrelated_system_error_keeps_its_own_message(monkeypatch):
    """A SystemError from anywhere else must not be blamed on the codec."""
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "_cv2_lacks_video_backend", lambda: False)

    with pytest.raises(SystemError, match="VideoCapture"):
        await loader._decode_video_bytes(b"vp9-bytes", _SystemErrorMediaIO())


def test_cv2_lacks_video_backend_is_false_when_cv2_absent(monkeypatch):
    """An absent cv2 is not this case: the ImportError path names it better."""
    video_loader_module._cv2_lacks_video_backend.cache_clear()
    monkeypatch.setitem(sys.modules, "cv2", None)
    try:
        assert video_loader_module._cv2_lacks_video_backend() is False
    finally:
        video_loader_module._cv2_lacks_video_backend.cache_clear()


@pytest.mark.asyncio
async def test_load_video_batch_preserves_missing_decoder_error():
    """The batch aggregate wraps failures in a generic Exception; the
    missing-decoder type must survive it (review finding)."""
    loader = VideoLoader()
    err = video_loader_module.video_decoder_missing(
        "vllm", "opencv-python-headless", "cv2", "vp9"
    )
    loader.load_video = AsyncMock(side_effect=err)  # type: ignore[method-assign]

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_video_batch([{"Url": "https://example.com/x.mp4"}])

    assert exc_info.value is err


@pytest.mark.asyncio
async def test_load_video_preserves_missing_decoder_error(monkeypatch):
    """The generic ValueError wrap must not erase the decoder-missing type.

    A missing decoder is deployment configuration, not a bad request, so it
    must not degrade into the ValueError that handlers map to a client error.
    """
    loader = VideoLoader()
    err = video_loader_module.video_decoder_missing(
        "vllm", "opencv-python-headless", "cv2", "vp9"
    )
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=err
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_video("https://example.com/x.mp4")

    assert exc_info.value is err
