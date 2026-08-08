# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import numpy as np
import pytest

import dynamo.common.multimodal.video_loader as video_loader_module
from dynamo.common.http.url_validator import UrlValidationPolicy
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

    with pytest.raises(ValueError, match="not allowed"):
        await loader.load_video("http://example.com/x.mp4")


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
async def test_decode_video_bytes_routes_h264_to_nvdec(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    meta = {"fps": 30.0, "frames_indices": [0, 1], "total_num_frames": 2}
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "h264")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)
    monkeypatch.setattr(
        video_loader_module, "decode_video_nvdec", lambda b, n: (frames, meta)
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

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader._decode_video_bytes(b"vp9-bytes", _ImportErrorMediaIO())

    msg = str(exc_info.value)
    assert "'vp9'" in msg  # names the codec
    assert VALIDATED_SPECS["opencv-python-headless"] in msg  # bounded spec
    assert "install_media_decoders vllm" in msg  # installer command
    assert "cv2" in msg


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
