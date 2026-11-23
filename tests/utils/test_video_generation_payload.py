# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import binascii
from contextlib import closing
from unittest.mock import MagicMock

import pytest

from tests.utils import payloads
from tests.utils.payloads import I2VPayload, VideoGenerationPayload

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.timeout(30),
]


@pytest.fixture(scope="module")
def mp4_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    with closing(
        imageio_ffmpeg.write_frames(
            str(path), (32, 24), fps=24, codec="libvpx-vp9", macro_block_size=1
        )
    ) as frames:
        next(frames)
        for index in range(3):
            frames.send(bytes([40 + index * 20]) * (32 * 24 * 3))
    return path.read_bytes()


def _response(entry):
    response = MagicMock()
    response.json.return_value = {"status": "completed", "data": [entry]}
    return response


def _payload(payload_type=VideoGenerationPayload, **expectations):
    return payload_type(body={}, expected_response=[], expected_log=[], **expectations)


@pytest.mark.parametrize("payload_type", [VideoGenerationPayload, I2VPayload])
def test_valid_video_is_decoded(payload_type, mp4_video):
    payload = _payload(
        payload_type, expected_video_shape=(3, 24, 32), expected_video_fps=24
    )
    assert (
        payload.response_handler(
            _response({"b64_json": base64.b64encode(mp4_video).decode()})
        )
        == "b64_video_returned"
    )


@pytest.mark.parametrize(
    "expectations, message",
    [
        ({"expected_video_shape": (4, 24, 32)}, "Expected 4 frames, got 3"),
        ({"expected_video_shape": (3, 16, 32)}, "Expected 32x16 video, got 32x24"),
        ({"expected_video_shape": (3, 24, 16)}, "Expected 16x24 video, got 32x24"),
        ({"expected_video_fps": 30}, "Expected 30 FPS, got 24"),
    ],
)
def test_video_must_meet_each_expectation(mp4_video, expectations, message):
    with pytest.raises(AssertionError, match=message):
        _payload(**expectations).response_handler(
            _response({"b64_json": base64.b64encode(mp4_video).decode()})
        )


def test_corrupt_media_does_not_pass_as_a_completed_video(mp4_video):
    with pytest.raises(AssertionError, match="Could not decode video"):
        _payload(expected_video_shape=(3, 24, 32)).response_handler(
            _response({"b64_json": base64.b64encode(mp4_video[:32]).decode()})
        )


def test_corrupt_frames_with_valid_container_metadata_are_rejected(mp4_video):
    # Keep the MP4 stream metadata intact but destroy the encoded frames.
    # FFprobe still returns success for this clip, so its exit code alone is
    # insufficient to prove successful decoding.
    corrupt = bytearray(mp4_video)
    mdat = corrupt.index(b"mdat")
    atom_size = int.from_bytes(corrupt[mdat - 4 : mdat], "big")
    corrupt[mdat + 4 : mdat - 4 + atom_size] = bytes(atom_size - 8)
    with pytest.raises(AssertionError, match="Could not decode video"):
        _payload(expected_video_shape=(3, 24, 32)).response_handler(
            _response({"b64_json": base64.b64encode(corrupt).decode()})
        )


def test_non_mp4_media_is_rejected():
    with pytest.raises(AssertionError, match="Expected an MP4 container"):
        _payload(expected_video_fps=24).response_handler(
            _response({"b64_json": base64.b64encode(b"not a video").decode()})
        )


def test_invalid_base64_is_rejected():
    with pytest.raises(binascii.Error):
        _payload(expected_video_fps=24).response_handler(
            _response({"b64_json": "invalid!!!"})
        )


def test_url_cannot_bypass_decode_checks():
    with pytest.raises(AssertionError, match="require b64_json"):
        _payload(expected_video_fps=24).response_handler(
            _response({"url": "file:///unused.mp4"})
        )


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"url": "file:///unused.mp4"}, "file:///unused.mp4"),
        ({"b64_json": "nonempty"}, "b64_video_returned"),
    ],
)
def test_unconfigured_payload_does_not_require_a_decoder(monkeypatch, entry, expected):
    monkeypatch.setattr(payloads.shutil, "which", lambda _: None)
    assert _payload().response_handler(_response(entry)) == expected


def test_decode_checks_require_ffprobe(monkeypatch, mp4_video):
    monkeypatch.setattr(payloads.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffprobe is required"):
        _payload(expected_video_fps=24).response_handler(
            _response({"b64_json": base64.b64encode(mp4_video).decode()})
        )
