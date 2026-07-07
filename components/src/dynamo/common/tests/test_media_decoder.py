# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_MEDIA_DECODER_PY = Path(__file__).resolve().parents[1] / "utils" / "media_decoder.py"


def _load_media_decoder_module():
    spec = importlib.util.spec_from_file_location("media_decoder", _MEDIA_DECODER_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


media_decoder = _load_media_decoder_module()
DYN_MM_VIDEO_DECODER_BACKEND = media_decoder.DYN_MM_VIDEO_DECODER_BACKEND
DYN_MM_VIDEO_NUM_FRAMES = media_decoder.DYN_MM_VIDEO_NUM_FRAMES
build_frontend_image_decoder_options = (
    media_decoder.build_frontend_image_decoder_options
)
build_frontend_video_decoder_options = (
    media_decoder.build_frontend_video_decoder_options
)
enable_frontend_video_decoding = media_decoder.enable_frontend_video_decoding


class _FakeMediaDecoder:
    def __init__(self) -> None:
        self.image_options = None
        self.video_options = None

    def enable_image(self, options):
        self.image_options = options

    def enable_video(self, options):
        self.video_options = options


def test_frontend_decoder_options_defaults(monkeypatch):
    monkeypatch.delenv(DYN_MM_VIDEO_DECODER_BACKEND, raising=False)
    monkeypatch.delenv(DYN_MM_VIDEO_NUM_FRAMES, raising=False)

    assert build_frontend_image_decoder_options() == {
        "limits": {"max_alloc": 128 * 1024 * 1024}
    }
    assert build_frontend_video_decoder_options() == {
        "limits": {"max_alloc": 512 * 1024 * 1024},
        "num_frames": 32,
    }


@pytest.mark.parametrize("backend", ["ffmpeg", "opencv"])
def test_frontend_video_decoder_options_from_env(monkeypatch, backend):
    monkeypatch.setenv(DYN_MM_VIDEO_DECODER_BACKEND, backend)
    monkeypatch.setenv(DYN_MM_VIDEO_NUM_FRAMES, "8")

    assert build_frontend_video_decoder_options() == {
        "limits": {"max_alloc": 512 * 1024 * 1024},
        "num_frames": 8,
        "backend": backend,
    }


def test_frontend_video_decoder_options_reject_invalid_backend(monkeypatch):
    monkeypatch.setenv(DYN_MM_VIDEO_DECODER_BACKEND, "invalid")

    with pytest.raises(
        ValueError,
        match="DYN_MM_VIDEO_DECODER_BACKEND must be one of: ffmpeg, opencv, video_rs",
    ):
        build_frontend_video_decoder_options()


def test_enable_frontend_video_decoding(monkeypatch):
    monkeypatch.setenv(DYN_MM_VIDEO_DECODER_BACKEND, "opencv")
    decoder = _FakeMediaDecoder()

    enable_frontend_video_decoding(decoder)

    assert decoder.video_options == {
        "limits": {"max_alloc": 512 * 1024 * 1024},
        "num_frames": 32,
        "backend": "opencv",
    }


def test_enable_frontend_video_decoding_warns_for_binding_without_video(
    monkeypatch, caplog
):
    monkeypatch.setenv(DYN_MM_VIDEO_DECODER_BACKEND, "opencv")

    enable_frontend_video_decoding(object())

    assert DYN_MM_VIDEO_DECODER_BACKEND in caplog.text
    assert "frontend video decoding support" in caplog.text
