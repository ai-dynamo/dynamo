# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the actionable unsupported-codec error builders."""

import pytest

import dynamo.common.multimodal.codec_errors as codec_errors
from dynamo.common.multimodal.codec_errors import (
    MissingMediaDecoderError,
    audio_decoder_missing,
    video_decoder_missing,
)
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_video_message_names_codec_spec_and_installer(monkeypatch):
    monkeypatch.setattr(codec_errors, "nvdec_available", lambda: True)
    err = video_decoder_missing("vllm", "opencv-python-headless", "cv2", "vp9")

    assert isinstance(err, MissingMediaDecoderError)
    msg = str(err)
    assert "'vp9'" in msg
    assert "cv2" in msg
    # The bounded spec comes verbatim from the installer's constants, so the
    # message and the documented install can never drift apart.
    assert VALIDATED_SPECS["opencv-python-headless"] in msg
    assert "install_media_decoders vllm" in msg
    # Non-hardware codec: the hardware alternative is re-encoding.
    assert "H.264/H.265" in msg


def test_hw_codec_without_nvdec_points_at_driver_capability(monkeypatch):
    """H.264 with NVDEC unavailable is a capability problem first.

    The primary remedy is granting the container the 'video' driver
    capability, not installing a software decoder -- the message must lead
    with that.
    """
    monkeypatch.setattr(codec_errors, "nvdec_available", lambda: False)
    err = video_decoder_missing("vllm", "opencv-python-headless", "cv2", "h264")

    msg = str(err)
    assert "NVDEC is unavailable" in msg
    assert "NVIDIA_DRIVER_CAPABILITIES" in msg
    assert VALIDATED_SPECS["opencv-python-headless"] in msg


def test_hw_codec_with_nvdec_available_uses_generic_wording(monkeypatch):
    """h264 + NVDEC available but decode still fell through to software:
    the capability lead would be wrong, so the generic wording applies."""
    monkeypatch.setattr(codec_errors, "nvdec_available", lambda: True)
    err = video_decoder_missing("vllm", "opencv-python-headless", "cv2", "h264")

    assert "NVDEC is unavailable" not in str(err)


def test_unknown_codec_still_actionable(monkeypatch):
    monkeypatch.setattr(codec_errors, "nvdec_available", lambda: True)
    err = video_decoder_missing("sglang", "decord2", "decord", None)

    msg = str(err)
    assert "an undetected codec" in msg
    assert VALIDATED_SPECS["decord2"] in msg
    assert "install_media_decoders sglang" in msg


def test_audio_message_has_no_hardware_alternative():
    err = audio_decoder_missing("vllm")

    msg = str(err)
    assert isinstance(err, MissingMediaDecoderError)
    assert VALIDATED_SPECS["av"] in msg
    assert "NVDEC does not decode audio" in msg
    assert "install_media_decoders vllm" in msg


def test_cause_text_is_appended_and_optional(monkeypatch):
    """The underlying decoder text must survive wraps whose handlers ship
    only str(exc) to the client; absent a cause, no dangling suffix."""
    monkeypatch.setattr(codec_errors, "nvdec_available", lambda: True)
    with_cause = video_decoder_missing(
        "vllm", "opencv-python-headless", "cv2", "vp9", cause="No module named 'cv2'"
    )
    assert "(decoder reported: No module named 'cv2')" in str(with_cause)
    without = video_decoder_missing("vllm", "opencv-python-headless", "cv2", "vp9")
    assert "decoder reported" not in str(without)
    audio = audio_decoder_missing("vllm", cause="Please install vllm[audio]")
    assert "(decoder reported: Please install vllm[audio])" in str(audio)


def test_error_is_not_a_value_error():
    """Handlers map ValueError to client 4xx; a missing decoder is deployment
    configuration and must not be blamed on the request."""
    assert not issubclass(MissingMediaDecoderError, ValueError)
    assert issubclass(MissingMediaDecoderError, RuntimeError)


def test_every_referenced_package_has_a_validated_spec():
    """The builders promise a bounded spec for these packages; keep the
    installer's table covering them."""
    for package in ("opencv-python-headless", "decord2", "av"):
        spec = VALIDATED_SPECS[package]
        assert ">=" in spec and ",<" in spec
