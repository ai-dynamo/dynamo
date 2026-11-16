# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AudioSpeechPayload``'s waveform expectations.

A byte-count check alone passes on a WAV carrying a header and a fraction of a
second of silence, which is exactly what a mis-assembled cumulative chunk
stream produces. These tests drive ``response_handler`` over synthesized WAVs
so the duration, RMS, and sample-rate assertions are exercised without a GPU or
a served model — the serve test that configures them is skipped on CI capacity.
"""

import base64
import math
import struct
import wave
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from tests.utils.payloads import AudioSpeechPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _wav_bytes(
    *,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.2,
    lead_amplitude: Optional[float] = None,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """A PCM WAV holding one constant sample value, so its RMS == ``amplitude``.

    ``lead_amplitude`` overrides the first frame only. That makes the opening
    frame unrepresentative of the waveform, which is what the whole-waveform
    RMS checks need to distinguish a real measurement from a peek at the head.
    """
    frame_count = int(duration_s * sample_rate)
    buffer = BytesIO()

    def _frame(level: float) -> bytes:
        if sample_width == 2:
            return struct.pack(f"<{channels}h", *([int(level * 32767)] * channels))
        return bytes([128] * channels * sample_width)

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        if lead_amplitude is not None and frame_count:
            wav.writeframes(_frame(lead_amplitude))
            frame_count -= 1
        wav.writeframes(_frame(amplitude) * frame_count)
    return buffer.getvalue()


def _payload(
    *,
    min_duration_s: float = 0.0,
    min_rms: float = 0.0,
    expected_sample_rate: Optional[int] = None,
) -> AudioSpeechPayload:
    return AudioSpeechPayload(
        body={"input": "hi"},
        expected_response=[],
        expected_log=[],
        min_duration_s=min_duration_s,
        min_rms=min_rms,
        expected_sample_rate=expected_sample_rate,
    )


def _binary_response(audio_bytes: bytes) -> Any:
    """A response delivering the WAV as ``audio/wav`` bytes."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.headers = {"content-type": "audio/wav"}
    response.content = audio_bytes
    return response


def _b64_response(audio_bytes: bytes) -> Any:
    """A response delivering the WAV as base64 in the JSON envelope."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.headers = {"content-type": "application/json"}
    response.json.return_value = {
        "status": "completed",
        "data": [
            {
                "output_format": "wav",
                "b64_json": base64.b64encode(audio_bytes).decode(),
            }
        ],
    }
    return response


@pytest.mark.parametrize(
    "make_response", [_binary_response, _b64_response], ids=["binary", "b64"]
)
def test_waveform_meeting_every_expectation_passes(make_response):
    """Both delivery shapes run the same checks."""
    payload = _payload(min_duration_s=1.0, min_rms=0.01, expected_sample_rate=16000)

    payload.response_handler(make_response(_wav_bytes()))


@pytest.mark.parametrize(
    "make_response", [_binary_response, _b64_response], ids=["binary", "b64"]
)
def test_truncated_waveform_fails(make_response):
    """A mis-assembled chunk stream yields a short but parseable WAV."""
    payload = _payload(min_duration_s=1.0)
    audio = _wav_bytes(duration_s=0.2)

    with pytest.raises(AssertionError, match="shorter than the expected minimum"):
        payload.response_handler(make_response(audio))


@pytest.mark.parametrize(
    "make_response", [_binary_response, _b64_response], ids=["binary", "b64"]
)
def test_silent_waveform_fails(make_response):
    """A header plus silence is the other broken-decoder signature."""
    payload = _payload(min_rms=0.01)

    with pytest.raises(AssertionError, match="is below the expected minimum"):
        payload.response_handler(make_response(_wav_bytes(amplitude=0.0)))


def test_wrong_sample_rate_fails():
    """Audex decodes at 16 kHz; another rate means the wrong decode path."""
    payload = _payload(expected_sample_rate=16000)

    with pytest.raises(AssertionError, match="Expected 16000 Hz audio, got 24000 Hz"):
        payload.response_handler(_binary_response(_wav_bytes(sample_rate=24000)))


def test_rms_is_compared_against_the_configured_floor():
    """The floor has to be applied: above it passes, below it fails."""
    payload = _payload(min_rms=0.2)

    payload.response_handler(_binary_response(_wav_bytes(amplitude=0.25)))
    with pytest.raises(AssertionError, match=r"Audio RMS \S+ is below the expected"):
        payload.response_handler(_binary_response(_wav_bytes(amplitude=0.15)))


def test_a_loud_first_frame_does_not_pass_a_silent_waveform():
    """RMS over the whole signal, not a peek at the head of the buffer."""
    payload = _payload(min_rms=0.01)
    audio = _wav_bytes(amplitude=0.0, lead_amplitude=0.5)

    with pytest.raises(AssertionError, match=r"Audio RMS \S+ is below the expected"):
        payload.response_handler(_binary_response(audio))


def test_a_silent_first_frame_does_not_fail_a_loud_waveform():
    """The converse: a quiet opening frame is not the whole signal either."""
    payload = _payload(min_rms=0.2)

    payload.response_handler(
        _binary_response(_wav_bytes(amplitude=0.25, lead_amplitude=0.0))
    )


def test_stereo_waveform_is_accepted():
    """Duration is frame-based, so channel count must not scale it."""
    payload = _payload(min_duration_s=1.0, min_rms=0.01, expected_sample_rate=16000)

    payload.response_handler(_binary_response(_wav_bytes(channels=2)))


def test_non_16_bit_audio_with_min_rms_is_reported_not_mismeasured():
    """The RMS unpack assumes 16-bit PCM, so anything else must fail loudly."""
    payload = _payload(min_rms=0.01)

    with pytest.raises(AssertionError, match="RMS check supports 16-bit PCM only"):
        payload.response_handler(_binary_response(_wav_bytes(sample_width=1)))


def test_all_checks_disabled_skips_waveform_decoding():
    """The default payload must not require a decodable WAV."""
    payload = _payload()

    assert payload.response_handler(_binary_response(b"x" * 200)).startswith(
        "binary_audio_"
    )


def test_url_response_skips_the_waveform_checks():
    """A URL response carries no bytes to decode."""
    payload = _payload(min_duration_s=1.0, min_rms=0.01)
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.headers = {"content-type": "application/json"}
    response.json.return_value = {
        "status": "completed",
        "data": [{"url": "http://media/audios/req-1.wav"}],
    }

    assert payload.response_handler(response) == "http://media/audios/req-1.wav"


def test_helper_rms_matches_its_amplitude():
    """Guards the fixture the RMS assertions above are calibrated against."""
    audio = _wav_bytes(amplitude=0.25, duration_s=0.5)
    with wave.open(BytesIO(audio), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0

    assert rms == pytest.approx(0.25, abs=0.001)
