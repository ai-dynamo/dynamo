# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.audio_protocol module."""

import pytest
from pydantic import ValidationError

from dynamo.common.protocols.audio_protocol import (
    AudioData,
    NvAudioSpeechResponse,
    NvCreateAudioSpeechRequest,
)

# ---------------------------------------------------------------------------
# NvCreateAudioSpeechRequest
# ---------------------------------------------------------------------------


class TestNvCreateAudioSpeechRequest:
    def test_minimal_required_fields(self):
        req = NvCreateAudioSpeechRequest(input="hello")
        assert req.input == "hello"
        assert req.data_source is None
        assert req.response_format == "wav"
        assert req.speed == 1.0

    def test_data_source_url(self):
        req = NvCreateAudioSpeechRequest(input="hi", data_source="url")
        assert req.data_source == "url"

    def test_data_source_b64_json(self):
        req = NvCreateAudioSpeechRequest(input="hi", data_source="b64_json")
        assert req.data_source == "b64_json"

    def test_data_source_and_response_format_coexist(self):
        req = NvCreateAudioSpeechRequest(
            input="hi", data_source="b64_json", response_format="mp3"
        )
        assert req.data_source == "b64_json"
        assert req.response_format == "mp3"

    def test_response_format_valid_values(self):
        for fmt in ("wav", "pcm", "flac", "mp3", "aac", "opus"):
            req = NvCreateAudioSpeechRequest(input="hi", response_format=fmt)
            assert req.response_format == fmt

    def test_response_format_arbitrary_string(self):
        req = NvCreateAudioSpeechRequest(input="hi", response_format="webm")
        assert req.response_format == "webm"

    def test_speed_bounds(self):
        with pytest.raises(ValidationError):
            NvCreateAudioSpeechRequest(input="hi", speed=0.1)
        with pytest.raises(ValidationError):
            NvCreateAudioSpeechRequest(input="hi", speed=5.0)


# ---------------------------------------------------------------------------
# AudioData
# ---------------------------------------------------------------------------


class TestAudioData:
    def test_output_format_required(self):
        with pytest.raises(ValidationError):
            AudioData()  # missing output_format

    def test_output_format_missing_raises(self):
        with pytest.raises(ValidationError):
            AudioData(url="http://example.com/audio.wav")

    def test_minimal_with_output_format(self):
        d = AudioData(output_format="wav")
        assert d.output_format == "wav"
        assert d.url is None
        assert d.b64_json is None

    def test_with_url(self):
        d = AudioData(output_format="mp3", url="http://example.com/audio.mp3")
        assert d.url == "http://example.com/audio.mp3"
        assert d.output_format == "mp3"

    def test_with_b64_json(self):
        d = AudioData(output_format="opus", b64_json="abc123==")
        assert d.b64_json == "abc123=="
        assert d.output_format == "opus"

    def test_json_round_trip(self):
        d = AudioData(output_format="flac", url="http://example.com/a.flac")
        restored = AudioData.model_validate_json(d.model_dump_json())
        assert restored.output_format == "flac"
        assert restored.url == d.url

    def test_all_codec_values_accepted(self):
        for fmt in ("wav", "mp3", "pcm", "flac", "aac", "opus"):
            d = AudioData(output_format=fmt)
            assert d.output_format == fmt

    def test_output_format_arbitrary_string(self):
        d = AudioData(output_format="webm")
        assert d.output_format == "webm"


# ---------------------------------------------------------------------------
# NvAudioSpeechResponse
# ---------------------------------------------------------------------------


class TestNvAudioSpeechResponse:
    def test_data_list_carries_output_format(self):
        resp = NvAudioSpeechResponse(
            id="r1",
            model="qwen-tts",
            created=0,
            data=[AudioData(output_format="mp3", b64_json="xyz")],
        )
        assert resp.data[0].output_format == "mp3"

    def test_empty_data_list_default(self):
        resp = NvAudioSpeechResponse(id="r2", model="m", created=0)
        assert resp.data == []
