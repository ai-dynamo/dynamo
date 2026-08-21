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

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_audio_request_wire_shape_and_defaults():
    request = NvCreateAudioSpeechRequest(
        input="hello",
        model="qwen-tts",
        voice="vivian",
        data_source="b64_json",
        task_type="CustomVoice",
        language="English",
    )

    assert request.model_dump(exclude_none=True) == {
        "input": "hello",
        "model": "qwen-tts",
        "voice": "vivian",
        "data_source": "b64_json",
        "response_format": "wav",
        "speed": 1.0,
        "task_type": "CustomVoice",
        "language": "English",
    }


@pytest.mark.parametrize("speed", [0.1, 5.0])
def test_audio_request_rejects_speed_outside_supported_range(speed):
    with pytest.raises(ValidationError):
        NvCreateAudioSpeechRequest(input="hi", speed=speed)


def test_audio_response_wire_shape():
    response = NvAudioSpeechResponse(
        id="r1",
        model="qwen-tts",
        created=0,
        data=[AudioData(output_format="mp3", b64_json="xyz")],
    )

    assert response.model_dump(exclude_none=True) == {
        "id": "r1",
        "object": "audio.speech",
        "model": "qwen-tts",
        "status": "completed",
        "progress": 100,
        "created": 0,
        "data": [{"output_format": "mp3", "b64_json": "xyz"}],
    }
