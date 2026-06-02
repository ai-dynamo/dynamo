# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni output modality request classification."""

import pytest

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.utils.output_modalities import RequestType, parse_request_type

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_messages_request_stays_chat_when_audio_is_registered():
    parsed, request_type = parse_request_type(
        {"messages": [{"role": "user", "content": "hello"}], "modalities": ["audio"]},
        ["text", "audio"],
    )

    assert parsed["messages"][0]["content"] == "hello"
    assert request_type == RequestType.CHAT_COMPLETION


def test_audio_request_shape_wins_when_multiple_modalities_registered():
    parsed, request_type = parse_request_type(
        {"input": "hello", "response_format": "mp3"},
        ["text", "audio"],
    )

    assert isinstance(parsed, NvCreateAudioSpeechRequest)
    assert parsed.input == "hello"
    assert parsed.response_format == "mp3"
    assert request_type == RequestType.AUDIO_GENERATION
