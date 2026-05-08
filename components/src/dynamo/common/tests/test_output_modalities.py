# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.utils.output_modalities import RequestType, parse_request_type

pytestmark = [pytest.mark.gpu_0, pytest.mark.unit, pytest.mark.pre_merge]


def test_parse_request_type_prefers_chat_messages_for_multimodal_models():
    parsed, request_type = parse_request_type(
        {
            "messages": [{"role": "user", "content": "say hello"}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
        },
        ["text", "audio"],
    )

    assert request_type == RequestType.CHAT_COMPLETION
    assert parsed["modalities"] == ["text", "audio"]


def test_parse_request_type_recognizes_audio_endpoint_with_text_first_model():
    parsed, request_type = parse_request_type(
        {"input": "say hello", "voice": "Cherry", "response_format": "wav"},
        ["text", "audio"],
    )

    assert request_type == RequestType.AUDIO_GENERATION
    assert isinstance(parsed, NvCreateAudioSpeechRequest)
    assert parsed.input == "say hello"
