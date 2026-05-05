# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.protocols.image_protocol import NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
from dynamo.common.utils.output_modalities import (
    RequestType,
    normalize_output_modalities,
    parse_request_type,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def test_normalize_output_modalities_splits_comma_and_space_tokens():
    assert normalize_output_modalities(["text,audio", " image "]) == [
        "text",
        "audio",
        "image",
    ]


def test_parse_request_type_uses_messages_for_chat_in_mixed_modalities():
    parsed, request_type = parse_request_type(
        {"messages": [{"role": "user", "content": "hello"}]},
        ["text", "audio"],
    )

    assert parsed == {"messages": [{"role": "user", "content": "hello"}]}
    assert request_type == RequestType.CHAT_COMPLETION


def test_parse_request_type_uses_input_for_audio_in_mixed_modalities():
    parsed, request_type = parse_request_type(
        {"input": "hello", "voice": "ethan"},
        ["text,audio"],
    )

    assert isinstance(parsed, NvCreateAudioSpeechRequest)
    assert request_type == RequestType.AUDIO_GENERATION


def test_parse_request_type_uses_image_model_for_prompt_request():
    parsed, request_type = parse_request_type(
        {"prompt": "a red apple"},
        ["image"],
    )

    assert isinstance(parsed, NvCreateImageRequest)
    assert request_type == RequestType.IMAGE_GENERATION


def test_parse_request_type_uses_video_model_for_video_specific_request():
    parsed, request_type = parse_request_type(
        {"prompt": "a short clip", "model": "video-model", "seconds": 2},
        ["image", "video"],
    )

    assert isinstance(parsed, NvCreateVideoRequest)
    assert request_type == RequestType.VIDEO_GENERATION
