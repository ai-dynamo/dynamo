# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.utils import extract_mm_urls

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_returns_none_for_text_only():
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert extract_mm_urls(messages) is None


def test_returns_none_for_empty_messages():
    assert extract_mm_urls([]) is None


def test_extracts_image_urls():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {"image_url": [{"Url": "https://example.com/cat.png"}]}


def test_extracts_audio_urls():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64,UklGRg=="},
                },
                {"type": "text", "text": "What sound is this?"},
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {"audio_url": [{"Url": "data:audio/wav;base64,UklGRg=="}]}


def test_extracts_video_urls():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"},
                },
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {"video_url": [{"Url": "https://example.com/clip.mp4"}]}


def test_extracts_mixed_modalities():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.jpg"},
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": "https://example.com/audio.wav"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/video.mp4"},
                },
                {"type": "text", "text": "Describe all of these."},
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {
        "image_url": [{"Url": "https://example.com/img.jpg"}],
        "audio_url": [{"Url": "https://example.com/audio.wav"}],
        "video_url": [{"Url": "https://example.com/video.mp4"}],
    }


def test_extracts_multiple_items_per_modality():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/a.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/b.png"},
                },
                {"type": "text", "text": "Compare these images."},
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {
        "image_url": [
            {"Url": "https://example.com/a.png"},
            {"Url": "https://example.com/b.png"},
        ]
    }


def test_ignores_non_user_messages():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/fake.png"},
                },
            ],
        },
        {"role": "user", "content": "Hello"},
    ]
    assert extract_mm_urls(messages) is None


def test_handles_malformed_content_non_dict():
    """Non-dict items in content list should be skipped, not crash."""
    messages = [
        {
            "role": "user",
            "content": [
                "a plain string instead of a dict",
                42,
                None,
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/ok.png"},
                },
            ],
        }
    ]
    result = extract_mm_urls(messages)
    assert result == {"image_url": [{"Url": "https://example.com/ok.png"}]}
