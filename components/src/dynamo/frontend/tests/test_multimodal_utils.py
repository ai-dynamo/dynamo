# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.utils import extract_mm_urls

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _sglang_prepost():
    pytest.importorskip(
        "sglang.srt.parser.jinja_template_utils",
        reason="sglang frontend parser modules are not installed in this environment",
    )
    import dynamo.frontend.sglang_prepost as sglang_prepost

    return sglang_prepost


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


# ---------------------------------------------------------------------------
# Chat-template content normalization (image_url -> image, etc.)
# ---------------------------------------------------------------------------
#
# Regression coverage for the dynamo-sglang chat processor bug where the
# Python prepost path called ``apply_chat_template`` on raw OpenAI messages.
# Modern VLM chat templates branch on ``item.type == 'image'`` /
# ``'video'`` and never fire for ``image_url`` / ``video_url``, so the
# rendered prompt loses its placeholder tokens and the worker has no slot
# to bind media bytes to.

# A chat template that iterates ``message.content`` as a list. This is
# what triggers sglang's content-format detector to return ``"openai"``.
_OPENAI_FORMAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.content is iterable and message.content is not string %}"
    "{% for chunk in message.content %}"
    "{% if chunk.type == 'image' %}<IMG>"
    "{% elif chunk.type == 'video' %}<VID>"
    "{% elif chunk.type == 'text' %}{{ chunk.text }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
)


def test_normalize_messages_converts_image_url_to_image():
    """``image_url`` content parts become ``{"type": "image"}`` for the template."""

    class T:
        chat_template = _OPENAI_FORMAT_TEMPLATE

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
            ],
        }
    ]

    out = _sglang_prepost()._normalize_messages_for_template(messages, T())
    chunk_types = [c["type"] for c in out[0]["content"]]
    assert "image" in chunk_types
    assert "image_url" not in chunk_types


def test_normalize_messages_converts_video_url_to_video():
    """``video_url`` content parts become ``{"type": "video"}`` for the template."""

    class T:
        chat_template = _OPENAI_FORMAT_TEMPLATE

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

    out = _sglang_prepost()._normalize_messages_for_template(messages, T())
    chunk_types = [c["type"] for c in out[0]["content"]]
    assert "video" in chunk_types
    assert "video_url" not in chunk_types


def test_normalize_messages_passes_through_text_only():
    """Pure-text messages survive unchanged."""

    class T:
        chat_template = _OPENAI_FORMAT_TEMPLATE

    messages = [{"role": "user", "content": "Hello"}]
    out = _sglang_prepost()._normalize_messages_for_template(messages, T())
    assert out == [{"role": "user", "content": "Hello"}]


def test_preprocess_chat_request_renders_image_placeholder():
    """End-to-end: an ``image_url`` chunk reaches ``apply_chat_template`` as
    ``image``, and the rendered prompt contains the template's image
    placeholder token. This is the regression assertion for the
    multimodal bug under ``--dyn-chat-processor sglang``.
    """

    captured = {}

    class TemplateTokenizer:
        chat_template = _OPENAI_FORMAT_TEMPLATE

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            # The template would resolve `<IMG>` for an image chunk; mirror
            # that here by emitting a sentinel token id when the placeholder
            # would have rendered.
            ids = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for chunk in content:
                        if chunk.get("type") == "image":
                            ids.append(424242)  # <IMG> sentinel
                        elif chunk.get("type") == "text":
                            ids.append(1)
            return ids

        def encode(self, prompt):
            raise AssertionError("encode should not be called on template path")

    request = {
        "model": "fake-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
    }

    result = _sglang_prepost().preprocess_chat_request(
        request,
        tokenizer=TemplateTokenizer(),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    # The bug: without normalization, the template saw ``type == 'image_url'``
    # and the sentinel never landed. With the fix, the sentinel is present.
    assert 424242 in result.prompt_token_ids
    seen_types = [c["type"] for c in captured["messages"][0]["content"]]
    assert "image" in seen_types
    assert "image_url" not in seen_types
