# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal content normalization on the sglang chat-processor path.

Regression coverage for the bug where ``apply_chat_template`` ran on raw
OpenAI messages: VLM templates branch on ``type == 'image'`` while OpenAI
sends ``image_url``, so placeholders never rendered.
"""

import pytest

from dynamo.frontend.sglang_prepost import (
    _normalize_messages_for_template,
    preprocess_chat_request,
)

# Needs sglang packages (gpu_1 container) but allocates no GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

# A list-iterating template; this is what makes sglang's detector report the
# "openai" content format that drives normalization.
_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.content is iterable and message.content is not string %}"
    "{% for chunk in message.content %}"
    "{% if chunk.type == 'image' %}<IMG>"
    "{% elif chunk.type == 'video' %}<VID>"
    "{% elif chunk.type == 'audio' %}<AUD>"
    "{% elif chunk.type == 'text' %}{{ chunk.text }}"
    "{% endif %}{% endfor %}{% endif %}{% endfor %}"
)


def _url_chunk(kind: str) -> dict:
    return {"type": f"{kind}_url", f"{kind}_url": {"url": "x://m"}}


@pytest.fixture
def tokenizer():
    class Tokenizer:
        chat_template = _TEMPLATE

    return Tokenizer()


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_url_chunk_normalized_to_bare_type(tokenizer, kind):
    messages = [{"role": "user", "content": [_url_chunk(kind)]}]
    out = _normalize_messages_for_template(messages, tokenizer)
    assert [c["type"] for c in out[0]["content"]] == [kind]


def test_text_only_passes_through(tokenizer):
    messages = [{"role": "user", "content": "hello"}]
    assert _normalize_messages_for_template(messages, tokenizer) == messages


def test_preprocess_feeds_normalized_chunks_to_template():
    class Tokenizer:
        chat_template = _TEMPLATE

        def apply_chat_template(self, messages, **_):
            self.seen = messages[0]["content"]
            return [42 if c["type"] == "image" else 1 for c in self.seen]

        def encode(self, prompt):
            raise AssertionError("text path must not be taken")

    tok = Tokenizer()
    request = {
        "model": "fake-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    _url_chunk("image"),
                ],
            }
        ],
    }

    result = preprocess_chat_request(
        request,
        tokenizer=tok,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    assert 42 in result.prompt_token_ids
    assert [c["type"] for c in tok.seen] == ["text", "image"]


def test_kimi_k3_custom_template_emits_one_media_pad_per_image():
    """K3's custom tokenizer has no Jinja template and needs image_prompts."""

    class Tokenizer:
        chat_template = None

        def apply_chat_template(self, messages, **kwargs):
            self.seen = messages[0]["content"]
            self.image_prompts = kwargs.get("image_prompts")
            return [42 if chunk["type"] == "image" else 1 for chunk in self.seen]

    tok = Tokenizer()
    request = {
        "model": "moonshot-ai/Kimi-K3",
        "messages": [
            {
                "role": "user",
                "content": [
                    _url_chunk("image"),
                    _url_chunk("image"),
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
    }

    result = preprocess_chat_request(
        request,
        tokenizer=tok,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    assert result.prompt_token_ids == [42, 42, 1]
    assert [chunk["type"] for chunk in tok.seen] == ["image", "image", "text"]
    assert tok.image_prompts == ["<|media_pad|>", "<|media_pad|>"]
