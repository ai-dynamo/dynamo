# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chat message utility functions."""

import pytest

from dynamo.vllm.multimodal_utils.chat_message_utils import extract_user_text
from dynamo.vllm.multimodal_utils.protocol import (
    ChatMessage,
    ImageContent,
    ImageURLDetail,
    TextContent,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_extract_user_text_single_message():
    """Test extracting text from a single user message with one text content."""
    messages = [
        ChatMessage(
            role="user", content=[TextContent(type="text", text="Hello, world!")]
        )
    ]

    result = extract_user_text(messages)
    assert result == "Hello, world!"


def test_extract_user_text_multiple_text_parts():
    """Test extracting text from a user message with multiple text content items."""
    messages = [
        ChatMessage(
            role="user",
            content=[
                TextContent(type="text", text="First part "),
                ImageContent(
                    type="image_url",
                    image_url=ImageURLDetail(url="http://example.com/image.jpg"),
                ),
                TextContent(type="text", text="second part"),
            ],
        )
    ]

    result = extract_user_text(messages)
    assert result == "First part second part"


def test_extract_user_text_multi_turn():
    """Test extracting text from multi-turn conversation."""
    messages = [
        ChatMessage(
            role="user", content=[TextContent(type="text", text="First question")]
        ),
        ChatMessage(
            role="assistant", content=[TextContent(type="text", text="First answer")]
        ),
        ChatMessage(
            role="user", content=[TextContent(type="text", text="Second question")]
        ),
    ]

    result = extract_user_text(messages)
    assert result == "First question\nSecond question"


def test_extract_user_text_only_images():
    """Test that ValueError is raised when messages contain only images."""
    messages = [
        ChatMessage(
            role="user",
            content=[
                ImageContent(
                    type="image_url",
                    image_url=ImageURLDetail(url="http://example.com/image.jpg"),
                )
            ],
        )
    ]

    with pytest.raises(ValueError, match="No text content found in user messages"):
        extract_user_text(messages)


def test_extract_user_text_empty_messages():
    """Test that ValueError is raised when messages list is empty."""
    messages: list[ChatMessage] = []

    with pytest.raises(ValueError, match="No text content found in user messages"):
        extract_user_text(messages)


def test_extract_user_text_no_user_messages():
    """Test that ValueError is raised when there are no user role messages."""
    messages = [
        ChatMessage(
            role="assistant",
            content=[TextContent(type="text", text="Just an assistant message")],
        )
    ]

    with pytest.raises(ValueError, match="No text content found in user messages"):
        extract_user_text(messages)


def test_extract_user_text_mixed_roles():
    """Test extracting text only from user messages, ignoring other roles."""
    messages = [
        ChatMessage(
            role="system", content=[TextContent(type="text", text="System prompt")]
        ),
        ChatMessage(
            role="user", content=[TextContent(type="text", text="User message 1")]
        ),
        ChatMessage(
            role="assistant",
            content=[TextContent(type="text", text="Assistant response")],
        ),
        ChatMessage(
            role="user", content=[TextContent(type="text", text="User message 2")]
        ),
    ]

    result = extract_user_text(messages)
    assert result == "User message 1\nUser message 2"


def test_extract_user_text_empty_text_content():
    """Test that empty text content items are ignored."""
    messages = [
        ChatMessage(
            role="user",
            content=[
                TextContent(type="text", text=""),
                TextContent(type="text", text="Valid text"),
                TextContent(type="text", text=""),
            ],
        )
    ]

    result = extract_user_text(messages)
    assert result == "Valid text"
