# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for processing chat messages."""

from typing import List

from dynamo.vllm.multimodal_utils.protocol import ChatMessage


def extract_user_text(messages: List[ChatMessage]) -> str:
    """Extract and concatenate text content from user messages."""

    # This function finds all text content items from "user" role messages,
    # and concatenates them. For multi-turn conversation, it adds a newline
    # between each turn. This is not a perfect solution as we encode multi-turn
    # conversation as a single turn. However, multi-turn conversation in a
    # single request is not well defined in the spec.

    # TODO: Revisit this later when adding multi-turn conversation support.
    user_texts = []
    for message in messages:
        if message.role == "user":
            # Collect all text content items from this user message
            text_parts = []
            for item in message.content:
                if item.type == "text" and item.text:
                    text_parts.append(item.text)
            # If this user message has text content, join it and add to user_texts
            if text_parts:
                user_texts.append("".join(text_parts))

    if not user_texts:
        raise ValueError("No text content found in user messages")

    # Join all user turns with newline separator
    return "\n".join(user_texts)
