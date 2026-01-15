# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for processing chat messages."""


def extract_user_text(messages) -> str:
    """Extract and concatenate text content from user messages."""
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
