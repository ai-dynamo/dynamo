#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for parsing and handling endpoint types."""

from dynamo.llm import ModelType


def parse_endpoint_types(endpoint_types_str: str) -> ModelType:
    """Parse comma-separated endpoint types into ModelType flags.

    Args:
        endpoint_types_str: Comma-separated list of endpoint types.
                          Valid values: 'chat', 'completions'
                          Examples: 'chat', 'completions', 'chat,completions'

    Returns:
        ModelType flags combined with bitwise OR

    Raises:
        ValueError: If any invalid endpoint type is provided or string is empty

    Examples:
        >>> parse_endpoint_types("chat")
        ModelType.Chat
        >>> parse_endpoint_types("completions")
        ModelType.Completions
        >>> parse_endpoint_types("chat,completions")
        ModelType.Chat | ModelType.Completions
    """
    if not endpoint_types_str or not endpoint_types_str.strip():
        raise ValueError("Endpoint types string cannot be empty")

    types = [t.strip().lower() for t in endpoint_types_str.split(",") if t.strip()]

    if not types:
        raise ValueError("No valid endpoint types provided")

    result = None
    for t in types:
        if t == "chat":
            flag = ModelType.Chat
        elif t == "completions":
            flag = ModelType.Completions
        else:
            raise ValueError(
                f"Invalid endpoint type: '{t}'. Valid options: 'chat', 'completions'"
            )

        result = flag if result is None else result | flag

    return result
