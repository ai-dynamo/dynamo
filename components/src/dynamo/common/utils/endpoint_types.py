#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for parsing and handling endpoint types."""

from dynamo.llm import ModelType


def parse_endpoint_types(endpoint_types_str: str) -> ModelType:
    """Parse comma-separated endpoint types into ModelType flags.

    Args:
        endpoint_types_str: Comma-separated list of endpoint types.
                          Valid values: 'chat', 'completions', 'internal'
                          Examples: 'chat', 'completions', 'chat,completions',
                          'internal'

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
        >>> parse_endpoint_types("internal")
        ModelType.Empty
    """
    if not endpoint_types_str or not endpoint_types_str.strip():
        raise ValueError("Endpoint types string cannot be empty")

    types = [t.strip().lower() for t in endpoint_types_str.split(",") if t.strip()]

    if not types:
        raise ValueError("No valid endpoint types provided")
    if "internal" in types and len(types) != 1:
        raise ValueError("'internal' cannot be combined with public endpoint types")

    result: ModelType | None = None
    for t in types:
        if t == "chat":
            flag = ModelType.Chat
        elif t == "completions":
            flag = ModelType.Completions
        elif t == "internal":
            flag = ModelType.Empty
        else:
            raise ValueError(
                f"Invalid endpoint type: '{t}'. Valid options: "
                "'chat', 'completions', 'internal'"
            )

        result = flag if result is None else result | flag

    # `types` is validated as non-empty above, so result is guaranteed to be set.
    assert result is not None
    return result
