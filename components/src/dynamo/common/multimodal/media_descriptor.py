# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for frontend-decoded media descriptors."""

from typing import Any, Final

CONTENT_HASH_KEY: Final = "content_hash"


def decoded_content_hash_key(metadata: Any) -> str | None:
    """Return the canonical hash from a frontend-decoded descriptor, if valid.

    Rust serializes this key as exactly 16 lowercase hexadecimal characters.
    Missing or malformed keys return ``None`` so callers can bypass caches while
    still processing the decoded media.
    """
    if not isinstance(metadata, dict):
        return None
    key = metadata.get(CONTENT_HASH_KEY)
    if not isinstance(key, str) or len(key) != 16:
        return None
    if any(char not in "0123456789abcdef" for char in key):
        return None
    return key
