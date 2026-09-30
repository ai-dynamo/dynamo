#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for engine response processing."""

import logging


def normalize_finish_reason(finish_reason: str) -> str:
    """
    Normalize engine finish reasons to Dynamo-compatible values.

    Engine may return finish reasons that aren't recognized by Dynamo's Rust layer.
    This method maps them to compatible values.
    [TODO]: Remove this method and add the right code in the Rust layer.
    """
    # Map engine's "abort" to Dynamo's "cancelled"
    if finish_reason and finish_reason.startswith("abort"):
        logging.debug(f"Normalizing finish reason: {finish_reason} to cancelled")
        return "cancelled"
    return finish_reason


def trailing_stop_prefix_len(text: str, stop_strings: set[str]) -> int:
    """Return the longest trailing substring that prefixes a stop string."""
    if not text or not stop_strings:
        return 0
    max_len = min(len(text), max(len(stop) for stop in stop_strings))
    for suffix_len in range(max_len, 0, -1):
        suffix = text[-suffix_len:]
        if any(stop.startswith(suffix) for stop in stop_strings):
            return suffix_len
    return 0
