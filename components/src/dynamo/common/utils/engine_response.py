#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for engine response processing."""

import logging


def normalize_finish_reason(finish_reason: str) -> str:
    """
    Normalize engine finish reasons to Dynamo-compatible values.

    Engine may return finish reasons that aren't recognized by Dynamo's Rust
    layer OR by the OpenAI-wire FinishReason schema (only stop/length/
    tool_calls/content_filter/function_call are valid on the wire). This
    method maps them to compatible values.
    [TODO]: Remove this method and add the right code in the Rust layer.
    """
    # Map engine's "abort" to the wire-compatible "stop". Note: mapping to
    # "cancelled" (as before) is NOT sufficient — while Dynamo's internal
    # FinishReason enum (protocols::common::FinishReason) accepts
    # "cancelled" via a serde alias, the external dynamo-protocols crate's
    # FinishReason (used for the actual OpenAI response/stream chunk) has
    # no Cancelled/Abort variant at all, so "cancelled" still crashes at
    # that later deserialization boundary. "stop" is lossy (loses the
    # abort signal) but is valid everywhere downstream.
    if finish_reason and finish_reason.startswith("abort"):
        logging.debug(f"Normalizing finish reason: {finish_reason} to stop")
        return "stop"
    return finish_reason
