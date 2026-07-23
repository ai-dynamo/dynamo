#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for engine response processing."""

import logging
from typing import Optional

NormalizedFinishReason = str | dict[str, str] | None


def normalize_finish_reason(finish_reason: Optional[str]) -> NormalizedFinishReason:
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
    # Rust serializes FinishReason::Error(String) as {"error": "..."}.
    # Returning the bare string "error" causes deserialization to fail on the
    # frontend when backend workers report a generation error.
    if finish_reason == "error":
        logging.debug("Normalizing bare error finish reason to structured error")
        return {"error": "backend error"}
    if finish_reason and finish_reason.startswith("error:"):
        logging.debug("Normalizing string error finish reason to structured error")
        return {"error": finish_reason.split(":", 1)[1].strip() or "backend error"}
    return finish_reason
