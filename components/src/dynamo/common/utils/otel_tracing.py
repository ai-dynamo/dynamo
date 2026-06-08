# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for forwarding OTel trace context to engines."""

from __future__ import annotations

from dynamo._core import Context


def build_trace_headers(
    context: Context | None,
    *,
    enabled: bool = True,
) -> dict[str, str] | None:
    """Return W3C trace headers for an engine request when available."""
    if not enabled or context is None:
        return None
    return context.trace_headers()
