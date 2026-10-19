#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared access to the HTTP request's process-local TTFT timeline."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any


class _TtftContext:
    """Cache native methods once per request, only when collection is enabled."""

    ttft_enabled = True

    def __init__(self, context: Any) -> None:
        self.start_ttft_stage = context.start_ttft_stage
        self.record_ttft_stage_ago = context.record_ttft_stage_ago


def ttft_context(context: Any | None) -> Any | None:
    """Resolve the explicit Rust enable bit once before Python preprocessing."""
    if isinstance(context, _TtftContext):
        return context
    if not getattr(context, "ttft_enabled", False):
        return None
    return _TtftContext(context)


def ttft_stage(context: Any | None, stage: str) -> AbstractContextManager:
    """Record a stage when the caller carries a native HTTP timing context."""
    start_stage = (
        getattr(context, "start_ttft_stage", None)
        if getattr(context, "ttft_enabled", False)
        else None
    )
    return start_stage(stage) if start_stage is not None else nullcontext()
