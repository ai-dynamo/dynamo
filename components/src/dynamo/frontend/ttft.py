#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared access to the HTTP request's process-local TTFT timeline."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any


def ttft_stage(context: Any | None, stage: str) -> AbstractContextManager:
    """Record a stage when the caller carries a native HTTP timing context."""
    start_stage = getattr(context, "start_ttft_stage", None)
    return start_stage(stage) if start_stage is not None else nullcontext()
