# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical errors for optional native decision adapters."""

from __future__ import annotations

from dynamo.llm.exceptions import HttpError, InvalidArgument


class NativeRuntimeError(RuntimeError):
    """The optional installed runtime does not implement its pinned contract."""


class NativeValidationError(InvalidArgument):
    """Request-derived content or resource limits failed before engine admission."""


class NativeExecutionError(HttpError):
    """The private executor contract failed before or during engine execution."""

    def __init__(self, message: str):
        super().__init__(502, message)
