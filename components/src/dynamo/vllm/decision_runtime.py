# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Effective model implementation checks for pinned native decision profiles."""

from collections.abc import Awaitable
from typing import Any, TypeVar

_T = TypeVar("_T")


def is_native_decision_model(model: Any, *, diffusion: bool) -> bool:
    """Exclude replacement implementations even when architecture names match."""
    try:
        return (
            model.model_impl in ("auto", "vllm")
            and not model.model_class_overrides
            and model.convert_type == "none"
            and model.is_diffusion is diffusion
            and model.is_encoder_decoder is False
        )
    except AttributeError:
        return False


async def translate_native_overload(operation: Awaitable[_T], native: Any) -> _T:
    """Bridge only source-qualified native admission exceptions to Dynamo HTTP."""
    try:
        return await operation
    except native.overload_errors as cause:
        raise native.http_error(
            503, "Native decision engine is temporarily unavailable"
        ) from cause
