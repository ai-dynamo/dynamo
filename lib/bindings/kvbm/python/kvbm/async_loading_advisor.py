# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared async loading advisor registry for KVBM integrations.

This hook is intentionally kept outside the TRT-LLM integration package so
leader-side advisory wiring can evolve with less connector churn.
"""

from collections.abc import Callable
from typing import Optional

AsyncLoadingAdvisor = Callable[
    [str, list[int], Optional[str], Optional[str], int, int], None
]

__all__ = [
    "AsyncLoadingAdvisor",
    "register_async_loading_advisor",
    "clear_async_loading_advisor",
    "get_async_loading_advisor",
]

_async_loading_advisor: Optional[AsyncLoadingAdvisor] = None


def register_async_loading_advisor(advisor: Optional[AsyncLoadingAdvisor]) -> None:
    global _async_loading_advisor
    _async_loading_advisor = advisor


def clear_async_loading_advisor() -> None:
    register_async_loading_advisor(None)


def get_async_loading_advisor() -> Optional[AsyncLoadingAdvisor]:
    return _async_loading_advisor
