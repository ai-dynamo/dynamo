# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecation policy for engine-only replay through Dynamo entry points."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator

logger = logging.getLogger(__name__)

_MIGRATION_GUIDE = (
    "https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/"
    "developer-guide/knowledge-base/modular-components/"
    "ai-simulate-experimental/overview.md"
)
_suppress_api_warning: ContextVar[bool] = ContextVar(
    "dynamo_replay_suppress_api_deprecation_warning",
    default=False,
)


def uses_dynamo_integration(
    *,
    replay_mode: str,
    router_mode: str,
    router_config: object | None,
    aic_perf_config: object | None,
    planner_config: object | None,
    model_name: str | None,
) -> bool:
    """Return whether a replay selects a Dynamo-owned integration."""

    return (
        replay_mode == "online"
        or router_mode != "round_robin"
        or router_config is not None
        or aic_perf_config is not None
        or planner_config is not None
        or model_name is not None
    )


def warn_engine_only_replay(
    entry_point: str,
    replacement: str,
    *,
    log: bool = False,
) -> None:
    """Warn once per legacy entry point and replacement pair."""

    if _suppress_api_warning.get():
        return
    _warn_engine_only_replay_once(entry_point, replacement, log=log)


@lru_cache(maxsize=None)
def _warn_engine_only_replay_once(
    entry_point: str,
    replacement: str,
    *,
    log: bool,
) -> None:
    message = (
        f"Dynamo engine-only offline replay via `{entry_point}` is deprecated "
        "and will be removed in Dynamo 1.6.0. Use "
        f"`{replacement}` instead. Dynamo 1.5.0 retains this compatibility "
        "path; Router, Planner, and online replay remain Dynamo-owned and are "
        f"not deprecated. Migration guide: {_MIGRATION_GUIDE}"
    )
    warnings.warn(message, FutureWarning, stacklevel=4)
    if log:
        logger.warning("%s", message)


@contextmanager
def suppress_engine_only_api_warning() -> Iterator[None]:
    """Avoid a second API warning after a CLI or adapter boundary warned."""

    token = _suppress_api_warning.set(True)
    try:
        yield
    finally:
        _suppress_api_warning.reset(token)
