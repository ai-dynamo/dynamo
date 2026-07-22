# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os

logger = logging.getLogger(__name__)


def is_truthy(val: str) -> bool:
    """Truthy values: "1", "true", "on", "yes" (case-insensitive)."""
    return val.strip().lower() in ("1", "true", "on", "yes")


def get_consolidator_mode() -> str:
    """Return the KV event consolidator mode from DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE.

    Returns "dedup" or "passthrough"; invalid/unset values fall back to "dedup".
    """
    mode = os.getenv("DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE", "dedup").strip().lower()
    if mode in ("dedup", "passthrough"):
        return mode

    logger.warning(
        "Invalid DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE=%r. Falling back to 'dedup'.",
        mode,
    )
    return "dedup"


try:
    from nvtx import annotate  # type: ignore
except ImportError:

    def annotate(*args, **kwargs):
        """Dummy decorator when nvtx is not available."""
        # If called with a single callable argument and no kwargs,
        # it's being used as @annotate (without parentheses)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        # Otherwise, it's @annotate(...) and should return a decorator
        def decorator(func):
            return func

        return decorator


def is_dyn_runtime_enabled() -> bool:
    """
    Return True if DYN_RUNTIME_ENABLED_KVBM is set to '1' or 'true' (case-insensitive).
    DYN_RUNTIME_ENABLED_KVBM indicates if KVBM should use the existing DistributedRuntime
    in the current environment.

    WRN: Calling DistributedRuntime.detached() can crash the entire process if
    dependencies are not satisfied, and it cannot be caught with try/except in Python.
    TODO: Make DistributedRuntime.detached() raise a catchable Python exception and
    avoid crashing the process.
    """
    val = os.environ.get("DYN_RUNTIME_ENABLED_KVBM", "").strip().lower()
    return val in {"1", "true"}


def nvtx_annotate(func=None, *, domain="kvbm", category=None):
    """Decorator for NVTX annotation. Use as @nvtx_annotate or @nvtx_annotate(category="...")."""

    def decorator(f):
        kwargs = dict(message=f.__qualname__, color="green", domain=domain)
        if category is not None:
            kwargs["category"] = category
        return annotate(**kwargs)(f)

    if func is not None:
        return decorator(func)
    return decorator
