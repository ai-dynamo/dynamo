# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing environment variables."""

import logging
import os

_TRUTHY = ("true", "1", "yes")
_FPM_TRACE_TRUTHY = (*_TRUTHY, "on")
_FPM_TRACE_FALSY = ("false", "0", "no", "off")

logger = logging.getLogger(__name__)
_fpm_trace_invalid_warning_emitted = False


def env_bool(name: str, *, default: bool = False) -> bool:
    """Return True if env var `name` is set to a truthy value.

    Truthy values (case-insensitive): "true", "1", "yes". Any other non-empty
    value is treated as False. When the var is unset or empty, returns `default`.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    return raw.lower() in _TRUTHY


def fpm_trace_enabled() -> bool:
    """Strictly parse the ``DYN_FPM_TRACE`` master switch.

    Unlike :func:`env_bool`, a set-but-invalid value is configuration error
    worth surfacing. It disables FPM tracing and emits at most one warning per
    worker process so the backend's scheduler and relay checks do not duplicate
    the message.
    """
    raw = os.environ.get("DYN_FPM_TRACE")
    if raw is None:
        return False

    normalized = raw.strip().lower()
    if normalized in _FPM_TRACE_TRUTHY:
        return True
    if normalized in _FPM_TRACE_FALSY:
        return False

    global _fpm_trace_invalid_warning_emitted
    if not _fpm_trace_invalid_warning_emitted:
        _fpm_trace_invalid_warning_emitted = True
        logger.warning(
            "Invalid DYN_FPM_TRACE value %r; expected one of 1/0, "
            "true/false, on/off, or yes/no. FPM tracing is disabled for this worker.",
            raw,
        )
    return False
