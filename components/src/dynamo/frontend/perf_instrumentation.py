#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Performance instrumentation for diagnosing frontend preprocessing bottlenecks.

Activated by passing --debug-perf to dynamo.frontend.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency gauge
# ---------------------------------------------------------------------------

_active_requests = 0
_peak_requests = 0


def enter_generator() -> int:
    """Increment active request count. Returns current count.

    Safe without a lock: only called while the GIL is held (all callers are
    in Python code), so the read-modify-write on the global int is atomic
    with respect to other Python threads.
    """
    global _active_requests, _peak_requests
    _active_requests += 1
    count = _active_requests
    if count > _peak_requests:
        _peak_requests = count
    return count


def exit_generator() -> int:
    """Decrement active request count. Returns current count."""
    global _active_requests
    _active_requests -= 1
    return _active_requests


def get_active_requests() -> int:
    return _active_requests


def get_peak_requests() -> int:
    return _peak_requests


# ---------------------------------------------------------------------------
# sys.monitoring profiler for key functions
# ---------------------------------------------------------------------------

_TOOL_ID = 3  # Unassigned slot in sys.monitoring
_monitoring_active = False

# Track entry times per (thread_id, code_id) to handle concurrency.
# No lock needed: callbacks run while the GIL is held.
_entry_times: dict[tuple[int, int], float] = {}

# Accumulated stats: {qualified_name: [call_count, total_time_ms]}
# No lock needed: only mutated from callbacks which run under the GIL.
_func_stats: dict[str, list[int | float]] = {}

# Set of code.co_qualname substrings we want to trace.
_TRACED_QUALNAMES: set[str] = set()

# Log interval
_monitoring_last_log = 0.0
_MONITORING_LOG_INTERVAL_S = 30.0


def _on_py_start(code: Any, instruction_offset: int) -> Any:
    qualname = code.co_qualname
    # Fast check: does this qualname match any of our traced functions?
    # For non-matching functions, returning DISABLE tells sys.monitoring to
    # suppress future callbacks for this specific (code, offset) pair.
    # This means the substring scan only runs ONCE per unique call site,
    # not on every invocation — so steady-state overhead is near zero for
    # the vast majority of Python functions.
    for prefix in _TRACED_QUALNAMES:
        if prefix in qualname:
            _entry_times[(threading.get_ident(), id(code))] = time.monotonic()
            return None
    return sys.monitoring.DISABLE


def _on_py_return(code: Any, instruction_offset: int, retval: Any) -> Any:
    key = (threading.get_ident(), id(code))
    start = _entry_times.pop(key, None)
    if start is None:
        return

    elapsed_ms = (time.monotonic() - start) * 1000.0
    qualname = code.co_qualname

    entry = _func_stats.get(qualname)
    if entry is None:
        _func_stats[qualname] = [1, elapsed_ms]
    else:
        entry[0] += 1
        entry[1] += elapsed_ms

    _maybe_log_func_stats()


def _maybe_log_func_stats() -> None:
    global _monitoring_last_log
    now = time.monotonic()
    if now - _monitoring_last_log < _MONITORING_LOG_INTERVAL_S:
        return
    _monitoring_last_log = now

    if not _func_stats:
        return

    # Sort by total time descending
    sorted_stats = sorted(_func_stats.items(), key=lambda x: x[1][1], reverse=True)
    lines = ["[perf] sys.monitoring function profile (last 30s):"]
    for qualname, (count, total_ms) in sorted_stats[:15]:
        avg_ms = total_ms / count if count else 0
        lines.append(
            f"  {qualname}: calls={count} total={total_ms:.1f}ms avg={avg_ms:.2f}ms"
        )
    logger.info("\n".join(lines))

    _func_stats.clear()


def start_monitoring_profiler(
    extra_qualnames: list[str] | None = None,
) -> None:
    """
    Start sys.monitoring-based profiler for key preprocessing functions.
    Traces wall-clock time spent in specified function qualnames.
    """
    global _monitoring_active, _monitoring_last_log
    if _monitoring_active:
        return

    # Default set of functions to trace
    _TRACED_QUALNAMES.update(
        {
            # prepost.py
            "preprocess_chat_request",
            # vllm input processing
            "InputProcessor.process_inputs",
            "InputPreprocessor.preprocess",
            # vllm output processing
            "OutputProcessor.process_outputs",
            # tokenization
            "encode",
            "decode",
            # Chat template rendering
            "render_messages_async",
            "apply_chat_template",
            # Pydantic validation
            "model_validate",
            # Detokenizer
            "Detokenizer.update",
            # Post-processing
            "process_output",
            # Sampling params
            "SamplingParams.__init__",
        }
    )
    if extra_qualnames:
        _TRACED_QUALNAMES.update(extra_qualnames)

    try:
        sys.monitoring.use_tool_id(_TOOL_ID, "dynamo_perf")
    except ValueError:
        logger.info(
            "[perf] sys.monitoring tool_id %d already in use, skipping profiler",
            _TOOL_ID,
        )
        return

    sys.monitoring.register_callback(
        _TOOL_ID, sys.monitoring.events.PY_START, _on_py_start
    )
    sys.monitoring.register_callback(
        _TOOL_ID, sys.monitoring.events.PY_RETURN, _on_py_return
    )
    sys.monitoring.set_events(
        _TOOL_ID,
        sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
    )

    _monitoring_active = True
    _monitoring_last_log = time.monotonic()
    logger.info(
        "[perf] sys.monitoring profiler started (tracing %d function patterns)",
        len(_TRACED_QUALNAMES),
    )


def stop_monitoring_profiler() -> None:
    global _monitoring_active
    if not _monitoring_active:
        return
    try:
        sys.monitoring.set_events(_TOOL_ID, sys.monitoring.events.NO_EVENTS)
        sys.monitoring.register_callback(_TOOL_ID, sys.monitoring.events.PY_START, None)
        sys.monitoring.register_callback(
            _TOOL_ID, sys.monitoring.events.PY_RETURN, None
        )
        sys.monitoring.free_tool_id(_TOOL_ID)
    except Exception:
        pass
    _monitoring_active = False
    logger.info("[perf] sys.monitoring profiler stopped")


# ---------------------------------------------------------------------------
# Timer context manager for manual hot-path instrumentation
# ---------------------------------------------------------------------------


@contextmanager
def timed_section(name: str, request_id: str):
    """
    Context manager that logs wall-clock time for a named section.
    Usage:
        with timed_section("preprocess", request_id):
            ...
    """
    t0 = time.monotonic()
    yield
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if elapsed_ms > 0.5:  # Only log if > 0.5ms to reduce noise
        logger.info(
            "[perf] %s: %.2fms (request=%s active_requests=%d)",
            name,
            elapsed_ms,
            request_id,
            _active_requests,
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start_all() -> None:
    """Start all performance instrumentation."""
    logger.info("[perf] Starting performance instrumentation")
    start_monitoring_profiler()


def stop_all() -> None:
    """Stop all performance instrumentation."""
    stop_monitoring_profiler()
    logger.info("[perf] Performance instrumentation stopped")
