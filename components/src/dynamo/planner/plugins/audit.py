# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AuditLogger`` — structured audit log for plugin lifecycle + decisions.

Why a dedicated audit layer instead of scattered ``logger.info`` calls
---------------------------------------------------------------------

Plugin-era observability needs two different log surfaces:

1. **Diagnostic logs** (`logger.info/warning/error`) — human-readable,
   filtered by log level, written to stderr.  Useful when debugging a
   running planner.
2. **Audit events** (this module) — machine-parseable JSON per event,
   stable field names, never filtered out by log level.  Useful when
   reconstructing what a planner decided during an incident, or when
   diffing two runs bit-exactly (replay vs live).

The audit event stream is what feeds dashboards, replay, and post-hoc
forensics; it MUST be stable across refactors.  The helper here:

- Pins every event to the fixed catalog in ``AuditEvent`` so typos
  surface at call time rather than during dashboard queries.
- Forces ``tick_id`` / ``decision_id`` / ``plugin_id`` context on every
  emit (missing values are explicitly ``None`` in the JSON, never
  dropped) so downstream joins against these keys never silently lose
  rows.
- Routes to a dedicated logger (``dynamo.planner.audit``) so operators
  can tee the stream to a separate handler without touching root.

Design notes
------------

We emit a single JSON object per log line, written via
``logger.info(<json>)``.  The surrounding `structlog`-style story was
considered but rejected for v1:

- Adds a runtime dependency the rest of the planner doesn't have.
- `logging`'s ``extra=`` + a formatter we control gives the same
  output shape with no new deps.
- Replay assertions want to grep individual JSON lines, not dig into
  ``structlog`` rendering plugins.

All event names live in ``AuditEvent`` — a plain string enum — so we
get IDE autocomplete and mypy coverage, but the wire format stays a
flat string for downstream consumers.
"""

from __future__ import annotations

import enum
import json
import logging
from typing import Any, Optional

log = logging.getLogger("dynamo.planner.audit")

_AUDIT_LOG_PREFIX = "AUDIT "


class AuditEvent(str, enum.Enum):
    """Canonical audit event catalog.

    Inherits from ``str`` so `audit.emit(AuditEvent.PLUGIN_EVALUATED, ...)`
    serialises identically to `audit.emit("plugin_evaluated", ...)`.
    """

    # --- Plugin lifecycle --------------------------------------------------
    PLUGIN_EVALUATED = "plugin_evaluated"
    """Plugin method invoked; ``result`` field carries accept / reject / set / etc."""

    PLUGIN_DEGRADED = "plugin_degraded"
    """Plugin failed in a way that the orchestrator tolerated (e.g. held over)."""

    PLUGIN_TIMEOUT = "plugin_timeout"
    """Plugin request exceeded ``request_timeout_seconds``."""

    PLUGIN_CIRCUIT_OPEN = "plugin_circuit_open"
    """Circuit breaker opened after repeated failures; plugin calls short-circuit."""

    PLUGIN_REJECTED = "plugin_rejected"
    """Plugin returned an explicit ``RejectResult``; stage short-circuits."""

    # --- EXECUTE -----------------------------------------------------------
    EXECUTE_INVOKED = "execute_invoked"
    """Connector.set_component_replicas was called."""

    EXECUTE_SUCCEEDED = "execute_succeeded"
    """Execute completed without error."""

    EXECUTE_FAILED = "execute_failed"
    """Execute raised / connector returned ERROR."""

    EXECUTE_SKIPPED_REJECTED = "execute_skipped_rejected"
    """Final decision was REJECT; execute not attempted."""

    EXECUTE_SKIPPED_NO_CHANGE = "execute_skipped_no_change"
    """Final targets == current workers; execute not attempted."""

    EXECUTE_ADVISORY = "execute_advisory"
    """Planner running in advisory / observation mode; decision logged only."""

    EXECUTE_IN_PROGRESS = "execute_in_progress"
    """Previous scaling still in progress; skipping this tick."""

    # --- Multi-cadence scheduling -----------------------------------------
    TICK_SKIPPED = "tick_skipped"
    """Plugin was skipped this tick because its execution_interval hadn't elapsed."""

    TICK_TIMEOUT = "tick_timeout"
    """Full tick exceeded ``tick_max_duration_seconds``."""

    # --- Cross-cutting -----------------------------------------------------
    GLOBAL_SCALE_REQUEST_REJECTED = "global_scale_request_rejected"
    """GlobalPlanner refused to honour a scale request."""

    PLUGIN_CONSTRAIN_SET_DROPPED = "plugin_constrain_set_dropped"
    """CONSTRAIN plugin returned SET; orchestrator dropped it (CONSTRAIN is AT_LEAST/AT_MOST only)."""

    ORCHESTRATOR_DRIFT_DETECTED = "orchestrator_drift_detected"
    """Reserved (dual-execution removed; kept for enum stability)."""


class AuditLogger:
    """Structured audit logger for orchestrator and plugin subsystems.

    Usage::

        audit = AuditLogger()
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="tick-12",
            decision_id="d-42",
            plugin_id="builtin_load_propose",
            stage="propose",
            result="accept",
            latency_ms=3.2,
        )

    The ``tick_id`` / ``decision_id`` / ``plugin_id`` positional kwargs
    are conventionally passed to every emit.  Missing values should be
    sent as ``None`` (or omitted entirely) rather than the empty string
    so downstream consumers can distinguish "not applicable" from
    "known but blank".
    """

    def __init__(self, logger_: Optional[logging.Logger] = None) -> None:
        self._log = logger_ if logger_ is not None else log

    def emit(
        self,
        event: AuditEvent | str,
        *,
        tick_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """Emit a single audit event line.

        ``tick_id`` / ``decision_id`` / ``plugin_id`` are surfaced as
        dedicated parameters so call sites never forget them and so a
        type checker catches misspellings. They are always present in
        the output (as ``null`` when not provided).

        Any additional ``**fields`` are included in the JSON payload
        with their keys unchanged; values MUST be JSON-serialisable
        (primitives, lists, dicts). Non-serialisable values are coerced
        via ``repr()`` — a lossy fallback that never drops the event.
        """
        name = event.value if isinstance(event, AuditEvent) else str(event)
        payload: dict[str, Any] = {
            "event": name,
            "tick_id": tick_id,
            "decision_id": decision_id,
            "plugin_id": plugin_id,
        }
        payload.update(fields)
        try:
            line = json.dumps(payload, default=self._json_fallback)
        except (TypeError, ValueError):
            # Ultra-defensive: if json.dumps itself fails, degrade to
            # repr() of the whole payload — the event still lands.
            line = json.dumps(
                {
                    "event": name,
                    "tick_id": tick_id,
                    "decision_id": decision_id,
                    "plugin_id": plugin_id,
                    "_payload_repr": repr(fields),
                    "_serialize_failed": True,
                }
            )
        self._log.info(_AUDIT_LOG_PREFIX + line)

    @staticmethod
    def _json_fallback(value: Any) -> Any:
        """Per-value fallback for ``json.dumps(default=)``. Handles enum
        subclasses and dataclasses; everything else → repr() string."""
        if isinstance(value, enum.Enum):
            return value.value
        try:
            import dataclasses

            if dataclasses.is_dataclass(value):
                return dataclasses.asdict(value)
        except Exception:
            pass
        return repr(value)


__all__ = ["AuditEvent", "AuditLogger"]
