# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AuditLogger + AuditEvent catalog (PR 8 sub-task 8-7)."""

from __future__ import annotations

import json
import logging

import pytest

from dynamo.planner.plugins.audit import AuditEvent, AuditLogger

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# AuditEvent catalog
# ---------------------------------------------------------------------------


# Canonical event set — any additions should be reviewed because
# downstream dashboards / replay assertions key off these names.
_EXPECTED_EVENTS = {
    # Plugin lifecycle
    "plugin_evaluated",
    "plugin_degraded",
    "plugin_timeout",
    "plugin_circuit_open",
    "plugin_rejected",
    # EXECUTE
    "execute_invoked",
    "execute_succeeded",
    "execute_failed",
    "execute_skipped_rejected",
    "execute_skipped_no_change",
    "execute_advisory",
    "execute_in_progress",
    # Multi-cadence
    "tick_skipped",
    "tick_timeout",
    # Cross-cutting
    "global_scale_request_rejected",
    "plugin_constrain_set_dropped",
    "orchestrator_drift_detected",
}


def test_event_catalog_matches_expected_set():
    actual = {e.value for e in AuditEvent}
    assert actual == _EXPECTED_EVENTS


def test_event_str_equivalence():
    """AuditEvent inherits from str, so it compares equal to its value."""
    assert AuditEvent.PLUGIN_EVALUATED == "plugin_evaluated"
    assert "plugin_evaluated" == AuditEvent.PLUGIN_EVALUATED


# ---------------------------------------------------------------------------
# Emission shape
# ---------------------------------------------------------------------------


def _capture_audit(caplog):
    """Extract the JSON payload from the most recent AUDIT log line."""
    for record in reversed(caplog.records):
        if record.msg.startswith("AUDIT "):
            return json.loads(record.msg[len("AUDIT "):])
    raise AssertionError("no AUDIT record captured")


def test_emit_enum_event(caplog):
    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="t-1",
            decision_id="d-1",
            plugin_id="pl-a",
            stage="propose",
            result="accept",
            latency_ms=3.2,
        )
    payload = _capture_audit(caplog)
    assert payload["event"] == "plugin_evaluated"
    assert payload["tick_id"] == "t-1"
    assert payload["decision_id"] == "d-1"
    assert payload["plugin_id"] == "pl-a"
    assert payload["stage"] == "propose"
    assert payload["result"] == "accept"
    assert payload["latency_ms"] == 3.2


def test_emit_string_event(caplog):
    """String event names are accepted (tests and ad-hoc instrumentation)."""
    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit("plugin_timeout", tick_id="t-9", plugin_id="slow")
    payload = _capture_audit(caplog)
    assert payload["event"] == "plugin_timeout"
    assert payload["tick_id"] == "t-9"
    assert payload["plugin_id"] == "slow"


def test_emit_preserves_context_none_when_missing(caplog):
    """Missing tick_id / decision_id / plugin_id MUST be null in output,
    not absent. Downstream joins rely on the key always being present."""
    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(AuditEvent.TICK_TIMEOUT)
    payload = _capture_audit(caplog)
    assert payload["event"] == "tick_timeout"
    assert payload["tick_id"] is None
    assert payload["decision_id"] is None
    assert payload["plugin_id"] is None


# ---------------------------------------------------------------------------
# Every event can emit cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", list(AuditEvent))
def test_every_event_emits_without_raising(caplog, event):
    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            event,
            tick_id="t",
            decision_id="d",
            plugin_id="p",
        )
    payload = _capture_audit(caplog)
    assert payload["event"] == event.value


# ---------------------------------------------------------------------------
# Serialisation edge cases
# ---------------------------------------------------------------------------


def test_enum_values_in_fields_serialize(caplog):
    """Passing an Enum as a field value shouldn't break serialisation."""

    class _Stage(str, __import__("enum").Enum):
        PROPOSE = "propose"

    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="t",
            decision_id="d",
            plugin_id="p",
            stage=_Stage.PROPOSE,
        )
    payload = _capture_audit(caplog)
    assert payload["stage"] == "propose"


def test_dataclass_field_is_serialised_as_dict(caplog):
    """A dataclass-typed value should round-trip through asdict()."""
    import dataclasses

    @dataclasses.dataclass
    class _Thing:
        x: int
        y: str

    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="t",
            decision_id="d",
            plugin_id="p",
            detail=_Thing(x=42, y="abc"),
        )
    payload = _capture_audit(caplog)
    assert payload["detail"] == {"x": 42, "y": "abc"}


def test_non_json_value_falls_back_to_repr(caplog):
    """An object that's neither Enum, dataclass, nor JSON-native still
    lands as a string — we never drop an event."""
    audit = AuditLogger()

    class _Opaque:
        def __repr__(self):
            return "<Opaque>"

    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="t",
            decision_id="d",
            plugin_id="p",
            blob=_Opaque(),
        )
    payload = _capture_audit(caplog)
    assert payload["blob"] == "<Opaque>"


def test_audit_uses_dedicated_logger_by_default(caplog):
    audit = AuditLogger()
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(AuditEvent.PLUGIN_EVALUATED, tick_id="t", plugin_id="p")
    names = {r.name for r in caplog.records if r.msg.startswith("AUDIT ")}
    assert names == {"dynamo.planner.audit"}


def test_custom_logger_is_honoured(caplog):
    """Callers can inject a sub-logger (e.g. ``dynamo.planner.audit.replay``)
    for stream splitting."""
    custom = logging.getLogger("dynamo.planner.audit.replay")
    audit = AuditLogger(logger_=custom)
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit.replay"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED, tick_id="t", plugin_id="p"
        )
    names = {r.name for r in caplog.records if r.msg.startswith("AUDIT ")}
    assert names == {"dynamo.planner.audit.replay"}
