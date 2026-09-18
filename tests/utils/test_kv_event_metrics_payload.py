# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``KvEventMetricsPayload`` and the Prometheus text helpers.

The regression these cover: the KV publisher's
``engines_dropped_events_total`` counter never reached the metrics registry
because it declared ``worker_id`` as a variable label, colliding with the const
label the runtime auto-injects under that name. The counter only increments when
an ``event_id`` gap is detected, so it sits at zero on a healthy worker — which
means a value-based assertion cannot tell "registered, never incremented" apart
from "never registered at all". These tests pin down that distinction and prove
the payload's presence check fires on the broken shape.
"""

import pytest

from tests.utils.payloads import KvEventMetricsPayload
from tests.utils.prometheus import find_metric_samples, sum_metric_samples

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

ZMQ_EVENTS = "dynamo_component_kv_publisher_zmq_events_total"
DROPPED_EVENTS = "dynamo_component_kv_publisher_engines_dropped_events_total"

# Labels the runtime auto-injects on every component metric. `worker_id` is the
# one that collided with the variable label in the original bug.
_AUTO_LABELS = (
    'dynamo_namespace="dynamo",dynamo_component="backend",'
    'dynamo_endpoint="generate",worker_id="7f3a1c"'
)


def _zmq_event_lines(*, received: int = 4, accepted: int = 4) -> str:
    return (
        f"# HELP {ZMQ_EVENTS} Total number of ZMQ KV events seen by the relay\n"
        f"# TYPE {ZMQ_EVENTS} counter\n"
        f'{ZMQ_EVENTS}{{{_AUTO_LABELS},stage="received",event_type="stored"}} {received}\n'
        f'{ZMQ_EVENTS}{{{_AUTO_LABELS},stage="accepted",event_type="stored"}} {accepted}\n'
    )


def _dropped_event_lines(value: int = 0) -> str:
    return (
        f"# HELP {DROPPED_EVENTS} Total number of raw events dropped by engines\n"
        f"# TYPE {DROPPED_EVENTS} counter\n"
        f"{DROPPED_EVENTS}{{{_AUTO_LABELS}}} {value}\n"
    )


def _payload() -> KvEventMetricsPayload:
    return KvEventMetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        settle_seconds=0.0,
    )


# ── find_metric_samples vs sum_metric_samples ──────────────────────────────


def test_find_metric_samples_distinguishes_absent_from_zero():
    """The reason the helper exists: 0.0 is ambiguous, an empty list is not."""
    registered_at_zero = _dropped_event_lines(value=0)
    absent = _zmq_event_lines()

    # Both summed values are 0.0 — indistinguishable.
    assert sum_metric_samples(registered_at_zero, DROPPED_EVENTS) == 0.0
    assert sum_metric_samples(absent, DROPPED_EVENTS) == 0.0

    # find_metric_samples tells them apart.
    assert find_metric_samples(registered_at_zero, DROPPED_EVENTS) == [0.0]
    assert find_metric_samples(absent, DROPPED_EVENTS) == []


def test_find_metric_samples_ignores_help_and_type_lines():
    """A `# TYPE` line alone must not read as a registered sample."""
    comments_only = (
        f"# HELP {DROPPED_EVENTS} Total number of raw events dropped by engines\n"
        f"# TYPE {DROPPED_EVENTS} counter\n"
    )
    assert find_metric_samples(comments_only, DROPPED_EVENTS) == []


def test_find_metric_samples_filters_on_label_subset():
    content = _zmq_event_lines(received=4, accepted=3)

    assert find_metric_samples(
        content, ZMQ_EVENTS, {"stage": "received", "event_type": "stored"}
    ) == [4.0]
    assert find_metric_samples(
        content, ZMQ_EVENTS, {"stage": "accepted", "event_type": "stored"}
    ) == [3.0]
    assert (
        find_metric_samples(content, ZMQ_EVENTS, {"stage": "nonexistent-stage"}) == []
    )


# ── KvEventMetricsPayload.validate ─────────────────────────────────────────


def test_validate_passes_when_dropped_counter_registered_at_zero():
    """The healthy shape: gap counter exposed, never incremented."""
    content = _zmq_event_lines() + _dropped_event_lines(value=0)
    _payload().validate(None, content)


def test_validate_fails_when_dropped_counter_absent():
    """The regression shape: ZMQ counters fine, gap counter never registered.

    Without the presence check this content passes, because the only signal is a
    metric that is missing rather than wrong.
    """
    content = _zmq_event_lines()
    with pytest.raises(AssertionError, match="is absent from /metrics"):
        _payload().validate(None, content)


def test_validate_still_fails_on_missing_zmq_events():
    """The pre-existing value assertions are unchanged by the new check."""
    content = _dropped_event_lines(value=0)
    with pytest.raises(AssertionError, match="received KV events"):
        _payload().validate(None, content)
