---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Audit Events
---

The Planner emits structured audit events through the
`dynamo.planner.audit` logger. Each event is a JSON object preceded by
the literal string `AUDIT ` so log forwarders can grep / filter on the
prefix without parsing every line.

These complement the [Prometheus metrics](observability.md): metrics
give counts and rates, audit events give per-decision context (which
plugin, which decision_id, which tick) for forensic analysis and
replay.

## Why audit events live in their own logger

`dynamo.planner.audit` is a dedicated logger so operators can:

- Tee the audit stream to a long-term store (S3 / object storage) for
  post-hoc analysis without retaining the rest of the planner's verbose
  logs
- Pin a separate log level on the audit channel — audit events ALWAYS
  emit at INFO regardless of root planner log level, so a debug-noisy
  cluster doesn't drown them
- Pre-filter on the literal `AUDIT ` prefix before any JSON parsing,
  cutting downstream log-pipeline cost

## Common fields

Every event carries these context keys (set to `null` when not
applicable rather than omitted, so downstream joins on these keys
never silently lose rows):

| Field | Type | Meaning |
|---|---|---|
| `event` | string | Event name from the catalog below |
| `tick_id` | string \| null | Identifier for the tick the event belongs to |
| `decision_id` | string \| null | Identifier for the scaling decision being audited |
| `plugin_id` | string \| null | Plugin emitting / referenced by the event |

Additional event-specific fields are documented per-event below.

## Event catalog

The catalog is defined as `AuditEvent` (a `str`-backed enum) in
`dynamo.planner.plugins.audit`. Adding a new event requires
appending to the enum and updating this page — never silently rename
or remove an event, since dashboards and replay assertions key off
these names.

### Plugin lifecycle events

| Event | When | Extra fields |
|---|---|---|
| `plugin_evaluated` | After a plugin call completes (success or interpretable failure) | `stage`, `result`, `latency_ms` |
| `plugin_degraded` | Plugin failed in a tolerated way (e.g. result held over from cache) | `stage`, `failure` |
| `plugin_timeout` | Plugin call exceeded `request_timeout_seconds` | `stage`, `timeout_s` |
| `plugin_circuit_open` | Circuit breaker transitioned to OPEN after repeated failures | `failure_count`, `cooldown_s` |
| `plugin_rejected` | Plugin returned `RejectResult`; stage will short-circuit | `stage`, `reason` |

### EXECUTE stage events

| Event | When | Extra fields |
|---|---|---|
| `execute_invoked` | `connector.set_component_replicas(...)` was called | `targets` |
| `execute_succeeded` | Connector returned without error | `latency_ms` |
| `execute_failed` | Connector raised or returned ERROR | `error`, `latency_ms` |
| `execute_skipped_rejected` | Final decision was REJECT; execute not attempted | `rejecting_plugin` |
| `execute_skipped_no_change` | Final targets equal current workers | — |
| `execute_advisory` | Planner running advisory-only; decision logged not applied | `targets` |
| `execute_in_progress` | Previous scaling still running; this tick deferred | — |

### Multi-cadence scheduling

| Event | When | Extra fields |
|---|---|---|
| `tick_skipped` | Plugin skipped because its execution_interval hasn't elapsed | `stage`, `last_call_at`, `due_at` |
| `tick_timeout` | Whole tick exceeded `tick_max_duration_seconds` | `elapsed_s`, `deadline_s` |

### Cross-cutting

| Event | When | Extra fields |
|---|---|---|
| `global_scale_request_rejected` | GlobalPlanner refused the scale request | `caller_namespace`, `reason` |
| `plugin_constrain_set_dropped` | A CONSTRAIN-stage plugin returned SET (forbidden); orchestrator silently dropped | `key` |
| `orchestrator_drift_detected` | Reserved (PR 5 v1.2 removed dual-execution; kept for enum stability) | — |

## Example event lines

```
AUDIT {"event":"plugin_evaluated","tick_id":"t-12","decision_id":"d-42","plugin_id":"builtin_load_propose","stage":"propose","result":"set","latency_ms":3.2}
AUDIT {"event":"execute_invoked","tick_id":"t-12","decision_id":"d-42","plugin_id":null,"targets":[{"sub_component_type":"prefill","replicas":3}]}
AUDIT {"event":"execute_succeeded","tick_id":"t-12","decision_id":"d-42","plugin_id":null,"latency_ms":42.1}
AUDIT {"event":"plugin_circuit_open","tick_id":"t-15","decision_id":null,"plugin_id":"buggy_user_plugin","failure_count":5,"cooldown_s":30}
```

## Suggested log queries

**All events for a single decision_id**:
```bash
grep '"decision_id":"d-42"' planner.log
```

**Plugin error rate per plugin in the last hour** (using `jq`):
```bash
grep '^AUDIT ' planner.log \
  | sed 's/^AUDIT //' \
  | jq -r 'select(.event == "plugin_evaluated" and .result == "error") | .plugin_id' \
  | sort | uniq -c
```

**Tick deadline breaches** (root cause via plugin latency in same tick):
```bash
grep '"event":"tick_timeout"' planner.log | jq .tick_id | while read tick; do
  echo "=== tick $tick ==="
  grep "\"tick_id\":$tick" planner.log
done
```

## Adding a new event

1. Append a new `AuditEvent.<NAME>` value in
   `components/src/dynamo/planner/plugins/audit.py` (insertion order is
   preserved; never re-order or rename existing values)
2. Update the test
   `tests/plugins/test_audit_logger.py::test_event_catalog_matches_expected_set`
   so the catalog regression guard catches future drift
3. Add a row to the catalog in this file under the appropriate section
4. Emit the event from the originating call site:

   ```python
   from dynamo.planner.plugins.audit import AuditEvent, AuditLogger

   audit = AuditLogger()
   audit.emit(
       AuditEvent.MY_NEW_EVENT,
       tick_id=ctx.tick_id,
       decision_id=ctx.decision_id,
       plugin_id=plugin.plugin_id,
       my_extra_field=...,
   )
   ```

## See also

- [Observability metrics reference](observability.md) — Prometheus
  metric catalog
- `plugins/audit.py` source — `AuditEvent` enum + `AuditLogger`
  implementation
