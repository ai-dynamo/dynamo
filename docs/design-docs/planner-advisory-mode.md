---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Advisory Mode
---

# Planner Advisory Mode

## Overview

Advisory Mode allows the Dynamo Planner to compute and log scaling recommendations
**without executing them**. This is designed for operators who want to validate
planner behavior on production workloads before enabling automatic scaling.

## Motivation

Customers deploying Planner in production consistently report the same concern:

> "We want to observe what the planner would do before letting it control our GPU fleet."

Advisory mode addresses this by providing a zero-risk observation window where
operators can:

1. Validate that scaling recommendations are reasonable
2. Understand the relationship between FPM data, regression models, and scaling decisions
3. Build confidence before enabling auto-scaling

## Architecture

Advisory mode is implemented in the **adapter layer** (`NativePlannerBase`),
not in the state machine. The `PlannerStateMachine` runs identically in all
modes — it always computes `ScalingDecision`. The difference is what happens
after:

```
PlannerStateMachine.on_tick()
    → PlannerEffects { scale_to, diagnostics }
        ↓
    [ScalingMode check in NativePlannerBase.run()]
        ├── ACTIVE:   _apply_effects() → connector.set_replicas()
        ├── ADVISORY: _log_advisory_decision() → log + Prometheus
        └── NOOP:     skip
```

This design ensures:
- The state machine remains pure (zero I/O, no mode awareness)
- Advisory and active modes produce identical decision logic
- Switching modes requires only a config change, no code change

## Configuration

```yaml
# In planner config (JSON or YAML):
scaling_mode: advisory          # "active" | "advisory" | "noop"
advisory_log_interval: 60      # seconds between advisory log entries
```

## Observability

### Logs

Advisory decisions are logged at INFO level with `[ADVISORY]` prefix:

```
[ADVISORY] SCALE_UP | current: prefill=2 decode=4 |
  recommended: prefill=3 decode=6 (delta: +1 / +2) |
  load_reason=scale_up throughput_reason=scale |
  est_ttft=120.5ms est_itl=32.1ms
```

### Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `dynamo_planner_advisory_recommended_replicas` | Total recommended replicas (not executed) |
| `dynamo_planner_advisory_current_replicas` | Current replica count |

All existing diagnostics metrics (`dynamo_planner_estimated_ttft_ms`,
`dynamo_planner_load_scaling_decision`, etc.) continue to work in advisory mode.

### HTML Reports

The `DiagnosticsRecorder` HTML reports (introduced in PR #8078) work
identically in advisory mode. Configure with:

```yaml
report_interval_hours: 1.0
report_output_dir: /tmp/planner_reports
```

## Transition Path

1. Deploy planner with `scaling_mode: advisory`
2. Monitor logs and Grafana dashboards for 24-48 hours
3. Validate recommendations against expected behavior
4. Switch to `scaling_mode: active` to enable auto-scaling

## Related PRs

- PR #7961: Unified FPM regression models (the decision logic advisory mode exposes)
- PR #8046: State machine extraction (the architecture advisory mode builds on)
- PR #8078: Diagnostics metrics and HTML reports (complementary observability)
