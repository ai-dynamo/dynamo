---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: ThunderAgent Scheduler Reference
subtitle: Control-loop flags, constraints, capacity behavior, and scheduler log records
---

<Warning>
`dynamo.thunderagent_router` is experimental. The command-line interface and lifecycle contract can change.
</Warning>

The ThunderAgent scheduler uses the program working set as a fraction of each worker's retention budget. With SGLang HiCache enabled, the budget is the sum of published GPU KV capacity and host HiCache capacity. Mooncake capacity is excluded because it is conditional content-addressed storage rather than guaranteed per-program retention.

## Control-Loop Configuration

| Flag | Environment variable | Default | Description |
|---|---|---|---|
| `--pause-threshold` | `DYN_THUNDERAGENT_PAUSE_THRESHOLD` | `0.95` | Working-set fraction of the retention budget that starts a pause cycle. |
| `--soft-demote-threshold` | `DYN_THUNDERAGENT_SOFT_DEMOTE_THRESHOLD` | `0.80` | Start of the soft-demote band below the pause threshold. |
| `--pause-target` | `DYN_THUNDERAGENT_PAUSE_TARGET` | `0.80` | Setpoint that a pause cycle drives utilization toward. |
| `--resume-hysteresis` | `DYN_THUNDERAGENT_RESUME_HYSTERESIS` | `0.10` | Headroom below `pause-threshold` required before resume. |
| `--resume-priority-boost` | `DYN_THUNDERAGENT_RESUME_PRIORITY_BOOST` | `1.0` | Transient priority seconds added to a resumed request. |
| `--resume-timeout-seconds` | `DYN_THUNDERAGENT_RESUME_TIMEOUT_SECONDS` | `1800.0` | Forced-resume cap that bounds starvation. |
| `--scheduler-interval-seconds` | `DYN_THUNDERAGENT_SCHEDULER_INTERVAL_SECONDS` | `5.0` | Scheduler tick period. |
| `--soft-demote-priority-jump` | `DYN_THUNDERAGENT_SOFT_DEMOTE_PRIORITY_JUMP` | `-2.0` | Priority seconds applied to soft-demoted programs. |
| `--acting-token-weight` | `DYN_THUNDERAGENT_ACTING_TOKEN_WEIGHT` | `1.0` | Multiplier on `token_total` for acting programs in the pause-side working set. |
| `--acting-decay-tau-seconds` | `DYN_THUNDERAGENT_ACTING_DECAY_TAU_SECONDS` | `1.0` | Exponential-decay time constant for acting tokens in the resume-side working set. |

All `KvRouter` flags accepted by `dynamo.router`, including `--router-temperature`, `--use-kv-events`, and `--router-track-output-blocks`, are also accepted and forwarded.

## Constraints

The service rejects configurations that violate any of these relationships:

```text
0 <= pause_threshold <= 1
0 <= pause_target <= pause_threshold
0 <= resume_hysteresis <= pause_threshold
0 <= soft_demote_threshold <= pause_threshold
acting_token_weight > 0
scheduler_interval_seconds > 0
resume_timeout_seconds > 0
```

The control loop has three bands:

- At or above `pause-threshold`, pause acting programs until utilization reaches `pause-target`.
- From `soft-demote-threshold` up to `pause-threshold`, lower program priority without pausing.
- Resume only after utilization falls at least `resume-hysteresis` below `pause-threshold`.

## Session Inputs

Programs are keyed by the normalized `session_id`. Use the canonical headers defined in [Session IDs](../../use-cases/agents/session-ids.mdx):

- `X-Dynamo-Session-ID`
- `X-Dynamo-Parent-Session-ID`
- `X-Dynamo-Session-Final`

Requests without session identity bypass program admission and pause/resume.

## Scheduler Logs

The scheduler emits pause-side and resume-side summaries at INFO level.

```text title="Pause-side summary"
scheduler.tick worker=<id> paused=<N> marked=<M> util=<X> -> <Y>
```

- `paused`: acting programs paused during the tick
- `marked`: reasoning programs marked to pause at the next tool boundary
- `util`: worker utilization before and after the pause cycle

```text title="Resume-side summary"
scheduler.tick resumed=<N> still_paused=<M>
```

- `resumed`: programs resumed during the tick
- `still_paused`: programs remaining in the paused table

Lower the log level for `dynamo.thunderagent_router` to DEBUG for per-program records:

```text
Paused program <program_id> (tokens=<n>)
Resumed program <program_id> -> worker=<id> (tokens=<n>)
```

For request-level tracing, enable `DYN_REQUEST_TRACE=1` on the frontend. Explicit harness tool spans additionally require `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` and a configured publisher. See [Request Trace Reference](../observability/request-traces.mdx).

## Related Pages

- Run the ThunderAgent Scheduler
- [ThunderAgent Scheduler Design](../../use-cases/agents/thunderagent-program-scheduler.md)
- [Priority Scheduling](../../use-cases/agents/priority-scheduling.md)
