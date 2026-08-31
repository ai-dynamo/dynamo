<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone Batch Planner Control-Loop Runner

- Date: 2026-08-28
- Implementation: [batch_planner_control_loop.py](../workloads/batch_planner_control_loop.py)
- Usage: [experiment workloads](../workloads/README.md#standalone-planner-control-loop)
- Result: implementation and hermetic validation passed

## Implemented Scope

The runner collects live Batch Gateway jobs and llm-d Async Prometheus feedback,
adds static online offered load and ready replica inputs, evaluates the public
single-pool batch policy, and writes one audit record per bounded iteration.
Records contain observations, the advisory replica floor, the leased drain
decision, diagnostics, actuation state, and sanitized failures.

Dry run is the default and never constructs a Redis actuator. Apply mode requires
an explicit flag, Redis URL, and exact control key. It renews the policy lease
each iteration. Replica scaling is outside this POC and is never invoked.

The controlled deployment applies `redis-leased-rate` at the
`ap.workerPools[0]` admission boundary for pool `dynamo-batch`, using Redis key
`llm-d-async:drain-limit:dynamo-batch`. It does not install the gate in
`queuesConfig`; the observation pool, policy pool, and actuation key therefore
refer to the same worker-pool boundary.

Batch job observation is tenant-local, through `X-MaaS-Username`, while the
leased cap is worker-pool-global. The runner defaults to the baseline workload's
`planner-poc-baseline` tenant. Mixed tenants sharing `dynamo-batch` would make a
tenant-local demand observation unsafe for a global cap and are explicitly out
of scope for this POC.

The Prometheus URL must be an actual Prometheus HTTP API with
`/api/v1/query`; llm-d Async's raw `:9090/metrics` endpoint is not a valid
controller target. Both usage examples point to a distinct local port intended
for a port-forward to the cluster Prometheus service.

## Safety Behavior

- The interval must be shorter than the decision lease.
- Policy decisions are validated for pool, rate, expiry, decision identity, and
  advisory replica-floor shape before an apply is possible.
- Collection and policy failures in apply mode trigger one best-effort pause
  lease, are recorded, and exit nonzero.
- Dry-run failures never invoke an actuator.
- An actuation failure exits nonzero without a second mutation. Previously
  published leases expire into dispatcher fail-closed behavior.
- Output files use exclusive creation, contain no endpoint or Redis fields, and
  sanitize common credential shapes.

## Validation Evidence

The 17-test hermetic suite passed in 0.10 seconds. It covers dry-run behavior,
periodic renewal, unique decision IDs, collection and policy fail-closed pauses,
failed pause preservation, actuation failure, invalid policy output, strict JSON,
credential redaction, explicit nonzero startup failure, CLI gating, unsafe URL
rejection, and output immutability. Ruff lint/format and Python syntax checks
passed. After adding the trusted request-count handoff, the combined control-loop
and baseline experiment suites passed all 37 tests in 0.41 seconds.

No live endpoints or Redis instance were used for this controller test suite.
Apply mode additionally needs the optional `redis` Python package; the usage
guide installs it into the selected Dynamo `.venv` before invoking that
environment's Python directly.

Before rollout, confirm Prometheus is scraping the controlled Async pods and
returns every source metric required by `LlmdAsyncPrometheusSource`. The target
cluster's CRD and cross-namespace selectors were inspected, and the controlled
overlay enables a 5-second PodMonitor. The later canonical controlled run
verified that target live before applying a drain decision; see
[the controlled-run report](20260828-controlled-live.md).
