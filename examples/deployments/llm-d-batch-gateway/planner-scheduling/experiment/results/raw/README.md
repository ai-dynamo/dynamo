<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Raw Run Artifacts

Harness executions write a new
`YYYYMMDDTHHMMSSZ-{baseline|planner-controlled|planner-native}-<suffix>/`
directory. Controller executions write
`YYYYMMDDTHHMMSSZ-planner-loop-<suffix>/`. Never edit or reuse a prior run
directory. Harness runs contain human-readable and JSON metadata, captured
stdout and stderr, the exact submitted workload, Batch API responses, progress
observations, optional online-request observations, and read-only Kubernetes
evidence. Native Planner runs additionally prove the expected Planner pod and
ConfigMap were present and preserve recurring in-run decision-log evidence.
Controller runs contain one immutable JSONL decision record per iteration.

An autonomous scale-from-zero treatment may also contain
`autonomous-scale-evidence/`. That directory is captured across a declared T0
through T1 boundary and contains before/after DGD and DGDSA objects, a
continuous DGDSA watch, periodic DGD/worker/Redis observations, Planner logs,
Async counter snapshots, terminal Redis state, and the exact read-only observer
scripts used. `assertions.json` is derived by
`../../workloads/verify_native_planner_e2e.py`; it must pass before the run is called
canonical. Establishing a pre-run replica condition must be recorded separately
and may not occur inside T0-T1.

Retries use a new run identifier and record the reason in the session worklog.
The raw tree is ignored by Git; reports must label it local-only and retain a
tracked sanitized/checksummed evidence manifest for handoff.
