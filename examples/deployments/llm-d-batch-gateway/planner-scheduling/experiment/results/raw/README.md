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

Planner Gym campaign cells use
`YYYYMMDDTHHMMSSZ-planner-gym-batch-impact-<suffix>/`. Their
`metadata.json` is schema version `1.0` with
`kind=planner-gym-batch-impact`. Treat these fields as the run-level contract:

- `status`, `exit_code`, and `error` record the outer runner outcome;
- `arm` identifies `online-only`, `stock`, or `planner-native`;
- `configuration` records endpoints, model, tenant, trace settings, expected
  gate types, and Batch settings;
- `source` records the Dynamo and Planner Gym roots, revisions, dirty state,
  and dirty paths;
- `inputs` records paths, byte counts, and SHA-256 values for the runner,
  Planner Gym script, Batch harness, generated Match Config, endpoint catalog,
  trace, and optional Batch dataset;
- `synchronization` records Batch creation/progress and the Planner Gym start
  boundary;
- `children`, `metrics_summary`, and optional `remote_cleanup` record child
  exit codes, continuous metric capture, and abort cleanup.

`config/` contains the exact generated endpoint catalog and Match Config.
`planner-gym/results.json` is the one-cell native Planner Gym result, while
`planner-gym/report.html` is its presentation. `logs/events.jsonl` preserves
the outer lifecycle. `source-state/` contains the commands and outputs used to
derive both source records. Batch arms also contain the nested
`batch-harness/results/raw/<run-id>/` evidence tree. `artifact-checksums.json`
is the integrity manifest for reportable files; verify it before compiling or
copying a run.

Campaign orchestration can keep an attempt index outside the individual run
directories. Preserve every attempted run in chronological order, including
failed attempts, when constructing the compiler's ordered `--run-directory`
arguments. The compiler uses per-arm chronological ordinal pairing and must not
silently skip an attempt.

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
