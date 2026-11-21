<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tracked POC Evidence

Raw and compiled experiment directories are intentionally ignored by Git. This
directory retains small, sanitized, reviewable fixtures that let a fresh
checkout understand the report and replay its fixture assertions.

- `20260828-canonical-controlled-pair.json` is the bidirectional pairing
  manifest for the canonical controller and workload runs. It records outcome
  rollups and SHA-256 digests of the local raw artifacts.
- `20260828-canonical-native-planner.json` is the canonical native control-plane
  manifest. It records the pre-run zero-worker setup, mutation-free experiment
  driver window, worker DGDSA `0 -> 1`, readiness-gated `0 -> 5 -> 0` admission,
  100/100 result, terminal worker/floor/lease state, validation rollups, and
  SHA-256 digests of the ignored raw and compiled artifacts.
- `20260919-planner-gym-batch-impact.json` is the reviewable rollup for the
  decision-ready CPU-Mocker impact campaign. It records all 15 run IDs, frozen
  workload and deployment signatures, per-arm aggregates, paired effects,
  native control-check totals, stopping assessment, and local artifact hashes.
- `20260828-local-preflight/` retains the four small, sanitized inputs linked by
  the local preflight report.
- `20260828-native-planner-verifier/` is a minimized, sanitized fixture derived
  from the canonical run. It contains every input consumed by
  `verify_native_planner_e2e.py`, so the recorded assertions can be replayed
  from a fresh checkout without the ignored raw-results tree.
- The exact successful port forwards, controller invocation, and workload
  invocation are preserved in the
  [workload guide](../../workloads/README.md#canonical-controlled-run).
- Async image source/build reconstruction is recorded in
  [the provenance note](../../research/20260828-async-image-provenance.md).

The manifests and verifier fixture are sanitized syntheses, not replacements
for the raw run and not proof of the historical deployed binary. A reviewer who
has the ignored artifacts can recompute every listed digest. The
standalone-controlled manifest records post-expiry Redis state, but no request
was submitted after expiry. The native manifest records a fresh terminal zero
lease; Async's idle drain gauge is explicitly last-evaluated rather than
authoritative.
