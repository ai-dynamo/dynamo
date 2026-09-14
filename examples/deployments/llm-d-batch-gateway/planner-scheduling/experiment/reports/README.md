<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reports

- [2026-08-28 local preflight](20260828-local-preflight.md) - Harness tests and
  deterministic workload preparation without cluster, API, or traffic access.
- [2026-08-28 control-loop runner](20260828-control-loop-runner.md) - Standalone
  observation, policy, leased actuation, and fail-closed runner implementation.
- [2026-08-28 stock live baseline](20260828-stock-live-baseline.md) - Live
  100-request result, throughput, provenance, and Gateway compatibility findings.
- [2026-08-28 Planner-controlled live run](20260828-controlled-live.md) - Live
  5-RPS leased admission result, stock comparison, and lease-expiry evidence.
- [2026-08-28 native Planner autonomous E2E](20260828-native-planner-e2e.md) -
  Worker DGDSA scale from zero, readiness-gated admission, 100/100 result, and
  terminal zero-lease evidence.
- [2026-09-19 Planner Gym Batch impact](20260919-planner-gym-batch-impact.md) -
  Decision-ready 15-run CPU-Mocker campaign comparing online-only, stock Batch
  drain, and native Planner-controlled drain under a frozen 5/8/5-RPS trace.
- [Tracked canonical evidence](evidence/README.md) - Bidirectional run pairing,
  sanitized outcome rollups, local-artifact checksums, and scope limitations.

Raw and compiled run trees are local and ignored by Git. Quantitative claims
link to a tracked sanitized manifest containing their local-artifact checksums.
