<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experiment Index

## Current

- [Experiment overview](README.md) - Goal, scope, criteria, and current status.

## Research

- [Baseline inputs](research/20260828-baseline-inputs.md) - Existing benchmark,
  Dynamo example, and GSM8K workload findings.
- [Gateway deadline/count compatibility](research/20260828-batch-gateway-deadlines.md) -
  Public-schema fallback, trust boundary, and supported POC scope.
- [Controlled stack](research/20260828-controlled-stack.md) - Leased gate,
  stable identifiers, PodMonitor, and rollout boundary.
- [Async image provenance](research/20260828-async-image-provenance.md) - Exact
  deployed digest, BuildKit record, source reconstruction, and limitations.

## Raw Results

- [Raw result conventions](results/raw/README.md) - Per-run evidence layout and
  immutability rules.

## Compiled Results

- [Compiled result conventions](results/compiled/README.md) - Derived summary
  layout and reproduction requirements.

## Reports

- [Report index](reports/README.md) - Preflight and live baseline reports.
- [Local preflight](reports/20260828-local-preflight.md) - Hermetic tests and a
  no-cluster, no-API dry run of the default 100-request workload.
- [Control-loop runner](reports/20260828-control-loop-runner.md) - Implemented
  scope, safety behavior, validation, and live-run prerequisites.
- [Stock live baseline](reports/20260828-stock-live-baseline.md) - 100-request
  live result, throughput, provenance, and compatibility findings.
- [Planner-controlled live run](reports/20260828-controlled-live.md) - Applied
  5-RPS lease, 100-request result, stock comparison, and deployment provenance.
- [Native Planner autonomous E2E](reports/20260828-native-planner-e2e.md) -
  Worker DGDSA `0 -> 1`, readiness-gated admission, 100/100 terminal result,
  and machine-verified terminal zero lease.
- [Tracked canonical evidence](reports/evidence/README.md) - Bidirectional run
  pairing and checksummed result rollup retained across checkouts.
