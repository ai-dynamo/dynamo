<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stock live baseline

## Outcome

Run `20260828T131444Z-baseline-9f7b71` completed all 100 deterministic
GSM8K requests through the stock constant-gate stack in a dedicated test
namespace. There were no failed requests and the compiler reported no
data-quality issues.

| Measure | Result |
| --- | ---: |
| Terminal status | `completed` |
| Requests | 100 total, 100 completed, 0 failed |
| Terminal duration | 6.180863 s |
| Average completion rate | 16.178970 RPS |
| Peak observed interval rate | 25.245067 RPS |
| Progress samples | 4 |
| Online load | disabled |

The progress samples observed 0 completions at 0 and 2.06 seconds, 48 at
4.12 seconds, and all 100 at 6.18 seconds. This is a batch-only throughput
baseline, not an online-interference result.

## Configuration and provenance

- Model: `Qwen/Qwen3-0.6B`
- Dataset: converted GSM8K test split, SHA-256
  `fc56d28e5522056856c064181dfe841d3fe927ea2f4f605f24b9fbe81db06fd0`
- Submitted slice: first 100 records, temperature 0, `max_tokens=128`, SHA-256
  `a75148f503f5518b00463a708ef79c67470857abb2c390a1bc8053de67f5dada`
- Namespace/context: dedicated POC namespace / cluster context omitted
- Gate at both start and end: `constant`
- The seven selected Gateway, Valkey, Async, frontend, and worker pods were
  Running (or Completed for the one-shot files init job) at both captures.

The immutable raw directory contains the normalized input, API responses,
progress stream, terminal object, validated output, pod/resource snapshots,
selected logs and metrics, source status, and checksums. The compiled summary
records SHA-256 hashes for every source file it consumed.

## Live findings that affect the controlled POC

The run exposed two Batch Gateway v0.3 public-schema behaviors:

1. `expires_at` is null for ordinary jobs; the internal completion-window
   deadline is not surfaced directly.
2. While a job is active, `request_counts.total` remains zero until the first
   completion or failure.

The Planner collector now derives a conservative deadline from
`created_at + completion_window`. New harness submissions attach immutable
string metadata `planner_request_count`, which the collector requires while an
active total is zero and verifies once the live total becomes positive. This
prevents a fail-closed zero-rate lease from creating a bootstrap deadlock.

## Evidence

- Raw run: `results/raw/20260828T131444Z-baseline-9f7b71/`
- Compiled result:
  `results/compiled/20260828T131444Z-baseline-9f7b71-summary/`
- Deadline/count compatibility note:
  `research/20260828-batch-gateway-deadlines.md`
