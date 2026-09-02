<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local Baseline Harness Preflight

- Date: 2026-08-28
- Run ID: `20260828T123305Z-baseline-23b789`
- Scope: local workload and artifact-path validation only
- Result: passed

## Outcome

The harness resolved the immutable GSM8K test input, prepared the default
100-request slice, finalized a raw run with exit code zero, and compiled that
run successfully. The preflight explicitly skipped Kubernetes and API access,
and its machine-readable summary records `traffic_created=false`.

## Evidence

- [Raw run metadata](../results/raw/20260828T123305Z-baseline-23b789/metadata.md)
- [Raw preflight summary](../results/raw/20260828T123305Z-baseline-23b789/preflight-summary.json)
- [Workload manifest](../results/raw/20260828T123305Z-baseline-23b789/workload-manifest.json)
- [Compiled summary](../results/compiled/20260828T123305Z-baseline-23b789-summary/summary.json)

The source dataset SHA-256 is
`fc56d28e5522056856c064181dfe841d3fe927ea2f4f605f24b9fbe81db06fd0`.
The normalized 100-request JSONL SHA-256 is
`a75148f503f5518b00463a708ef79c67470857abb2c390a1bc8053de67f5dada`.

The compiled result has no Batch progress or online-request samples, as
expected for a local preflight. It marks both inputs missing and does not impute
measurements.

## Validation

- 14 hermetic pytest tests passed.
- Ruff lint and formatting checks passed.
- Python compilation and Bash syntax checks passed.
- The shell launcher preserves the Python harness exit code.
- Raw run directories and compiled directories refuse reuse or overwrite.
- Relative Markdown links resolve, and all source/documentation files have SPDX
  headers.
- A credential-shape scan found no Hugging Face credential in the raw or
  compiled run.

## Limitation and Next Step

This run says nothing about live gateway availability, dispatch behavior,
throughput, or online interference. No live traffic should be submitted until
the Redis/Async/frontend/worker stack is explicitly verified. Then run the
documented read-only live preflight before creating a Batch job.
