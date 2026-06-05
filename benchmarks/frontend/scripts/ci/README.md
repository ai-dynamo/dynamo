<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend Perf PR Guard

`frontend_perf_pr_guard.py` is a small CI wrapper around the existing frontend
benchmark sweep. It is intended to catch large frontend regressions before a PR
merges.

The guard partitions the runner's allowed CPU set deterministically:

- the frontend is pinned to the first allowed CPU;
- mocker, aiperf, etcd, and NATS are pinned to all remaining allowed CPUs.

The default benchmark uses the in-repo TinyLlama tokenizer fixture, materialized
under the run output with a minimal chat template, so the run is offline and
does not depend on HuggingFace cache state.

## Local Dry Run

```bash
python3 benchmarks/frontend/scripts/ci/frontend_perf_pr_guard.py run --dry-run
```

## CI Run

```bash
python3 benchmarks/frontend/scripts/ci/frontend_perf_pr_guard.py run \
  --output-dir /tmp/frontend-perf-pr-guard \
  --baseline /path/to/runner-specific-baseline.json
```

Without `--baseline`, the guard runs in sanity-only mode and checks that the run
completed with non-zero core metrics. With a baseline, it fails on the default
coarse regression thresholds:

- request throughput drops more than 15%;
- output token throughput drops more than 15%;
- TTFT p50 rises more than 25%;
- TTFT p99 rises more than 35%;
- ITL p50 rises more than 20%;
- ITL p99 rises more than 35%.

## Create A Baseline Candidate

Run on the same runner class that will execute the PR guard:

```bash
python3 benchmarks/frontend/scripts/ci/frontend_perf_pr_guard.py run \
  --output-dir /tmp/frontend-perf-pr-guard \
  --export-baseline /tmp/frontend-perf-pr-guard-baseline.json
```

Baseline files are CPU and runner dependent. Keep separate baselines for
different runner labels or CPU allocations.
