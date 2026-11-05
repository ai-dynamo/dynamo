<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SWE-bench Pro Trace Benchmarking

Replay recorded SWE-bench Pro agent trajectories with AIPerf 0.13 to measure
serving throughput and latency. This benchmark does not run repository tasks or
measure task-resolution accuracy.

## Prepare the traces

`traces/swe-bench-pro-trace.tar.gz` contains 400 JSON trajectories in AIPerf's
`weka_trace` format, under `swe-bench-pro__super-3dot5__stacked/`.
The records contain `models`, `requests`, `totals`, `block_size`, and
`hash_id_scope`. The archive uses Super 3.5 formatting and block size 64.
Keep it unchanged when reproducing a run.

Install Git LFS, then run these commands from the repository root:

```bash
git lfs pull --include="recipes/nemotron-3.5-super-vl-120b-a12b/perf/traces/*.tar.gz"
export TRACE_ARCHIVE="$PWD/recipes/nemotron-3.5-super-vl-120b-a12b/perf/traces/swe-bench-pro-trace.tar.gz"
python3 - <<'CHECKSUM'
import hashlib
import os
from pathlib import Path

archive = Path(os.environ["TRACE_ARCHIVE"])
with archive.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
assert actual == "7be7eb6716d743601a8a308948730c1e07409be2e799c5e0b1fffbe388e2e1a4", actual
CHECKSUM
mkdir -p /tmp/swebench-traces
tar -xzf "$TRACE_ARCHIVE" -C /tmp/swebench-traces
export TRACE_DIR=/tmp/swebench-traces/swe-bench-pro__super-3dot5__stacked
```

A checksum mismatch can mean that Git LFS did not download the archive or that
the archive differs from this version.

## Run the benchmark

Copy the extracted directory to the `shared-model-cache` PVC at
`/shared-model-cache/traces/swe-bench-pro__super-3dot5__stacked`.
If you use another path, update `TRACE_DIR` in [perf.yaml](perf.yaml).
The Job uses the same cached tokenizer as the serving workers and needs network
access to install AIPerf. The checksum command requires Python 3.11 or later.

Set the empty Job environment values in `perf.yaml`: `ENDPOINT`, `CONCURRENCY`,
`BENCHMARK_DURATION`, `WARMUP_GRACE_PERIOD`, `REQUEST_TIMEOUT_SECONDS`,
`MAX_CONTEXT_LENGTH`, `ARTIFACT_DIR`, and `CACHE_SALT`.
Use seconds for durations and timeouts, and tokens for the context limit.
Use a new artifact directory and cache salt for each run.
Set `NAMESPACE` to your deployment namespace, then apply the Job:

```bash
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/perf/perf.yaml -n "${NAMESPACE}"
kubectl logs -f job/swe-bench-pro-aiperf-job -n "${NAMESPACE}"
```
