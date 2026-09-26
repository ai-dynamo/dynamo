<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 H200 EAGLE3 Benchmark

One AIPerf Job runs c16 → c32 → c64 → c128 against an existing deployment.
Both configurations use 32 H200 GPUs, FP8 weights/KV cache, EAGLE3 k=3 with
real acceptance, KV-aware routing, and a 262,144-token context limit.
If model initialization runs out of memory, lower `--max-model-len` to 32768
on every worker and record that change with the results.

| Target | Configuration | Sequence Limit per Engine |
| --- | --- | --- |
| `agg` | 4 × TP8 + expert parallelism | 64 |
| `disagg` | 2 prefill × TP8/DP1 + 2 decode × TP8, expert parallelism | 32 prefill / 64 decode |

## Run

Populate the shared `model-cache` PVC, deploy the selected H200 manifest,
and verify the endpoint and worker KV events. Set these three variables in
`perf.yaml`:

| Variable | Value |
| --- | --- |
| `ARM` | `agg` or `disagg` |
| `ENDPOINT` | Selected frontend `host:port` |
| `WORKLOAD` | `w1` (default) or `w2` |

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply -f recipes/ax-k2/perf/h200/perf.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait \
  --for=condition=Complete job/axk2-h200-perf --timeout=21600s
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs job/axk2-h200-perf -c aiperf
```

The Job needs no Kubernetes API permissions, creates no other Jobs, and does
not deploy or reset serving workers. The PVC must be writable by UID 1000.
On a 32-GPU cluster, benchmark one topology, replace the serving deployment,
then rerun this Job with the other endpoint. Delete the completed benchmark
Job before applying it again.

## Workload and Run Order

| Workload | Input Tokens | Output Tokens |
| --- | --- | --- |
| W1 | 32 shared prefixes × 10,240 + 6,144 unique tokens/request | 1,024 |
| W2 | 16,384 unique tokens/request, no configured shared prefix | 1,024 |

Each Job runs 64 W2 warmup requests (seed 43), then c16, c32, c64, and c128
(seed 42). c16–64 each send 1,024 measured requests once; c128 sends 2,048
requests three times. Outputs are fixed at 1,024 tokens. Any failed request,
wrong output length, or incomplete run fails the Job and stops the sweep.

Serving caches persist between points and repeats; identical prompts can be
reused from earlier points. Use the same starting cache state and run order
for both topologies. This sequential sweep differs from the historical
per-point cold-cache experiment; its results are a separate measurement series.

## Results

Artifacts are saved under
`/model-cache/perf/axk2/h200-eagle3/<arm>/<workload>/<pod-uid>/`:

- `warmup/`: readiness warmup reports.
- `c16/r1/`, `c32/r1/`, `c64/r1/`, `c128/r1/` through `r3/`: raw AIPerf reports,
  `inputs.json`, `run.txt`, and validated `summary.json`.
- `summary.csv`: all four concurrency points; c128 values average the three runs.

Compare system output tok/s, output tok/s/GPU, user output tok/s (mean),
TPOT p50/p99, and TTFT p50. Retain the serving manifest and Job YAML with the
reports, and verify matching input hashes for paired topology runs.
TPOT is AIPerf's request-average `inter_token_latency`; p99 is across request
averages, not individual token gaps. Mean user throughput is mean(1000/TPOT_ms).
