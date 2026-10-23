<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner Video Benchmark

This benchmark uses AIPerf 0.12.0 to measure streaming video inference against
the Cosmos3 Nano Reasoner Dynamo frontend. It runs 1,000 single-turn requests
at concurrency levels 1, 2, 4, 8, and 16: five questions for each of 200
30-second videos. Every request uses a fixed 128-token output length.

The manifest pins the private dataset
`jkiczka/cosmos3-benchmark-videos` to commit
`ce4b687c7736e29a13e00385cc8b6a6d28f355fd`. It downloads the complete dataset
snapshot to the shared PVC and selects the first 200 videos under `2fps-30s` in
sorted path order.

## Run the benchmark

The target deployment must be ready, `shared-model-cache` must be mounted in
the namespace, and `hf-token-secret` must contain a read-only `HF_TOKEN`.
Choose the profile matching the deployed four-worker target.

From the repository root:

```bash
export PROFILE=round-robin # or: session-affinity
kubectl -n "$NAMESPACE" apply \
  -f "recipes/cosmos3-nano-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl -n "$NAMESPACE" delete job cosmos3-nano-reasoner-perf \
  --ignore-not-found
kubectl -n "$NAMESPACE" apply \
  -f recipes/cosmos3-nano-reasoner/perf/perf.yaml
kubectl -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-perf
kubectl -n "$NAMESPACE" wait \
  --for=condition=Complete job/cosmos3-nano-reasoner-perf \
  --timeout=7200s
```

The log ends with the artifact directory. It has one subdirectory per
concurrency level, including AIPerf JSON, CSV, per-request, and server-metric
exports:

```text
/shared-model-cache/perf/cosmos3-nano-reasoner/<profile>/<epoch>_<job-name>/
```

To copy a completed run locally while the deployment is running:

```bash
RUN_ROOT="<artifact-directory-from-the-job-log>"
export DGD="cosmos3-nano-reasoner-agg-h200-${PROFILE}"
WORKER="$(kubectl -n "$NAMESPACE" get pods \
  -l "nvidia.com/dynamo-graph-deployment-name=${DGD}" \
  -o name | grep worker | head -n 1)"
kubectl -n "$NAMESPACE" cp \
  "${WORKER#pod/}:${RUN_ROOT}" "./$(basename "$RUN_ROOT")"
```

## Interpreting the sweep

The five concurrency levels execute sequentially with stable video URLs. Each
video has five self-contained questions sharing one session ID. This gives the
router repeated requests for the same video within every concurrency level,
rather than relying only on cache state carried over from an earlier level.

Both profiles use the same deterministic JSONL rows. Every row has a stable
session ID in the same form as the earlier Cosmos benchmark:

```text
2fps-30s:<video-filename>
```

For `round-robin`, AIPerf does not send that ID as a routing header. For
`session-affinity`, it adds `--session-header X-Dynamo-Session-ID`, so every
replay of a given video carries the same header and remains pinned to one
worker while the 300-second affinity TTL is active. The two profiles otherwise
use the same four-worker deployment shape and benchmark settings.

## Run the smoke test

Use `smoke.yaml` for a fast end-to-end check before the full sweep. It
downloads `2fps-30s/vid-0000.mp4` at the pinned dataset revision, exposes that
video to the model through a temporary in-cluster Service, and validates a
non-empty chat completion:

```bash
export PROFILE=round-robin # or: session-affinity
kubectl -n "$NAMESPACE" apply \
  -f "recipes/cosmos3-nano-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl -n "$NAMESPACE" delete job cosmos3-nano-reasoner-smoke \
  --ignore-not-found
kubectl -n "$NAMESPACE" apply -f recipes/cosmos3-nano-reasoner/perf/smoke.yaml
kubectl -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-smoke
```

## Compare the routing profiles

Deploy and benchmark one profile at a time so both runs use exactly four H200
workers. Compare request throughput, output tokens/s, TTFT, TPOT, and
prompt-cache reads at each concurrency level. In pure round robin, the five
questions can land on different workers. With session affinity, all five are
pinned to the worker selected for that video.

## Measured results

The following single-run measurements were collected on four H200 workers.
Every row contains 1,000 successful requests with no errors. Output length was
fixed at 128 tokens, so request and output-token throughput scale together.

| Concurrency | Profile | Requests/s | Output tokens/s | TTFT p50 / p95 (ms) | Request latency p50 / p95 (ms) | ITL p50 / p95 (ms) | Prompt cache read |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Round robin | 1.51 | 193 | 456 / 496 | 722 / 801 | 2.18 / 2.56 | 0.2% |
| 1 | Session affinity | 2.03 | 260 | 140 / 487 | 424 / 771 | 2.15 / 2.56 | 0.1% |
| 2 | Round robin | 3.48 | 445 | 413 / 480 | 633 / 781 | 2.18 / 2.58 | 25.1% |
| 2 | Session affinity | 4.21 | 538 | 142 / 498 | 424 / 780 | 2.16 / 2.56 | 0.5% |
| 4 | Round robin | 6.56 | 839 | 432 / 559 | 680 / 865 | 2.23 / 2.73 | 45.7% |
| 4 | Session affinity | 8.29 | 1,060 | 142 / 506 | 426 / 785 | 2.17 / 2.57 | 1.0% |
| 8 | Round robin | 10.20 | 1,305 | 525 / 694 | 826 / 1,062 | 2.52 / 3.49 | 42.1% |
| 8 | Session affinity | 14.62 | 1,871 | 210 / 322 | 529 / 665 | 2.50 / 3.22 | 96.0% |
| 16 | Round robin | 12.69 | 1,625 | 794 / 1,409 | 1,183 / 1,846 | 2.98 / 4.85 | 43.4% |
| 16 | Session affinity | 20.64 | 2,642 | 359 / 614 | 735 / 1,030 | 2.90 / 3.94 | 93.0% |

Session affinity improved request throughput at every tested concurrency. At
concurrency 16 it delivered 63% more requests and output tokens per second,
while reducing TTFT p50 by 55% and request-latency p50 by 38%. The largest
TTFT improvement appears in the median because four repeat questions can reuse
per-video locality after the first request. The reported prompt-cache counter
does not account for every form of multimodal reuse and should not be treated
as a direct measurement of video-cache hits.
