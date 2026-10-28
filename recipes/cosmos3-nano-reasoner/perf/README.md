<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner Video Benchmark

This benchmark uses AIPerf 0.12.0 to measure streaming video inference against
the Cosmos3 Nano Reasoner Dynamo frontend. It runs 4,000 single-turn requests
at concurrency levels 1, 2, 4, 8, and 16: 20 questions for each of 200
30-second videos. Every request uses a fixed 128-token output length.

The manifest pins the private dataset
`jkiczka/cosmos3-benchmark-videos` to commit
`ce4b687c7736e29a13e00385cc8b6a6d28f355fd`. It downloads the complete dataset
snapshot to the shared PVC and selects the first 200 videos under `2fps-30s` in
sorted path order.

## Run the benchmark

The target deployment must be ready, `shared-model-cache` must be mounted in
the namespace, and `hf-token-secret` must contain a read-only `HF_TOKEN`.
Choose the profile matching the deployed eight-worker target.

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
  --timeout=14400s
```

On eight H200 workers, the full sweep took about 70 minutes for round robin and
about 50 minutes for session affinity, including dataset download. Concurrency
1 accounts for about half of that time, and each level prints its AIPerf
summary to the log when it finishes. The 14,400-second timeout leaves headroom
for slower clusters.

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
video has 20 self-contained questions sharing one session ID. This gives the
router repeated requests for the same video within every concurrency level,
rather than relying only on cache state carried over from an earlier level.

Both profiles use the same deterministic JSONL rows, one row per request. All
20 rows for a video share a session ID made of the dataset folder and the video
filename, for example:

```text
2fps-30s:<video-filename>
```

For `round-robin`, AIPerf does not send that ID as a routing header. For
`session-affinity`, it adds `--session-header X-Dynamo-Session-ID`, so every
replay of a given video carries the same header and remains pinned to one
worker while the 300-second affinity TTL is active. The two profiles otherwise
use the same eight-worker deployment shape and benchmark settings.

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

Deploy and benchmark one profile at a time so both runs use exactly eight H200
workers. Compare request throughput, output tokens/s, TTFT, request latency,
and ITL at each concurrency level.

## Measured results

The following single-run AIPerf measurements were collected on eight H200
workers with the manifests in this directory. Every row contains 4,000
successful requests with no errors. Output length was fixed at 128 tokens, so
request and output-token throughput scale together.

| Concurrency | Profile | Requests/s | Output tokens/s | TTFT p50 / p95 (ms) | Request latency p50 / p95 (ms) | ITL p50 / p95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Round robin | 1.95 | 250 | 61 / 523 | 378 / 812 | 2.21 / 2.65 |
| 1 | Session affinity | 2.74 | 351 | 57 / 85 | 339 / 632 | 2.20 / 2.62 |
| 2 | Round robin | 4.10 | 524 | 60 / 530 | 368 / 817 | 2.18 / 2.65 |
| 2 | Session affinity | 5.69 | 728 | 58 / 137 | 339 / 634 | 2.20 / 2.65 |
| 4 | Round robin | 7.98 | 1,021 | 61 / 541 | 372 / 833 | 2.19 / 2.66 |
| 4 | Session affinity | 11.37 | 1,456 | 58 / 68 | 340 / 595 | 2.20 / 2.66 |
| 8 | Round robin | 14.60 | 1,869 | 99 / 625 | 401 / 939 | 2.28 / 2.90 |
| 8 | Session affinity | 22.49 | 2,879 | 58 / 448 | 340 / 630 | 2.21 / 2.65 |
| 16 | Round robin | 21.43 | 2,743 | 333 / 803 | 686 / 1,233 | 2.61 / 4.16 |
| 16 | Session affinity | 39.18 | 5,016 | 62 / 135 | 395 / 503 | 2.58 / 3.33 |

Session affinity improved request throughput at every tested concurrency. At
concurrency 16 it delivered 83% more requests and output tokens per second,
while reducing TTFT p50 by 81% and request-latency p50 by 42%. Each worker
fetches and decodes a video the first time it sees it and reuses the processed
video for later questions. Round robin spreads a video's 20 questions across
up to eight workers, so the same video is processed on each of them; session
affinity processes it once. The gap therefore shows mainly in TTFT p95 at low
concurrency and in every metric once the workers are saturated.
