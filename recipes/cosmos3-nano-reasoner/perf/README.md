<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner Video Benchmark

This benchmark uses AIPerf 0.12.0 to measure streaming video inference against
the Cosmos3 Nano Reasoner Dynamo frontend. It runs 4,000 single-turn requests
at concurrency levels 1, 2, 4, 8, and 16: 20 questions for each of 200
30-second videos. Every request uses a fixed 128-token output length.

Before the sweep, the Job generates 200 synthetic videos with FFmpeg, so no
external dataset or credentials are needed. Each video is a 640x640, 2 FPS
H.264 clip of a Game of Life pattern seeded with the video index, capped at
1.2 Mbit/s so its size (about 3.3 MB) stays close to a real 30-second clip at
that resolution. The content is meaningless, which is fine here: outputs are
fixed at 128 tokens, so the benchmark measures serving throughput and latency,
not answer quality.

## Run the benchmark

The target deployment must be ready and `shared-model-cache` must be mounted in
the namespace. Choose the profile matching the deployed eight-worker target.

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
about 55 minutes for session affinity, including video generation. Concurrency
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
20 rows for a video share a session ID made of the video folder and the video
filename, for example:

```text
synthetic-2fps-30s:<video-filename>
```

For `round-robin`, AIPerf does not send that ID as a routing header. For
`session-affinity`, it adds `--session-header X-Dynamo-Session-ID`, so every
replay of a given video carries the same header and remains pinned to one
worker while the 300-second affinity TTL is active. The two profiles otherwise
use the same eight-worker deployment shape and benchmark settings.

## Run the smoke test

Use `smoke.yaml` for a fast end-to-end check before the full sweep. It
generates the first synthetic video, `synthetic-2fps-30s/vid-0000.mp4`, exposes
it to the model through a temporary in-cluster Service, and validates a
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
| 1 | Round robin | 1.91 | 244 | 61 / 519 | 394 / 831 | 2.32 / 2.79 |
| 1 | Session affinity | 2.60 | 333 | 58 / 87 | 355 / 677 | 2.32 / 2.79 |
| 2 | Round robin | 3.90 | 499 | 61 / 518 | 388 / 828 | 2.31 / 2.77 |
| 2 | Session affinity | 5.46 | 699 | 58 / 450 | 354 / 672 | 2.31 / 2.77 |
| 4 | Round robin | 7.71 | 987 | 61 / 535 | 386 / 851 | 2.31 / 2.78 |
| 4 | Session affinity | 10.86 | 1,390 | 57 / 150 | 355 / 682 | 2.32 / 2.78 |
| 8 | Round robin | 14.38 | 1,841 | 90 / 601 | 407 / 941 | 2.39 / 3.04 |
| 8 | Session affinity | 21.44 | 2,745 | 58 / 468 | 354 / 697 | 2.31 / 2.77 |
| 16 | Round robin | 21.39 | 2,738 | 303 / 760 | 674 / 1,215 | 2.76 / 4.46 |
| 16 | Session affinity | 37.68 | 4,822 | 62 / 134 | 409 / 537 | 2.69 / 3.55 |

Session affinity improved request throughput at every tested concurrency. At
concurrency 16 it delivered 76% more requests and output tokens per second,
while reducing TTFT p50 by 80% and request-latency p50 by 39%. Each worker
fetches and decodes a video the first time it sees it and reuses the processed
video for later questions. Round robin spreads a video's 20 questions across
up to eight workers, so the same video is processed on each of them; session
affinity processes it once. The gap therefore shows mainly in throughput and
tail latency at low concurrency, and in every metric once the workers are
saturated.
