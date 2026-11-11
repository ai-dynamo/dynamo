<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Super Reasoner Video Benchmark

This benchmark uses AIPerf 0.12.0 to measure streaming video inference against
the Cosmos3 Super Reasoner Dynamo frontend. It runs 4,000 single-turn requests
at concurrency levels 1, 2, 4, 8, and 16: 20 questions for each of 200
30-second videos. Every request uses a fixed 128-token output length and puts
the video before the question, so all questions about one video share a prompt
prefix.

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

From the repository root, with `CONTEXT` and `NAMESPACE` set to the cluster
and namespace of the deployment:

```bash
export PROFILE=round-robin # or: session-affinity
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f "recipes/cosmos3-super-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job cosmos3-super-reasoner-perf \
  --ignore-not-found
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/cosmos3-super-reasoner/perf/perf.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs -f job/cosmos3-super-reasoner-perf
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait \
  --for=condition=Complete job/cosmos3-super-reasoner-perf \
  --timeout=14400s
```

On eight H200 workers, the full sweep took about 2 to 2.5 hours per profile,
including package installation and video generation. Concurrency 1 accounts
for about half of that time, and each level prints its AIPerf
summary to the log when it finishes. The 14,400-second timeout leaves headroom
for slower clusters.

The log ends with the artifact directory. It has one subdirectory per
concurrency level, including AIPerf JSON, CSV, per-request, and server-metric
exports:

```text
/shared-model-cache/perf/cosmos3-super-reasoner/<profile>/<epoch>_<job-name>/
```

To copy a completed run locally while the deployment is running:

```bash
RUN_ROOT="<artifact-directory-from-the-job-log>"
export DGD="cosmos3-super-reasoner-agg-h200-${PROFILE}"
WORKER="$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get pods \
  -l "nvidia.com/dynamo-graph-deployment-name=${DGD}" \
  -o name | grep worker | head -n 1)"
kubectl --context "$CONTEXT" -n "$NAMESPACE" cp \
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

AIPerf groups rows with the same session ID into one conversation and gives
that conversation its own generated identifier. For `round-robin`, no routing
header is sent. For `session-affinity`, `--session-header X-Dynamo-Session-ID`
sends the conversation identifier in that header, so all questions about a
video carry the same value and stay on one worker while the 300-second
affinity TTL is active. The two profiles otherwise
use the same eight-worker deployment shape and benchmark settings.

AIPerf's chat endpoint places a message's text before its media. The Job
installs a small AIPerf plugin that registers a `chat_video_first` endpoint
type, which moves each message's `video_url` parts ahead of its text, and runs
the sweep with `--endpoint-type chat_video_first`. With the video first, the
questions about one video share their leading prompt tokens, so a worker that
has already served the video can reuse that prefix from its KV cache instead
of prefilling it again.

## Run the smoke test

Use `smoke.yaml` for a fast end-to-end check before the full sweep. It
generates the first synthetic video, `synthetic-2fps-30s/vid-0000.mp4`, exposes
it to the model through a temporary in-cluster Service, and validates a
non-empty chat completion:

```bash
export PROFILE=round-robin # or: session-affinity
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f "recipes/cosmos3-super-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job cosmos3-super-reasoner-smoke \
  --ignore-not-found
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply -f recipes/cosmos3-super-reasoner/perf/smoke.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs -f job/cosmos3-super-reasoner-smoke
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

| Concurrency | Profile | Requests/s | Output tokens/s | TTFT p50 / p90 (ms) | Request latency p50 / p90 (ms) | ITL p50 / p90 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Round robin | 0.94 | 121 | 45 / 631 | 991 / 1,523 | 6.61 / 7.63 |
| 1 | Session affinity | 1.16 | 149 | 44 / 44 | 891 / 1,037 | 6.58 / 7.63 |
| 2 | Round robin | 1.90 | 243 | 45 / 637 | 980 / 1,529 | 6.62 / 7.67 |
| 2 | Session affinity | 2.36 | 302 | 44 / 44 | 888 / 1,039 | 6.56 / 7.61 |
| 4 | Round robin | 3.83 | 490 | 45 / 641 | 976 / 1,532 | 6.61 / 7.63 |
| 4 | Session affinity | 4.68 | 599 | 44 / 44 | 892 / 1,038 | 6.60 / 7.62 |
| 8 | Round robin | 7.38 | 944 | 48 / 716 | 989 / 1,600 | 6.70 / 7.84 |
| 8 | Session affinity | 9.35 | 1,196 | 44 / 44 | 889 / 1,035 | 6.56 / 7.64 |
| 16 | Round robin | 12.65 | 1,619 | 124 / 757 | 1,140 / 1,798 | 7.36 / 9.42 |
| 16 | Session affinity | 17.78 | 2,276 | 44 / 47 | 919 / 1,066 | 6.88 / 7.98 |

Session affinity improved request throughput at every tested concurrency, by
23% at concurrency 1 and 41% at concurrency 16. At concurrency 16 it also
reduced TTFT p50 by 65% and request-latency p50 by 19%. Each worker fetches
and decodes a video the first time it sees it, then reuses the processed video
and its cached prompt prefix for later questions. Round robin spreads a
video's 20 questions across up to eight workers, so the same video is
processed and prefilled on each of them; session affinity does it once. With
session affinity, TTFT p90 stays at the cached-prefix level of about 45 ms,
while round robin pays a full prefill of about 0.6 to 0.8 seconds on a
noticeable share of requests.

From concurrency 8 to 16, session-affinity throughput scaled almost linearly
(1.9x) with TTFT p50 flat at 44 ms. Round-robin throughput grew only 1.7x and
its TTFT p50 rose from 48 ms to 124 ms, so round robin was starting to queue.
