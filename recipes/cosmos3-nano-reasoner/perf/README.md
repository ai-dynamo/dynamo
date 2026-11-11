<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner Video Benchmark

This benchmark uses AIPerf 0.12.0 to measure streaming video inference against
the Cosmos3 Nano Reasoner Dynamo frontend. It runs 4,000 single-turn requests
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
  -f "recipes/cosmos3-nano-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job cosmos3-nano-reasoner-perf \
  --ignore-not-found
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/cosmos3-nano-reasoner/perf/perf.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-perf
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait \
  --for=condition=Complete job/cosmos3-nano-reasoner-perf \
  --timeout=14400s
```

On eight H200 workers, the full sweep took about 70 minutes for round robin and
about 55 minutes for session affinity, including package installation and video
generation. Concurrency 1 accounts for about half of that time, and each level
prints its AIPerf summary to the log when it finishes. The 14,400-second timeout leaves headroom
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
  -f "recipes/cosmos3-nano-reasoner/perf/profiles/${PROFILE}.yaml"
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job cosmos3-nano-reasoner-smoke \
  --ignore-not-found
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply -f recipes/cosmos3-nano-reasoner/perf/smoke.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-smoke
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
| 1 | Round robin | 1.94 | 249 | 45 / 513 | 371 / 822 | 2.33 / 2.75 |
| 1 | Session affinity | 2.66 | 340 | 44 / 71 | 341 / 465 | 2.32 / 2.77 |
| 2 | Round robin | 4.17 | 533 | 45 / 480 | 363 / 799 | 2.37 / 2.77 |
| 2 | Session affinity | 5.55 | 710 | 43 / 152 | 342 / 650 | 2.32 / 2.77 |
| 4 | Round robin | 8.15 | 1,044 | 45 / 483 | 366 / 805 | 2.38 / 2.79 |
| 4 | Session affinity | 11.11 | 1,422 | 43 / 55 | 340 / 458 | 2.32 / 2.77 |
| 8 | Round robin | 15.19 | 1,944 | 46 / 541 | 389 / 884 | 2.44 / 2.96 |
| 8 | Session affinity | 22.06 | 2,824 | 43 / 419 | 338 / 618 | 2.32 / 2.77 |
| 16 | Round robin | 24.15 | 3,091 | 215 / 657 | 583 / 1,065 | 2.73 / 3.75 |
| 16 | Session affinity | 43.93 | 5,622 | 43 / 97 | 349 / 445 | 2.40 / 2.98 |

Session affinity improved request throughput at every tested concurrency. At
concurrency 16 it delivered 82% more requests and output tokens per second,
while reducing TTFT p50 by 80% and request-latency p50 by 40%. Each worker
fetches and decodes a video the first time it sees it, then reuses the
processed video and its cached prompt prefix for later questions. Round robin
spreads a video's 20 questions across up to eight workers, so the same video is
processed and prefilled on each of them; session affinity does it once. The
gap therefore shows mainly in throughput and tail latency at low concurrency,
and in every metric once the workers are saturated.

Putting the video before the question raised throughput for both profiles.
With the question first, the same sweep reached 37.68 requests/s with session
affinity and 21.39 requests/s with round robin at concurrency 16; with the
video first it reached 43.93 and 24.15, gains of 17% and 13%. Round robin also
benefits because each worker still sees most videos more than once.
