<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner-controlled live run

## Outcome

Run `20260828T183835Z-planner-controlled-1451d7` completed all 100
deterministic GSM8K requests through the Planner-controlled llm-d Async stack
in a dedicated test namespace. There were no failed requests and the compiler
reported no data-quality issues. Its machine-readable metadata labels it
`planner-controlled` and pairs it with controller run
`20260828T183813Z-planner-loop-15424f`.

| Measure | Stock constant gate | Planner-controlled gate |
| --- | ---: | ---: |
| Requests completed | 100/100 | 100/100 |
| Failed requests | 0 | 0 |
| Terminal duration | 6.180863 s | 28.485065 s |
| Average completion rate | 16.178970 RPS | 3.510612 RPS |
| Peak observed completion interval | 25.245067 RPS | 5.905139 RPS |
| Configured admission ceiling | none beyond stock gate | 5 RPS |

The controlled average was 21.70% of the stock average, a 78.30% reduction,
and terminal duration was 4.61 times the stock run. The short
completion-interval peak is not an admission measurement: responses can
complete in bursts after earlier admission and the progress poll interval is
coarse.

All 100 response records had HTTP status 200 and `finish_reason="length"` at
the configured 128-token limit. Stock and controlled runs used the same
normalized input checksum and each recorded 9,923 prompt, 12,800 completion,
and 22,723 total tokens. This validates workload equivalence, transport,
cardinality, and pacing; it does not establish GSM8K answer accuracy.

## Control-loop evidence

Controller run `20260828T183813Z-planner-loop-15424f` completed 40/40 apply
iterations without an observation, policy, or actuation error.

- Iterations 1-19 published a leased 0-RPS pause while no active job existed.
- Iterations 20-30 observed one active 100-request job and published a 5-RPS
  ceiling plus an advisory replica floor of one.
- Iterations 31-40 observed the job terminal and returned the ceiling to 0 RPS.
- The maximum actual Prometheus scrape age was 5.027878 seconds against a
  15-second bound; no feedback observation was marked stale.
- The first Gateway validation sample reported its API total as zero, but the
  harness correctly exposed `100` total and `100` remaining from trusted
  `planner_request_count` metadata until the Gateway count became authoritative.
- After the final 10-second lease, Redis reported the control key absent with
  `PTTL=-2` at Redis server time `1787942445.468513`. A missing key is the
  gate's tested fail-closed condition, but no live request was submitted after
  expiry in this treatment.

The Batch API progress stream reached 100/100 in 28.485065 seconds with
monotonic progress. Async's dispatch-attempt counter increased from 100 to 200,
and the frontend recorded exactly 100 admissions over 22.053508 seconds
(4.489082 RPS across inter-arrival intervals), consistent with the 5-RPS
token-bucket ceiling.

## Deployment provenance

- Helm release: `async-dispatch`, revision 2, status `deployed`.
- Controlled image and live container `imageID`: matched; private registry
  coordinates are intentionally omitted.
- Kubernetes context: omitted from checked-in POC artifacts.
- Async, frontend, and worker pods were Ready with zero restarts after the
  canonical run.
- The 5-second `PodMonitor` target was healthy under Prometheus job
  `<poc-namespace>/async-dispatch-llm-d-async`.
- A direct frontend-to-worker chat completion returned exactly `READY`.

The GPU Reaper had scaled the Dynamo graph to zero before the run. One first
replacement worker encountered AWS CNI IP exhaustion and was safely
rescheduled by deleting only that controller-managed pod. The recreated
frontend also exposed a deployment gap: without the shared Hugging Face cache,
its tokenizer download failed and the model returned 404. The graph manifest
now mounts the same persistent model cache and token secret in the frontend;
model discovery and the direct inference smoke then succeeded.

## Scope and caveats

This proves the standalone single-pool POC path: Gateway job state and Async
feedback enter Planner, Planner emits a renewable leased drain decision, and
Async enforces it at worker-pool admission. It does not yet prove concurrent
online-load protection, meaningful deadline/headroom tradeoffs, replica
actuation, fault recovery, live post-expiry request behavior, or native Planner
plugin lifecycle integration.
Replica floor output remains advisory in this runner, arbitrary user-supplied
due dates remain unsupported by Gateway v0.3, and the pool must remain
dedicated to the one Planner tenant used for the observation.

## Evidence

- [Tracked canonical pair manifest](evidence/20260828-canonical-controlled-pair.json)
- [Tracked image provenance](../research/20260828-async-image-provenance.md)
- Local-only raw workload run (ignored by Git):
  `results/raw/20260828T183835Z-planner-controlled-1451d7/`
- Local-only compiled workload result (ignored by Git):
  `results/compiled/20260828T183835Z-planner-controlled-1451d7-summary/`
- Local-only raw controller decisions (ignored by Git):
  `results/raw/20260828T183813Z-planner-loop-15424f/control-loop-decisions.jsonl`
- Stock comparison: `reports/20260828-stock-live-baseline.md`
