<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym Batch Gateway Impact

## Decision

The local CPU-only Mocker campaign supports the POC hypothesis. Under the
frozen 5/8/5-RPS online trace, stock Batch Gateway drain reduced SLO-good online
throughput by 61.95% and raised mean p99 TTFT from 212 ms to 7.17 s. Native
Planner control retained 99.00% of online-only goodput and held mean p99 TTFT to
291 ms while still completing every Batch request.

Relative to stock, Planner-native control increased online goodput by 160.31%
and reduced p99 TTFT by 95.92%. The tradeoff was explicit and bounded: mean
Batch completion rate fell 15.49%, from 3.077 to 2.600 RPS, and mean Batch
duration rose 18.33%, from 487.6 to 576.9 seconds.

The strict campaign compiler exited zero. All 15 runs are valid, all five
paired repetitions per arm are present, no data-quality issue was reported,
and no additional repetition is required by the preregistered stopping rule.

## Experiment

- Environment: local `minikube`, namespace `sachalm`, native `linux/arm64`.
- Serving target: one Dynamo frontend and one CPU Mocker worker using the same
  image reference,
  `dynamo-planner:planner-batch-impact-full-3cbab809d8-arm64-local-20260919`,
  and image ID,
  `sha256:4b3b12d536a501037ab2cf661154fd7ca21fee39b10a17465f87478efd840165`.
- Async target in every arm: `ghcr.io/llm-d/llm-d-async:v0.10.0`, resolved
  directly by Kubernetes to
  `docker-pullable://ghcr.io/llm-d/llm-d-async@sha256:07c0d8655626679d417c62e39568fe3d6b74c5b889ecf1ab7f31771fe21a8ed8`.
- Native Planner target: the same local Dynamo image reference and image ID as
  the serving frontend and worker.
- Mocker timing: fixed 75-ms prefill and 12-ms decode scheduler passes. These
  timings create deterministic contention and are not Qwen GPU measurements.
- Online trace: 1,080 requests in 60-second 5/8/5-RPS phases
  (300/480/300 requests), TTFT SLO 300 ms, ITL SLO 50 ms.
- Batch workload: 1,500 deterministic GSM8K requests, temperature 0, maximum
  128 output tokens.
- Arms: online-only, stock Async `prometheus-query` gate, and native Planner
  control through the Async `redis-leased-rate` gate.
- Design: five balanced repetitions per arm. Every attempted run is retained
  and chronological ordinal pairing is used.

The frozen trace SHA-256 is
`b8dcdd7c630831dbd37f4d49a4d7f347c256941626f347d737ee50162d9ad314`.
The GSM8K source SHA-256 is
`488c0fb6b64ce349e8d8da1ea001ccec4f8339194949aec7a03b6fcddaaeab8f`.

## Results

All values below are means over five valid runs.

| Arm | Goodput RPS | Goodput CV | SLO-good rate | Mean TTFT | p99 TTFT | Request error rate | Batch RPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Online-only | 5.979 | 0.004% | 100.00% | 112.9 ms | 211.6 ms | 0% | n/a |
| Stock | 2.275 | 2.612% | 38.15% | 2,508.4 ms | 7,165.3 ms | 0% | 3.077 |
| Planner-native | 5.920 | 0.496% | 99.02% | 118.8 ms | 291.5 ms | 0% | 2.600 |

Paired treatment effects use treatment minus reference.

| Comparison | Goodput effect | p99 TTFT effect | Interpretation |
| --- | ---: | ---: | --- |
| Stock vs online-only | -3.704 RPS (-61.95%) | +6,953.7 ms | Uncontrolled Batch drain destroys interactive SLO headroom. |
| Planner-native vs online-only | -0.060 RPS (-1.00%) | +79.8 ms | Planner preserves nearly all control goodput with a modest tail-latency cost. |
| Planner-native vs stock | +3.644 RPS (+160.31%) | -6,873.9 ms (-95.92%) | Planner recovers the online service while Batch remains durable. |

All 10 Batch jobs completed 1,500/1,500 requests with zero failures, for
15,000 validated Batch outputs. During the online interval, Planner-native
reduced Batch completion rate from 2.105 to 0.513 RPS (-75.62%). Once online
pressure ended, the configured five-RPS ceiling reopened and every job drained.

## Control Evidence

Every native run passed all 26 causal-control checks (130/130 total): stable
low/high/low frontend bands, sufficient Planner and Async samples, positive
backlog, unexpired Planner and Async leases, expected cap ranges, high-load cap
reduction, and final-low recovery.

Across the five native repetitions, median high-phase frontend load was
7.89-8.07 RPS and both Planner's desired cap and Async's applied cap were 0
RPS. Low-phase cap medians were in the configured 0.25-1.0-RPS band, and the
post-online decision stream returned to the five-RPS drain ceiling. This links
the observed performance change to the normal Planner tick, fenced Redis lease,
and Async gate rather than an experiment-driver mutation.

The compiler validated the runtime identity records at both interval
boundaries. It directly observed one Async digest throughout the campaign. The
frontend, worker, and Planner records retained a consistent image reference and
pod continuity within each run, but their resolved local runtime digest comes
from the explicit attestation described below. Under that attestation, the
frontend/worker serving signature is identical across all 15 runs, the Async
and combined control-plane signatures are identical within every arm, and the
Planner image signature is identical across all five native runs. The shared
serving signature is
`08164e8dc59e18411742aee8ce4155305eaf1c85ea77f15a2fce1293a100d3f3`;
the Async signature in every arm is
`4f6cd7b63ffef55654f7570f780630b512c4ad397173931e344c531ced9b25be`;
and the native Planner signature is
`939dfc3ffb9c3c62410ffa96eafa1ddf63582593f70d2f0a29dd08e8d81a5924`.

## Validity and Caveats

- Compiler status is `valid`: 15 inputs, 15 valid, 0 invalid, 0 data-quality
  issues, 5/5 runs per arm, and strict exit code 0.
- Planner-native p99 TTFT retained 10.39% CV and in-window Batch rate retained
  8.51% CV, both above the 5% trigger. Two near-zero paired effects (request
  throughput and p99 ITL versus online-only) also changed sign. Per the frozen
  rule, these findings expanded the campaign from three to five repetitions;
  the target is now met. The primary goodput and TTFT directions did not change.
- Pairing is chronological ordinal within each arm, with five repetitions and
  no confidence interval or significance test. The paired percentages above
  are campaign effect estimates, not population bounds.
- The campaign used a deterministic CPU Mocker. It demonstrates scheduling and
  contention control, not production GPU throughput, model quality, or a
  universal admission-rate setting. A GPU confirmatory run should calibrate
  work in tokens or service time rather than copy the 5.5-RPS Mocker envelope.
- Dynamo provenance records commit
  `3cbab809d8cea98fee0778e8621ea6ed075fd23a` with the campaign's known dirty
  implementation paths. The deployed image identity, deployment signature,
  workload signature, frozen inputs, per-run manifests, and compiler inputs are
  individually pinned and consistent across all reportable runs.
- The raw artifact sanitizer retained the local image reference but replaced
  its non-URL container runtime ID with `<redacted-url>`. Compilation therefore
  used the already-preserved `runtime_image_id` in the tracked campaign fixture
  as an explicit attestation for that exact image reference. The lossless output
  marks frontend, worker, and Planner identities with `image_id_attested: true`;
  Async remains `false` because its full pullable digest was observed directly.
  The compiler rejects missing, duplicate, conflicting, or unused/ref-mismatched
  attestations. Because the local digest is one external image-reference-to-ID
  assertion, these sanitized artifacts cannot independently exclude a mutable
  local tag resolving to a different digest in one run. Current harness captures
  preserve standard Docker, containerd, and CRI-O SHA-256 runtime IDs directly;
  the attestation path remains for these historical artifacts and nonstandard
  sanitized IDs.
- Two optional point-in-time Planner metrics snapshots in each native run could
  not connect to port 9085. Required continuous frontend/Async
  telemetry, Planner decision logs, Redis lease samples, artifact hashes, and
  every compiler control check remained complete and valid.
- Low-phase caps passed the preregistered median checks but had transient
  within-phase outliers. The evidence supports correct phase behavior, not a
  claim that admission control was perfectly smooth at every sample.

## Artifacts

- Lossless local summary:
  `results/compiled/20260919-planner-gym-batch-impact-r5-identity-attested-v5/summary.json`
  (SHA-256 `28957ee313b4c6dc4dbd2252186e0055b458835a8e8f6a357653db72763a24d3`).
- Human-readable local compiler output:
  `results/compiled/20260919-planner-gym-batch-impact-r5-identity-attested-v5/README.md`
  (SHA-256 `5eaa4fdcc2924cea8015e22a5c8a01121949c8858e9aed168b30dbf0d6ce6c91`).
- Tracked review fixture:
  [20260919-planner-gym-batch-impact.json](evidence/20260919-planner-gym-batch-impact.json).
- Raw run IDs and exact aggregate values are retained in the tracked fixture;
  full raw/compiled trees remain local and write-once by design.

The exact local recompilation, from the repository root, was:

```zsh
prior=examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/results/compiled/20260919-planner-gym-batch-impact-r5/summary.json
output=examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/results/compiled/20260919-planner-gym-batch-impact-r5-identity-attested-v5
args=()
while IFS= read -r run; do
  args+=(--run-directory "$run")
done < <(jq -r '.provenance.input_directories[]' "$prior")
python examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/workloads/compile_planner_gym_batch_impact.py \
  "${args[@]}" \
  --image-id-attestation 'dynamo-planner:planner-batch-impact-full-3cbab809d8-arm64-local-20260919=sha256:4b3b12d536a501037ab2cf661154fd7ca21fee39b10a17465f87478efd840165' \
  --output-directory "$output" \
  --minimum-repetitions 5 \
  --cv-threshold-percent 5 \
  --minimum-batch-overlap-fraction 0.95
```

The five unrelated `dyn` DGDs paused for isolation were restored to one replica
per component after compilation. All 15 generated deployments completed their
rollouts.
