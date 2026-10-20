<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kubernetes rolling upgrades

`rolling-upgrade-compatibility.yml` adds a separate, dependent suite using the
candidate operator throughout. It upgrades the actual frontend/worker images
from each manifest release to the candidate, in both frontend-first and
worker-first order, for embedding and aggregated chat: eight sequential paths.
It reuses `ManagedDeployment`; the client wire checks live in `rolling_http.py`. PR execution honors
the existing `run_deploy_tests` gate and participates in `dynamo-status-check`;
nightly and manual runs call the same workflow. A skipped deploy lane is not
upgrade validation. Manual runs accept candidate image tags in the configured
ACR repository; the checkout and operator must correspond to those images.

Each path uses two one-GPU worker replicas, two CPU frontend replicas, Deployment
mode, `maxSurge=0`, `maxUnavailable=1`, and a namespace quota of two GPUs. The
suite owns a dedicated vCluster; do not run it in a namespace shared with other
GPU workloads. Kubernetes admission limits reserved GPUs, and Pod observations
also count terminating Pods. This is a resource-allocation bound, not physical
GPU telemetry. Worker init containers download the same pinned model revision into an
emptyDir; frontend Pods do not download or mount model snapshots; no checkout code is mounted into inference containers. Candidate
runtime version overrides are patched together with image references.

An independent client Pod sends requests directly to the stable frontend
Service, with fresh connections and no retry. It continues through both image
patches and post-upgrade observation periods. All three embedding cases and
chat unary/stream/max-token cases must run at least twice in each phase. The
same-worker derived-stop case remains in the static suite: output prefixes can
differ across engine versions. This is low-rate availability/contract coverage,
not a throughput, exact latency, or stream-migration benchmark.

The test requires new Pod UIDs, target image references and nonempty image IDs,
Ready old/new overlap retained from a Pod watch started at the pre-patch list
resourceVersion, zero container restarts, and generate-request counter
increases on both worker revisions during worker replacement. Active inference
canaries are disabled so they cannot supply those counter increases; Kubernetes
readiness and liveness probes remain. Counter samples establish revision-level
activity, not exact request-to-worker attribution or all four pairings within
one instant. If overlap/counters cannot be observed, the test fails rather than
claiming coverage. The static pair matrix supplies deterministic pairing checks.

Artifacts include raw HTTP/SSE records, final client counters, timestamped Pod
states and image IDs, raw Pod watch events, worker metrics, sampled container logs,
and stage reports.
Old Pod logs are sampled before they disappear. Every completed request,
including the last in-flight request when stopping the client, is included in
validation. The client stops before DGD teardown. A separate always-run job
removes the vCluster even when the test job times out. Cleanup preserves the
original failure, attempts every delete, and stops subsequent cases if resource
isolation cannot be re-established. The workflow requires all eight cases to
execute and pass; skipped or incomplete JUnit coverage fails the job.

Local invocation (a dedicated namespace with the current operator, etcd/NATS,
registry access, `hf-token-secret`, and capacity for two GPUs is required):

```bash
python3 -m pytest tests/deploy/test_n2_rolling_upgrade.py -n 0 -v \
  --namespace=YOUR_ISOLATED_NAMESPACE \
  --frontend-image=YOUR_CANDIDATE_FRONTEND --image=YOUR_CANDIDATE_SGLANG
```

This suite does not upgrade the operator binary, configure ingress weighting,
exercise Grove/OnDelete, or claim PD/KV-routing compatibility. GPU Kubernetes
execution is required before promoting the new suite from draft validation.
