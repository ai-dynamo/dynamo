<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Component compatibility tests

This suite applies the shared deployment API checks to four SGLang component
pairs: candidate frontend with N-1/N-2 workers, and N-1/N-2 frontends with a
candidate worker. Each pair runs chat and embedding, for eight deployments.
`releases.json` names the latest selected patch release in each historical minor
line. Update it when the candidate advances to a new minor; an absent release
fails setup instead of silently shrinking coverage.

## Shared coverage

`test_dgd.py` owns current/current coverage. PR and nightly workflows require the
ordinary SGLang deployment job to succeed before running this suite; skipped
controls do not satisfy the dependency. Manual dispatch invokes the same ordinary
chat and embedding tests first. Both suites call `check_deployment_api` for unary,
streaming, token limits, stop behavior and embedding default/float/batch responses.

Deployments load the chat or embedding YAML from
`examples/backends/sglang/deploy/`, then patch component images, runtime versions
and cache paths with `DeploymentSpec`. `ManagedDeployment` owns readiness,
port forwarding, Pod manifests (including image IDs), logs and cleanup. There is
no image digest resolver, Pod exec, injected inference code, private Docker
runner or per-test ResourceQuota.

## Run

Use Linux amd64 candidate image tags beginning with their runtime version, such
as `1.5.0-ci-<sha>-...`. The test requires a shared PVC accessible to both
frontend and worker Pods. Prepare the pinned snapshots in `model-download.yaml`
using a candidate SGLang image, then run:

```bash
python -m pytest tests/deploy/test_n2_compatibility.py \
  --namespace default --image "$WORKER_IMAGE" --frontend-image "$FRONTEND_IMAGE" \
  --model-cache-pvc model-cache --model-cache-mount /models -m k8s -n 0
```

The CI workflow prepares the cache with a GPU-free Job. Local invocations must
prepare it first. The Job writes both pinned snapshots; inference mounts the same
cache and uses offline snapshot paths. The ordinary deploy controls use the same
model IDs and request checks, but do not enforce these snapshot revisions, so
they are functional prerequisites rather than a strict experimental control.

`N2_RELEASE_LINE` overrides the candidate minor line inferred from `Cargo.toml`.
Image tags supply `runtimeVersionOverride` separately for each component, keeping
operator health defaults appropriate for historical runtimes. Kubernetes discovery
and TCP requests are used across the supported window.

CI schedules cases serially to bound GPU use. Each case has its own deployment
name and fixture; there is no cross-test baseline state. Compatibility failures
allow later cases to run. A cleanup failure stops the session after reporting the
current item because resource isolation is no longer established. The outer
workflow tears down its dedicated vCluster after failure or cancellation.

Raw API responses and a version-pair record accompany the normal deploy logs and
JUnit report. A skipped workflow or interrupted matrix provides no evidence for
unexecuted pairs. A successful CPU test run is not GPU compatibility evidence.

This is a static component matrix, not a rolling-upgrade, performance or release
promotion test. Coverage is currently SGLang aggregated chat and embedding.
