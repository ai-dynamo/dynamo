<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dedicated benchmark output storage

The shared VAST ancestor can return `EDQUOT` even when the campaign PVC itself has
ample nominal capacity. This overlay keeps the pinned `eval/` source tree unchanged,
retains VAST at `/artifacts` for harnesses and package caches, and mounts a dedicated
1 TiB Cinder PVC over `/artifacts/glm52-nscale` in both the runner and DinD containers.

The migration is read-only against VAST. It copies into a staging directory on Cinder,
hashes the source before and after the copy, verifies the destination identity, requires
the clean `vllm-serve-ab-r2` replay path to be absent, verifies the authoritative
Verified task-image manifest, atomically publishes the destination, and marks the Cinder
PV `Retain`. The original VAST tree remains the rollback source.

Before replacing the runner, export `/workspace/glm52-cache-prefill/verified`; it is an
`emptyDir`. After cutover, restore it and rerun the executable 500/500 completion gate:

```bash
snapshot=/Users/rmccormick/codex/glm52-campaign-artifacts/cache-prefill/verified-20260707T030800Z
benchmarks/glm52-nscale/artifact-storage/migrate.sh
benchmarks/glm52-nscale/eval/teardown-runner.sh
benchmarks/glm52-nscale/artifact-storage/deploy-runner.sh
benchmarks/glm52-nscale/artifact-storage/restore-prefill.sh "${snapshot}"
benchmarks/glm52-nscale/artifact-storage/assert-ready.sh
```

`artifact-storage/deploy-runner.sh` renders the top-level Kustomize overlay, binds the
pod annotation to that rendered manifest, resyncs the unchanged pinned evaluator source,
and reapplies the DinD registry mirror. The ordinary `eval/deploy-runner.sh` intentionally
does not apply this nested mount.

Rollback is explicit: first export any Cinder-only output, require an idle runner, delete
only the runner pod, then use the unchanged `eval/deploy-runner.sh`. The retained VAST
tree becomes visible again. Never copy Cinder back over VAST automatically.
