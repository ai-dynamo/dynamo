<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docker Hub pull-through cache

SWE-bench uses one remote `sweb.eval` image per instance. The evaluator runner's
Docker data is intentionally pod-local and the official harness removes instance
images after use, so repeated cells would otherwise pull the same tags from Docker
Hub and can exhaust the anonymous allowance.

This directory deploys a single CNCF Distribution registry in pull-through mode.
Its filesystem store lives under `/artifacts/cache/dockerhub-registry` on the
existing 1 TiB artifact PVC, so blobs survive evaluator-pod replacement and are
deduplicated across phases and serving variants. Cached entries use an explicit
one-year TTL so they outlive the campaign. The service is deliberately not named with the `glm52-` prefix, so
serving teardown does not remove or wait on it.

Deploy the cache, then online-reload the existing DinD daemon:

```bash
benchmarks/glm52-nscale/cache/deploy.sh
benchmarks/glm52-nscale/cache/configure-runner.sh
benchmarks/glm52-nscale/cache/assert-ready.sh
benchmarks/glm52-nscale/cache/verify.sh
```

Docker Engine supports live reload of `registry-mirrors` and
`insecure-registries`; `configure-runner.sh` verifies that the runner pod UID and
DinD restart count remain unchanged. By default it requires an idle runner. An
intentional mid-cell change requires `--allow-active` and remains protected by the
immutable task-image guards. The mirror changes image transport only.
Generation still records each immutable image ID and RepoDigest, and evaluation's
existing task-image guard rejects any content mismatch.

The first request for an uncached tag still reaches Docker Hub. Later requests are
served from the persistent cache. Optional Docker Hub credentials can be added to
the registry proxy configuration if the first-fill rate also exceeds anonymous
capacity; never commit those credentials.

The service is cluster-internal, stores only public Docker Hub content, and has no
upstream credentials. Transport from DinD to the in-cluster mirror is HTTP; the
existing generation/evaluation guard validates immutable image content before a
result can be imported. Run `assert-ready.sh` before every SWE-bench cell so a
missing mirror fails preflight instead of silently relying on direct pulls.
`verify.sh` additionally performs two identical cache-hit pulls, proves the Docker
Hub allowance did not change, and stores a sanitized JSON attestation on the
persistent artifact PVC. It requires an idle evaluator runner so concurrent cold
fills cannot contaminate the quota comparison. It warms its small public test image
before measuring; that initial cold fill can consume one anonymous pull.
