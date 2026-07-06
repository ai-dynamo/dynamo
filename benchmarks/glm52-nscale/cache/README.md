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
Its filesystem store lives under `/artifacts/cache/dockerhub-registry` inside the
dedicated, expandable 500 GiB `dockerhub-pull-cache-data` Cinder PVC. This keeps
cold-fill writes outside the shared VAST artifact quota, survives registry-pod
replacement, and deduplicates blobs across phases and serving variants. The PVC
is declared separately in `data-pvc.yaml` and is retained by teardown. Cached
entries use an explicit one-year TTL so they outlive the campaign. The service is deliberately not named with the `glm52-` prefix, so
serving teardown does not remove or wait on it.

`migrate.sh` patches the bound Cinder PV reclaim policy to `Retain`. The PVC
annotation is descriptive; the PV policy is the deletion safeguard. Deployment
and readiness checks accept later expansions above 500 GiB and never reapply a
smaller request.

Deploy the cache, then online-reload the existing DinD daemon:

```bash
benchmarks/glm52-nscale/cache/deploy.sh
benchmarks/glm52-nscale/cache/configure-runner.sh
benchmarks/glm52-nscale/cache/assert-ready.sh
benchmarks/glm52-nscale/cache/verify.sh
```

To move an existing artifact-PVC-backed cache onto the dedicated volume, run
`benchmarks/glm52-nscale/cache/migrate.sh` once while the evaluator is idle. On
a fresh namespace with no cache Deployment, run
`benchmarks/glm52-nscale/cache/migrate.sh --initialize-empty` instead; this is
the only path allowed to bind an intentionally empty cache.
The script stops the registry, performs and hashes an offline copy, switches the
Deployment without changing the Service endpoint, verifies cache hits, and rolls
back to the former store on failure. `deploy.sh` refuses the legacy backing so the
cache cannot be silently replaced with an empty volume. `deploy.sh` requires
the migration marker hash to be bound in both a ConfigMap and the PVC annotation;
if only the ConfigMap is lost, it is reconstructed from the retained PVC binding
and the registry init container proves that hash against the on-volume marker.
This prevents a missing or replaced cache from silently turning every SWE task
into another Docker Hub pull. Set
`CACHE_MIGRATION_TIMEOUT_SECONDS` when migrating a cache that needs more than the
default 30 minutes.

Docker Engine supports live reload of `registry-mirrors` and
`insecure-registries`; `configure-runner.sh` verifies that the runner pod UID and
DinD restart count remain unchanged. By default it requires an idle runner. An
intentional mid-cell change requires `--allow-active` and remains protected by the
immutable task-image guards. The mirror changes image transport only.
Generation still records each immutable image ID and RepoDigest, and evaluation's
existing task-image guard rejects any content mismatch.

The July 6 migration retained the former VAST-backed store at
`/artifacts/cache/dockerhub-registry` as a rollback copy. The active
registry requires `/artifacts/cache/.glm52-migration-v1.json` on its dedicated
PVC before it starts, and its init container checks that file against the
cluster-bound SHA-256 marker.
`registry-mirror-vast-rollback.yaml` is the committed, deterministic emergency
path back to the retained VAST copy; rollback does not depend on ReplicaSet
revision history.

The first request for an uncached tag still reaches Docker Hub. Later requests are
served from the persistent cache. Optional Docker Hub credentials can be added to
the registry proxy configuration if the first-fill rate also exceeds anonymous
capacity; never commit those credentials.

For the current SWE-bench Verified recovery, prefill the authoritative task-image
population from the completed immutable `task-images.json` manifest:

```bash
benchmarks/glm52-nscale/cache/prefill.sh start verified \
  /artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified/task-images.json
benchmarks/glm52-nscale/cache/prefill.sh status verified
benchmarks/glm52-nscale/cache/require-complete.sh verified \
  /artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified/task-images.json
```

The prefill atomically acquires the evaluator's global campaign lock before it
starts one named tmux session on the runner, so neither an official workload nor
another suite prefill can win the same start race. The remote wrapper releases
the lock only after the prefill process exits; an abrupt runner loss leaves the
lock in place for fail-closed manual recovery. It is
idempotent and resumable, retains its source manifest and per-instance evidence
under `/artifacts/glm52-nscale/cache/prefill/<suite>`, polls the anonymous Docker
Hub allowance before every pull, and leaves a five-pull reserve. A registry-side
rate-limit response forces a full reported-window cooldown. Every pulled image
must exactly match the source manifest's image ID, RepoDigests, and canonical
content-identity hash before progress is committed. State is bound to the active
PVC UID plus migration marker; every recorded entry is pulled and identity-checked
again after a resume, and an explicit second `start` revalidates a completed set.
Do not launch the next
official cell until `status.json` reports `state=complete`, `completed=total`, and
the tmux session has exited, and `prefill.sh status` reports
`cache_binding_matches=true`. If prefill exits early, its status remains failed or
incomplete even though the lock and tmux session are gone; the runbook completion
gate remains authoritative.
`require-complete.sh` is the executable preflight used by the Verified runbook;
it rejects incomplete state, an active prefill, a changed manifest/cache binding,
or anything other than 500 validated catalog entries.

This implementation is intentionally Verified-only. It relies on Verified's
one-repository-per-instance layout. SWE-bench Pro uses many tags in a shared
`jefzda/sweap-images` repository and therefore requires tag-level catalog/state
handling before Pro prefill; do not reuse this command for Pro or Multilingual.

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
