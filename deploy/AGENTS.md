<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes deployment — Agent Guide

`deploy/` is how Dynamo runs on Kubernetes: the operator that owns the custom
resources, the Helm charts that install it, the gateway integration, and the
node-level agents. It is mostly Go and YAML rather than Rust or Python, and it
has its own build and regeneration steps.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the environment setup and the
directory tour, and the root [`AGENTS.md`](../AGENTS.md) for repository-wide PR
conventions.

## Directory map

| Path | Contents |
|------|----------|
| `operator/` | The Dynamo operator: a Kubebuilder project owning the CRDs, controllers, and admission webhooks. Has its own agent files — see below. |
| `helm/charts/platform/` | The platform chart that installs the operator and its dependencies. |
| `helm/charts/power-agent/`, `helm/charts/snapshot/` | Charts for the two DaemonSets. |
| `inference-gateway/` | Gateway API Inference Extension integration: `epp/` (Go endpoint picker) and `ext-proc/` (Rust external processor). |
| `observability/` | Grafana dashboards and setup for Kubernetes deployments. |
| `power-agent/` | Privileged DaemonSet for GPU power-cap enforcement. |
| `snapshot/` | CRIU-based checkpoint/restore DaemonSet. |
| `pre-deployment/` | Scripts that check a cluster meets Dynamo's requirements before you deploy. |
| `utils/` | Utilities and manifests for benchmarking and profiling workflows. |

## The custom resources

The CRD sources are the Go types under
[`operator/api/`](operator/api); the generated schemas live in
`operator/config/crd/bases/`, one file per kind:

`DynamoGraphDeployment` and `DynamoGraphDeploymentRequest`,
`DynamoComponentDeployment`, `DynamoGraphDeploymentScalingAdapter`,
`DynamoModel`, `DynamoWorkerMetadata`, `DynamoCheckpoint`, and
`PodSnapshot`/`PodSnapshotContent`.

> [!NOTE]
> The generated artifacts are not editable by hand. Change the Go types or the
> `+kubebuilder:` markers and run `make manifests` in `operator/`, which
> regenerates both the CRD bases and the RBAC role vendored into the platform
> chart. `operator/AGENTS.md` states this rule and the reconciliation semantics
> that go with it.

## Operator work

The operator carries a nested set of agent files, and they are the local
authority for anything under `operator/`:

- [`operator/AGENTS.md`](operator/AGENTS.md) — RBAC regeneration, reconciliation
  and admission semantics, Go code style
- [`operator/api/AGENTS.md`](operator/api/AGENTS.md)
- [`operator/internal/AGENTS.md`](operator/internal/AGENTS.md),
  [`internal/controller/`](operator/internal/controller/AGENTS.md),
  [`internal/crdmigrator/`](operator/internal/crdmigrator/AGENTS.md),
  [`internal/webhook/validation/`](operator/internal/webhook/validation/AGENTS.md)
- [`operator/test/e2e/AGENTS.md`](operator/test/e2e/AGENTS.md)

Read the one closest to the file you are changing; when it and this file both
apply, the nested one wins.

## Deployment guides

The user-facing deployment documentation lives under
[`docs/fern/pages/kubernetes/`](../docs/fern/pages/kubernetes/) and the
ready-made manifests under [`recipes/`](../recipes/), which has its own
[`AGENTS.md`](../recipes/AGENTS.md) describing the Kustomize authoring model.
Prefer extending a recipe over writing a new manifest by hand.
