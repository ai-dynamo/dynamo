<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Run SWE-bench Verified with Harbor and Pi in Agent Substrate sandboxes.
Use a model served by NVIDIA Dynamo or a compatible external API.

Start with [Run SWE-bench Verified with Harbor, Pi, and Substrate](../../../docs/fern/pages/use-cases/agents/harbor-substrate.mdx).
That page owns the walkthrough, including prerequisites, prepared task images,
golden snapshots, one trial, a concurrent batch, result inspection, and cleanup.
It also outlines a planned full-dataset path using published task images and
runtime installation. Batch generation and verifier bootstrap for that path
are not yet included or validated.

This directory contains the runnable files:

- `substrate_environment.py`: Harbor's external environment provider. It talks
  to Substrate directly: `kubectl-ate` for actor lifecycle, and the Substrate
  router for commands and files inside the actor.
- `Dockerfile.harbor`: runner image with pinned Harbor, Substrate CLI, env client,
  and patched adapters under `/opt/harbor`.
- `harbor-runner.yaml`: Kubernetes Job with a pinned dataset init container,
  mounted configuration, projected credentials, and persistent results.
- `harbor-runner-rbac.yaml`: runner ServiceAccount and trust-bundle permissions.
- `harbor-results.yaml`: results PVC, retained when runner Jobs are deleted.
- `results-reader.yaml`: read-only Pod for inspecting and exporting saved results.
- `pi_trajectory.py`: local converter from copied Pi logs to Harbor ATIF trajectories;
  see the tutorial's result inspection instructions.
- `harbor-job.yaml`: editable Harbor configuration with task-to-template mappings.
- `Dockerfile.pi`: reusable preparation recipe for one task instance image,
  adding pinned Pi, provider, and offline-verifier dependencies.
- `substrate-install.yaml`: generic platform configuration for `ate-setup`, with
  the pinned release, bundled PostgreSQL, and RustFS storage defaults. Follow
  the installation guide to review its defaults and configure the storage plugins.
- Other YAML manifests: worker pool, task template, and the atespace access policy.
- `runner-kubeconfig.yaml`: in-cluster CLI identity using projected token files.
- `*.patch`: compatibility changes applied to the pinned upstream sources.
- [Install Substrate](../../../docs/fern/pages/kubernetes/agent-sandboxes/install-substrate.mdx):
  platform setup in the Kubernetes Guide.
- [Trace Sandboxed Agents](../../../docs/fern/pages/use-cases/agents/trace-sandboxed-agents.mdx)
  and [Monitor Substrate](../../../docs/fern/pages/kubernetes/agent-sandboxes/monitor-substrate.mdx):
  supporting procedures.

Generate the file list with `git ls-files examples/agents/harbor-substrate`.
Edit the supplied configuration files in place as the tutorial describes.
The tutorial includes x86-64 (`amd64`) and ARM64 (`arm64`) image and placement
choices. Configurations default to `amd64`; ARM64 execution remains unvalidated.
The control plane, Harbor runner, sandbox workers, and Dynamo can use different
node architectures. Match the guest and task images to their sandbox worker.
Generated builds, certificates, and results use the ignored `.build/`, `.trace/`,
and `results/` directories. Keep credentials out of configuration files.
The example does not contain model deployment manifests or run data.

The adapter requires Harbor 0.16.0 and the env client pinned in `requirements.txt`.
The runner Dockerfile installs the upstream Pi adapter with the supplied
patches. Changes to these pins need fresh compatibility and end-to-end checks.

Build the runner image on a native x86-64 or ARM64 machine with
`docker build -f Dockerfile.harbor -t your-registry/harbor-runner:prepared .`.
Publish it and set its digest and architecture in `harbor-runner.yaml`.
Each Job downloads the pinned task definitions into an `emptyDir` and writes
results to the separate PVC. No live container installation or input copies
are required. Run Jobs sequentially with the supplied shared ConfigMap and
`ReadWriteOnce` PVC; delete the results reader before the next Job.
