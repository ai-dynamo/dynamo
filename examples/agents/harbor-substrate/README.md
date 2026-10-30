<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Run SWE-bench Verified with Harbor and Pi in Agent Substrate sandboxes.
Use a model served by NVIDIA Dynamo or a compatible external API.

Start with [Run SWE-bench Verified with Harbor, Pi, and Substrate](../../../docs/fern/pages/use-cases/agents/harbor-substrate.mdx).
That page owns the walkthrough, including prerequisites, prepared task images,
golden snapshots, one trial, a concurrent batch, result inspection, and cleanup.

This directory contains the runnable files:

- `substrate_environment.py`: Harbor's external environment provider.
- `tutorial_pi.py`: upstream Pi adapter with runtime-only credential injection.
- `pi_trajectory.py`: conversion of recorded Pi events to Harbor ATIF.
- `prepare_workload.py`: task copies, image locks, actor templates, and Harbor configuration.
- `Dockerfile.pi`: prepared task image with pinned Pi and provider dependencies.
- YAML manifests: environment API, runner, worker pool, task template, and network policy.
- `*.patch`: compatibility changes applied to the pinned upstream sources.
- [Install Substrate integration](INSTALL.md), [Capture inference and tools](TRACING.md),
  and [Collect Substrate metrics](METRICS.md): supporting procedures.

Generate the file list with `git ls-files examples/agents/harbor-substrate`.
Keep local settings, credentials, builds, and results under the ignored `.local/`
directory. The example does not contain model deployment manifests or run data.

The adapter requires Harbor 0.16.0 and the env client pinned in `requirements.txt`.
The tutorial installs the upstream Pi adapter with the supplied patches. Changes
to these pins need a fresh compatibility check. See the tutorial's validation
section for the distinction between component checks and a complete deployment.
