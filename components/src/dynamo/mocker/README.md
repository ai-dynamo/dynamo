<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker engine

Run the mocker worker with `python -m dynamo.mocker`. The internal
`python -m dynamo.mocker._worker` entry point remains available for managed templates.

The user-facing availability notice lives at
[Simulate a Kubernetes Deployment with Mocker](../../../../docs/fern/pages/kubernetes/operations/simulation-with-dynosim/mocker-live-simulation.mdx).

Useful adjacent references:

- Aggregated deployment example: [`examples/backends/mocker/deploy/agg.yaml`](../../../../examples/backends/mocker/deploy/agg.yaml)
- Disaggregated deployment example: [`examples/backends/mocker/deploy/disagg.yaml`](../../../../examples/backends/mocker/deploy/disagg.yaml)
- Global planner mocker example: [`examples/global_planner/global-planner-mocker-test.yaml`](../../../../examples/global_planner/global-planner-mocker-test.yaml)
