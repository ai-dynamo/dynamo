<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker engine

The public `python -m dynamo.mocker` CLI is removed while online simulation is unavailable. The
engine and private worker launcher remain for Dynamo internals, tests, and managed templates.

The user-facing availability notice lives at
[Simulate a Kubernetes Deployment with Mocker](../../../../docs/fern/pages/kubernetes/operations/simulation-with-dynosim/mocker-live-simulation.mdx).

Useful adjacent references:

- Aggregated deployment example: [`examples/backends/mocker/deploy/agg.yaml`](../../../../examples/backends/mocker/deploy/agg.yaml)
- Disaggregated deployment example: [`examples/backends/mocker/deploy/disagg.yaml`](../../../../examples/backends/mocker/deploy/disagg.yaml)
- Global planner mocker example: [`examples/global_planner/global-planner-mocker-test.yaml`](../../../../examples/global_planner/global-planner-mocker-test.yaml)
