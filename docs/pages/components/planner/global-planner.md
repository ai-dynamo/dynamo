---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Planner
---

Centralized scaling execution service for hierarchical planner deployments.

The Global Planner receives scaling decisions from distributed planners and executes
replica updates against Kubernetes `DynamoGraphDeployment` resources.

## Overview

- Exposes a remote scaling endpoint for planner delegation
- Optionally authorizes caller namespaces
- Executes scaling through `KubernetesConnector`
- Returns operation status and observed replica counts
- Supports dry-run mode via `--no-operation`

## Runtime Endpoints

Given `DYN_NAMESPACE=<ns>`, this component serves:

- `<ns>.GlobalPlanner.scale_request`
- `<ns>.GlobalPlanner.health`

`health` returns:

- `status` (`healthy`)
- `component` (`GlobalPlanner`)
- `namespace`
- `managed_namespaces` (`all` when authorization is disabled)

## Usage

### Command Line

```bash
# Accept scale requests from any namespace
DYN_NAMESPACE=global-infra python -m dynamo.global_planner
```

```bash
# Restrict requests to specific planner namespaces
DYN_NAMESPACE=global-infra python -m dynamo.global_planner \
  --managed-namespaces app-ns-1 app-ns-2
```

```bash
# Dry-run mode (no Kubernetes updates)
DYN_NAMESPACE=global-infra python -m dynamo.global_planner --no-operation
```

### Arguments

Required environment variables:

- `DYN_NAMESPACE`: Dynamo namespace used to register runtime endpoints.

Optional environment variables:

- `POD_NAMESPACE`: Kubernetes namespace where Global Planner runs (defaults to `default` if unset).

CLI arguments:

- `--managed-namespaces <ns1> <ns2> ...`: Allowlist for `caller_namespace`. If omitted, accepts all namespaces.
- `--environment kubernetes`: Execution environment (currently only `kubernetes` is supported).
- `--no-operation`: Log incoming scale requests and return success without applying Kubernetes scaling.

## Scale Request Contract

The `scale_request` endpoint consumes `ScaleRequest` and returns `ScaleResponse`.

Request fields:

- `caller_namespace` (string): Namespace identity of the planner sending the request
- `graph_deployment_name` (string): Target `DynamoGraphDeployment` name
- `k8s_namespace` (string): Kubernetes namespace of the target deployment
- `target_replicas` (list): Desired replica targets
- `blocking` (bool, default `false`): Wait for scaling completion
- `timestamp` (optional float): Caller-provided request timestamp
- `predicted_load` (optional object): Caller-provided prediction context

`target_replicas` entries use:

- `sub_component_type`: `prefill` or `decode`
- `desired_replicas`: integer replica target
- `component_name`: optional component override

Response fields:

- `status`: `success` or `error`
- `message`: status detail
- `current_replicas`: map of observed replicas, for example `{"prefill": 3, "decode": 5}`

## Behavior

- If `--managed-namespaces` is set and `caller_namespace` is not authorized, Global Planner returns `error` and does not scale.
- In `--no-operation` mode, Global Planner logs the request and returns `success` with empty `current_replicas`.

## Related Documentation

- [Planner Guide](planner-guide.md) — Planner configuration and deployment workflow
- [Planner Design](../../design-docs/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
