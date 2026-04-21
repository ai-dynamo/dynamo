<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Global Planner

Centralized scaling execution service for multi-DGD planner deployments.

The Global Planner receives scaling decisions from local planners and executes
replica updates against Kubernetes `DynamoGraphDeployment` resources. It is useful
whenever multiple DGDs should delegate scaling through one centralized component,
whether or not those DGDs sit behind a single shared endpoint.

## What Problem This Solves

Without `GlobalPlanner`, each DGD's local planner scales only its own deployment directly.
That is fine for isolated deployments, but it becomes awkward when you want one place to:

- apply centralized scaling policies across multiple DGDs
- enforce shared constraints such as authorization or total GPU budget
- coordinate scaling for a single-endpoint, multi-pool deployment

`GlobalPlanner` solves that by becoming the common scale-execution endpoint for multiple local planners.

## Deployment Patterns

`GlobalPlanner` is used in two common patterns:

1. **Centralized scaling across independent DGDs**
   Each DGD keeps its own normal local planner, but the local planners delegate scale execution to one `GlobalPlanner`. This is useful when separate deployments or models should share a global policy such as a total GPU budget. You do **not** need `GlobalRouter` or a single shared endpoint for this pattern.
2. **Hierarchical single-endpoint deployment**
   Multiple pool DGDs for one model sit behind one public `Frontend` and one `GlobalRouter`. Each pool still has its own local planner, and those local planners delegate scaling to `GlobalPlanner`.

## Terminology

- **SLA Planner**: The normal `dynamo.planner` component that computes desired replica counts from SLA targets, profiles, and/or metrics.
- **Local planner**: An instance of that planner running inside one DGD or one pool.
- **GlobalPlanner**: The centralized execution and policy layer that receives scale requests from local planners and applies them to target DGDs.
- **Hierarchical planner**: An architecture term, not a separate binary. In practice it means multiple local planners feeding one `GlobalPlanner`, often together with `GlobalRouter`.

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

```bash
# Enforce a maximum total GPU budget across managed pools
DYN_NAMESPACE=global-infra python -m dynamo.global_planner --max-total-gpus 16
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
- `--max-total-gpus <n>`: Reject scale requests that would push the managed pools above the configured total GPU cap.

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

- [Planner Guide](../../../../docs/components/planner/planner-guide.md) — Planner configuration and deployment workflow
- [Global Planner Deployment Guide](../../../../docs/components/planner/global-planner.md) — Deployment patterns for `GlobalPlanner`, including multi-model coordination and single-endpoint multi-pool workflows
- [Planner Design](../../../../docs/design-docs/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
