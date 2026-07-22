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

```bash
# Fixed-total deployment: pin the cluster at 16 GPUs with cross-pool transfers
DYN_NAMESPACE=global-infra python -m dynamo.global_planner \
  --min-total-gpus 16 --max-total-gpus 16
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
- `--min-total-gpus <n>`: Floor for total GPUs across managed pools. Scale-down requests that would drop below the floor are denied unless they can be paired with a pending scale-up on another pool (intra-DGD or cross-DGD). `-1` (default) disables the floor.
- `--intent-cache-ttl-seconds <s>`: How long a cached scale intent from a pool is considered fresh for pairing (default `360`). Should be at least `2x` the local planner's slowest tick interval so opposite-direction intents can overlap; throughput-based scaling ticks every `180s` by default, so `360` covers two ticks.

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

### Minimum GPU budget and pool arbitration

When `--min-total-gpus` is set the Global Planner enforces a floor on total GPUs across all managed DGDs. Combined with `--max-total-gpus`, this lets you run a *fixed-size* deployment that scales load between pools without changing the total.

**Arbitration across all pools.** Every scale request that would breach the budget band triggers a search for opposite-direction pending intents on any other pool the Global Planner knows about — prefill ↔ decode within the same DGD, or across two different DGDs entirely (e.g., multiple agg DGDs sharing a cluster-wide budget).

**Multi-partner packing**: the search packs as many qualifying partners as fit. Candidates are tried in ascending `abs(delta_gpu)` order, fully admitted while the running total stays inside `[min - tolerance, max]`. When the next candidate's full inclusion would push past the band's far edge, the algorithm tries **partial consumption** — applying an intermediate replica count between the partner's current and its cached `last_desired` — that lands at the band edge. The cached intent's `last_desired` is **not** mutated by a partial application; the residual stays pending and can pair with future requests.

If no packing brings totals into the band, the request is rejected (`REJECTED`) and nothing is applied.

When the selected partners span multiple DGDs:

- **Intra-DGD partners** (same DGD as the request) are merged into the request's atomic `set_component_replicas` call — one patch updates request-side and intra-DGD partner pools together.
- **Cross-DGD partners** (different DGD) get their own per-DGD patch.
- **Direction-aware order**: scale-down DGDs apply first (freeing GPUs), then scale-up DGDs. If any second-or-later patch fails, the first has already landed and the system self-corrects from the new state on the next tick.

This scope is needed for "multiple agg pools sharing a budget" deployments such as [`examples/global_planner/global-planner-gpu-budget.yaml`](../../../../examples/global_planner/global-planner-gpu-budget.yaml).

**Tolerance for asymmetric pools.** When two paired pools have different `resources.limits.gpu` per replica, a single-worker step cannot always exactly cancel. Paired transfers may land up to `max(gpu_per_replica across the paired pools)` **below** `min` so the pair can still rebalance in whole-worker steps. `max` is a hard cluster-capacity bound and is never relaxed — pairs whose post-transfer total would exceed `max` are denied. Standalone (non-paired) requests must stay strictly within `[min, max]`.

**Intent cache.** Each scale request updates a per-pool cache with the pool's most recent desired replica count and a timestamp. Entries are eligible as pair partners when they are within `--intent-cache-ttl-seconds` of the current time and the pool's cached `desired` still differs from its current Kubernetes replica count (i.e., the intent is *pending*, not yet satisfied). An entry whose desired equals current is considered satisfied and is skipped when looking for partners.

**Soft floor at startup.** If the discovered initial total GPUs is below `--min-total-gpus`, the Global Planner logs a warning and continues. Scale-down requests that breach the floor are still denied, and natural scale-ups from local planners will drift toward the floor. No proactive fill is issued; if load is permanently low, the deployment may remain below the floor. The floor is a target enforced on the way down, not a hard invariant.

## Related Documentation

- [Planner Guide](../../../../docs/components/planner/planner-guide.md) — Planner configuration and deployment workflow
- [Global Planner Deployment Guide](../../../../docs/components/planner/global-planner.md) — Deployment patterns for `GlobalPlanner`, including multi-model coordination and single-endpoint multi-pool workflows
- [Planner Design](../../../../docs/design-docs/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
