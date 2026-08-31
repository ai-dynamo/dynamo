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
- Supports dry-run mode via `no_operation`

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
# Defaults: accept scale requests from any namespace, no GPU budget
DYN_NAMESPACE=global-infra python -m dynamo.global_planner
```

```bash
# From a config file (JSON or YAML)
DYN_NAMESPACE=global-infra python -m dynamo.global_planner \
  --config /etc/global-planner/config.yaml
```

```bash
# From an inline JSON string — a fixed-total deployment pinned at 16 GPUs
DYN_NAMESPACE=global-infra python -m dynamo.global_planner \
  --config '{"min_total_gpus": 16, "max_total_gpus": 16}'
```

### Arguments

Required environment variables:

- `DYN_NAMESPACE`: Dynamo namespace used to register runtime endpoints.

Optional environment variables:

- `POD_NAMESPACE`: Kubernetes namespace where Global Planner runs (defaults to `default` if unset).

CLI arguments:

- `--config <json-or-path>`: Inline JSON string or path to a JSON/YAML file holding the configuration below. Mirrors how `dynamo.planner` is configured. Omit to run with defaults.

Config fields:

- `managed_namespaces` (list): Allowlist for `caller_namespace`. If omitted, accepts all namespaces.
- `environment` (`"kubernetes"`): Execution environment (currently only `kubernetes` is supported).
- `no_operation` (bool): Log incoming scale requests and return success without applying Kubernetes scaling.
- `max_total_gpus` (int): Reject scale requests that would push the managed pools above the configured total GPU cap. `-1` (default) disables the ceiling.
- `min_total_gpus` (int): Floor for total GPUs across managed pools. Scale-down requests that would drop below the floor are denied unless they can be paired with a pending scale-up on another pool (intra-DGD or cross-DGD). `-1` (default) disables the floor.
- `intent_cache_ttl_seconds` (float): How long a cached scale intent from a pool is considered fresh for pairing (default `360`). Should be at least `2x` the local planner's slowest tick interval so opposite-direction intents can overlap; throughput-based scaling ticks every `180s` by default, so `360` covers two ticks.

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

## Configuration

Every setting lives in one place: the `--config` document. There are no
per-setting CLI flags, so there is no precedence to reason about and no way for
two sources to disagree.

```yaml
# /etc/global-planner/config.yaml
managed_namespaces:
  - app-ns-1
  - app-ns-2
min_total_gpus: 16
max_total_gpus: 16
intent_cache_ttl_seconds: 360
```

The config is validated at startup. Two configurations that previously started
and then misbehaved at request time are now startup failures:

- `min_total_gpus > max_total_gpus` — a band no total GPU count satisfies, which
  would have denied every request on one edge or the other.
- `intent_cache_ttl_seconds <= 0` — no cached intent is ever fresh, silently
  disabling all pool pairing.

## Pool Priorities

Pool priorities declare which pools should be served first when several compete
for one GPU budget. They are declared at server start, in `--config`:

```yaml
priority:
  default: 100                       # pools no selector matches
  pools:
    - selector: prod/chat            # participant: every pool under it
      priority: 900
    - selector: prod/chat/prefill    # one pool: overrides the line above
      priority: 950
    - selector: dev/*                # any deployment in the dev namespace
      priority: 10
```

**Higher numbers are more important.** This matches Kubernetes `PriorityClass`
and `nvext.agent_hints.priority`. It is the opposite of the plugin-stage
`priority` in `dynamo.planner`, where smaller is more authoritative — that one
orders pipeline stages, this one orders capacity allocation between pools.

### Selectors

A selector is a slash-separated path matched against `<participant_id>/<sub_type>`
— today `<k8s namespace>/<deployment>/<sub_type>`, though nothing hard-codes that
depth.

| Pattern | Matches |
|---------|---------|
| `*` | exactly one segment |
| `**` | any number of segments, including none |
| a selector shorter than the pool path | everything beneath it |

So `prod/chat` selects every pool of that deployment, `prod/chat/prefill` selects
one, `dev/*` selects every deployment in `dev`, and `a/*/prefill` matches
`a/b/prefill` but not `a/b/c/prefill` — a single `*` never spans a `/`. Use `**`
when you do want it to: `a/**/prefill` matches both. Deeper hierarchies work
without special-casing, e.g.
`global-pool/east-coast-regions/*/long-context/*`.

The most specific match wins, independent of the order entries appear in the
file. Specificity is the number of segments named exactly, then depth — so
`prod/chat/prefill` beats `prod/chat`, which beats `prod/*`, which beats `**`.
Ties fall back to declaration order. Pools matching nothing take
`priority.default`, which is also what a pool this GlobalPlanner has never seen
receives.

### Conditional priorities

A policy is always an ordered, first-match-wins list of rules whose final rule is
unconditional. `priority: <n>` is shorthand for exactly one such rule:

```yaml
    - selector: prod/chat
      rules:
        - priority: 900
```

Rule *conditions* — "this priority while traffic is above X", "that priority
while the pool is breaching its SLA" — are not implemented yet, and a config that
declares one is rejected at startup rather than silently treated as
always-matching. The structure is in place so adding them does not change the
declaration surface or any caller.

### How priorities affect scaling

Priorities order **pair-partner selection**, which is how the Global Planner
moves capacity between pools when one alone would breach the budget band. The
ordering inverts with the direction of the transfer:

| Request | Partners are | Served first |
|---------|--------------|--------------|
| Scale-up (needs GPUs) | donors giving GPUs up | the **least** important pool (lowest number) |
| Scale-down below the floor (frees GPUs) | recipients taking GPUs on | the **most** important pool (highest number) |

Priority is the leading term, ahead of the preference for keeping a transfer
inside one DGD. When every pool resolves to the same priority the term is
constant, so an unconfigured Global Planner arbitrates exactly as it did before
priorities existed.

Priority does **not** add new denials. Pairing only ever consumes a scale
*intent a pool already published*, so a high-priority pool never has capacity
taken from it against its will — it is only paired when it had already decided
to scale down. Taking capacity from a pool that did not ask to give it up is
proactive reclaim, which is not implemented.

## Behavior

- If `managed_namespaces` is set and `caller_namespace` is not authorized, Global Planner returns `error` and does not scale.
- In `no_operation` mode, Global Planner logs the request and returns `success` with empty `current_replicas`.

### Minimum GPU budget and pool arbitration

When `min_total_gpus` is set the Global Planner enforces a floor on total GPUs across all managed DGDs. Combined with `max_total_gpus`, this lets you run a *fixed-size* deployment that scales load between pools without changing the total.

**Arbitration across all pools.** Every scale request that would breach the budget band triggers a search for opposite-direction pending intents on any other pool the Global Planner knows about — prefill ↔ decode within the same DGD, or across two different DGDs entirely (e.g., multiple agg DGDs sharing a cluster-wide budget).

**Multi-partner packing**: the search packs as many qualifying partners as fit. Candidates are tried in ascending `abs(delta_gpu)` order, fully admitted while the running total stays inside `[min - tolerance, max]`. When the next candidate's full inclusion would push past the band's far edge, the algorithm tries **partial consumption** — applying an intermediate replica count between the partner's current and its cached `last_desired` — that lands at the band edge. The cached intent's `last_desired` is **not** mutated by a partial application; the residual stays pending and can pair with future requests.

If no packing brings totals into the band, the request is rejected (`REJECTED`) and nothing is applied.

When the selected partners span multiple DGDs:

- **Intra-DGD partners** (same DGD as the request) are merged into the request's atomic `set_component_replicas` call — one patch updates request-side and intra-DGD partner pools together.
- **Cross-DGD partners** (different DGD) get their own per-DGD patch.
- **Direction-aware order**: scale-down DGDs apply first (freeing GPUs), then scale-up DGDs. If any second-or-later patch fails, the first has already landed and the system self-corrects from the new state on the next tick.

This scope is needed for "multiple agg pools sharing a budget" deployments such as [`examples/global_planner/global-planner-gpu-budget.yaml`](../../../../examples/global_planner/global-planner-gpu-budget.yaml).

**Tolerance for asymmetric pools.** When two paired pools have different `resources.limits.gpu` per replica, a single-worker step cannot always exactly cancel. Paired transfers may land up to `max(gpu_per_replica across the paired pools)` **below** `min` so the pair can still rebalance in whole-worker steps. `max` is a hard cluster-capacity bound and is never relaxed — pairs whose post-transfer total would exceed `max` are denied. Standalone (non-paired) requests must stay strictly within `[min, max]`.

**Intent cache.** Each scale request updates a per-pool cache with the pool's most recent desired replica count and a timestamp. Entries are eligible as pair partners when they are within `intent_cache_ttl_seconds` of the current time and the pool's cached `desired` still differs from its current Kubernetes replica count (i.e., the intent is *pending*, not yet satisfied). An entry whose desired equals current is considered satisfied and is skipped when looking for partners.

**Soft floor at startup.** If the discovered initial total GPUs is below `min_total_gpus`, the Global Planner logs a warning and continues. Scale-down requests that breach the floor are still denied, and natural scale-ups from local planners will drift toward the floor. No proactive fill is issued; if load is permanently low, the deployment may remain below the floor. The floor is a target enforced on the way down, not a hard invariant.

## Related Documentation

- [Planner Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-guide.md) — Planner configuration and deployment workflow
- [Global Planner Deployment Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/global-planner-guide.md) — Deployment patterns for `GlobalPlanner`, including multi-model coordination and single-endpoint multi-pool workflows
- [Planner Design](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
