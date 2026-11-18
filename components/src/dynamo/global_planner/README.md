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
- `<ns>.GlobalPlanner.allocation_status` (managed allocation mode only)

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

- `managed_namespaces` (non-empty list): Allowlist for `caller_namespace`. If omitted or `null`, accepts all namespaces; an empty list is rejected.
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

- `status`: `success`, `scaling`, `rejected`, or `error`
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

## Declared GPU Costs

The GPU budget is computed from each pool's `gpu_per_replica`, read from the
DGD's `resources.limits["nvidia.com/gpu"]`. A pool that requests no GPU has no
cost to read — and with a budget enabled that fails the whole snapshot, so every
scale request errors. That rules out **mocker topologies**, which exist to
exercise a deployment without hardware but could not take part in the budget
they are meant to help test.

Declare what such a pool costs instead:

```yaml
gpu_cost:
  pools:
    - selector: sachalm/dsv4-**
      gpu_per_replica: 8
    - selector: sachalm/gpt-oss-**
      gpu_per_replica: 4
```

Selectors work exactly as they do for priorities below — same syntax, same
most-specific-wins rule, same shared implementation.

**A declared cost is a fallback, not an override.** It applies only where the DGD
is silent, so a pool that really does request GPUs is always charged what the
cluster says and a stale config can never make the planner under-count real
hardware. An explicit `gpu_per_replica: 0` is different from declaring nothing:
zero keeps a pool out of budget totals, while undeclared leaves it unreadable.

This mirrors the local planner, whose `_initialize_gpu_counts` already falls back
to configured GPU counts for "mockers that don't specify GPU resources".

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
file. Specificity is the number of segments named exactly, then literal
coverage within wildcard segments, then depth — so
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

Priority is the leading term. Among equal-priority pools, partners inside the
requesting DGD are preferred, followed by smaller absolute GPU changes. This
locality preference also applies when no priorities are configured, since every
pool then resolves to the same priority. It intentionally changes the previous
ordering, which preferred smaller absolute GPU changes and used locality only
to break ties: a same-DGD partner can now precede a cross-DGD partner with a
smaller GPU change.

Priority does **not** add new denials. Pairing only ever consumes a scale
*intent a pool already published*, so a high-priority pool never has capacity
taken from it against its will — it is only paired when it had already decided
to scale down. Taking capacity from a pool that did not ask to give it up is
proactive reclaim, which is available only in the opt-in managed mode below.

## Behavior

- If `managed_namespaces` is set and `caller_namespace` is not authorized, Global Planner returns `error` and does not scale.
- In `no_operation` mode, Global Planner logs the request and returns `success` with empty `current_replicas`.

### Minimum GPU budget and pool arbitration

When `min_total_gpus` is set the Global Planner enforces a floor on total GPUs across all managed DGDs. Combined with `max_total_gpus`, this lets you run a *fixed-size* deployment that scales load between pools without changing the total.

**Arbitration across all pools.** Every scale request that would breach the budget band triggers a search for opposite-direction pending intents on any other pool the Global Planner knows about — prefill ↔ decode within the same DGD, or across two different DGDs entirely (e.g., multiple agg DGDs sharing a cluster-wide budget).

**Multi-partner packing**: the search packs as many qualifying partners as fit. Candidates are tried in ascending `abs(delta_gpu)` order, fully admitted while the running total stays inside `[min - tolerance, max]`. When the next candidate's full inclusion would push past the band's far edge, the algorithm tries **partial consumption** — applying an intermediate replica count between the partner's current and its cached `last_desired` — that lands at the band edge. The cached intent's `last_desired` is **not** mutated by a partial application; the residual stays pending and can pair with future requests.

If no packing brings totals into the band, the request is rejected (`REJECTED`) and nothing is applied.

When the selected partners span multiple DGDs:

- **Intra-DGD partners** (same DGD as the request) share the request's `set_component_replicas` call, which patches component targets separately rather than transactionally.
- **Cross-DGD partners** (different DGD) get their own per-DGD patch.
- **Direction-aware order**: scale-down DGDs apply first (freeing GPUs), then scale-up DGDs. If any second-or-later patch fails, the first has already landed and the system self-corrects from the new state on the next tick.

This scope is needed for "multiple agg pools sharing a budget" deployments such as [`examples/global_planner/global-planner-gpu-budget.yaml`](../../../../examples/global_planner/global-planner-gpu-budget.yaml).

**Tolerance for asymmetric pools.** When two paired pools have different `resources.limits.gpu` per replica, a single-worker step cannot always exactly cancel. Paired transfers may land up to `max(gpu_per_replica across the paired pools)` **below** `min` so the pair can still rebalance in whole-worker steps. `max` is a hard cluster-capacity bound and is never relaxed — pairs whose post-transfer total would exceed `max` are denied. Standalone (non-paired) requests must stay strictly within `[min, max]`.

**Intent cache.** Each scale request updates a per-pool cache with the pool's most recent desired replica count and a timestamp. Entries are eligible as pair partners when they are within `intent_cache_ttl_seconds` of the current time and the pool's cached `desired` still differs from its current Kubernetes replica count (i.e., the intent is *pending*, not yet satisfied). An entry whose desired equals current is considered satisfied and is skipped when looking for partners.

**Soft floor at startup.** If the discovered initial total GPUs is below `min_total_gpus`, the Global Planner logs a warning and continues. Scale-down requests that breach the floor are still denied, and natural scale-ups from local planners will drift toward the floor. No proactive fill is issued; if load is permanently low, the deployment may remain below the floor. The floor is a target enforced on the way down, not a hard invariant.

## Managed Allocation (Experimental)

Set `managed_allocation` to enable the first live slice of
[DEP #16064](https://github.com/ai-dynamo/dynamo/issues/16064): Local Planners
publish demand, and the Global Planner grants allocations, reclaiming replicas
from lower-priority pools above their minima. Without this setting the legacy
budget and pairing behavior above is unchanged.

### Requirements

- **One controller.** Run a single Global Planner process; stop the old one
  before starting a replacement. It must be the only writer of replica targets
  (no HPA, manual scaling, worker rollouts or retuning).
- **One pool per DGD.** Each enrolled DGD lives in `POD_NAMESPACE` and has one
  prefill-only or decode-only worker component (CPU-only frontends and planners
  are allowed). Workers use the component workload provider, ordinary
  single-node Deployments, and `scalingAdapter: {}`.
- **Fixed GPU shape.** Every replica requests exactly the pool's scalar
  `nvidia.com/gpu` cost. Aggregated/composite workers, Grove, multinode, DRA,
  GPU-free mockers and changing shapes are unsupported.
- **Positive minimum.** Every pool keeps at least one replica: existing
  Planner permissions cannot prove that a zero-target Deployment released its
  GPUs.
- **Trusted callers.** `caller_namespaces` checks the claimed caller, but does
  not authenticate it. Scaling-adapter writer annotations fence late writes
  from a stopped controller; they are not leader election.

The shared budget counts only enrolled pools. Keep node capacity sufficient for
every allowed target. Existing Planner RBAC (read DGDs, adapters and Pods; patch
adapters and their scale subresource) is sufficient.

### Configuration

`managed_allocation` cannot be combined with `managed_namespaces`, legacy budget,
priority or GPU-cost settings, or `no_operation`.

```yaml
# Start with two ready replicas in each pool: 8 + 8 GPUs.
managed_allocation:
  poll_interval_seconds: 2
  operation_timeout_seconds: 1800
  caller_namespaces:          # pool ID -> Local Planner Dynamo namespace
    interactive: chat-planner
    batch: batch-planner
  policy:
    min_total_gpus: 16
    max_total_gpus: 16
    pools:
      - pool_id: interactive
        participant_id: prod/interactive
        component_name: DecodeWorker
        role: decode
        gpu_per_replica: 4
        min_gpus: 4
        max_gpus: 12
        priority: 100
      - pool_id: batch
        participant_id: prod/batch
        component_name: DecodeWorker
        role: decode
        gpu_per_replica: 4
        min_gpus: 4
        max_gpus: 12
        priority: 10
```

Run the service with `POD_NAMESPACE=prod`, `DYN_NAMESPACE=global-infra`, and
`--config <path>`. Configure each Local Planner with
`environment: global-planner`, the shared `global_planner_namespace`, single-role
scaling, and `global_planner_managed_allocation: true`. With that flag, stable
same-count recommendations are also published, so a hold cancels older
unsatisfied demand; legacy deployments never send them.

### Allocation and execution

The controller tracks each pool's latest demand separately from its grant and
its observed workers. Pending demand is reconsidered every poll in priority
order (no fairness guarantee). Growth uses spare capacity first; forced
reclamation only takes from strictly lower-priority pools above their minima,
while any pool may voluntarily return capacity above its own demand. In the
example, interactive demand moves `8/8` to `12/4`; when it falls and batch
still wants its previous allocation, capacity returns to `8/8`.

Capacity changes are serialized, one complete plan at a time:

1. Commit the plan and reserve the requester's capacity in memory.
2. Reduce donors through their scaling adapters.
3. Wait until donor Pods are gone. Pending, failed and terminating Pods stay
   charged.
4. Grow the requester and wait for the whole plan to settle.

Demand and status requests stay responsive throughout; newer demand is planned
after the current transfer settles. The global floor applies to committed
targets, so ready serving capacity can dip during step 3. A worker restart
outside a transfer only delays planning. Release checks assume graceful Pod
deletion and healthy nodes; draining uses existing worker shutdown behavior.

### Status and recovery

Managed responses include `allocation`: `pool_id`, `controller_id`,
`request_id`, `transfer_id`, `phase`, `reason`, and separate desired, granted,
observed, ready and terminating replica counts (`null` when unknown). The
`<ns>.GlobalPlanner.allocation_status` endpoint returns the same status without
changing demand, and the health endpoint reports the controller phase. A grant
is a target, never evidence of ready workers: the Local Planner keeps reading
readiness from Kubernetes.

| `status` | Meaning |
| --- | --- |
| `scaling` | Demand accepted; a transfer is pending or in progress |
| `success` | The pool's allocation is settled |
| `error` + phase `recovery_required` | Mutations stopped; manual recovery needed |

Requests must carry a `request_id` (older Local Planners are rejected). The last
256 IDs per pool are deduplicated; the bundled connector never re-sends a request
it has sent.

Unknown observations, target, identity or shape drift, uncertain writes and
settlement timeouts freeze mutations while keeping reservations. To recover:

1. Stop the controller and make sure no old instance remains.
2. Repair every enrolled DGD, adapter and Pod until all pools are settled within
   pool and global bounds.
3. Restart the controller. It fences every adapter and adopts only a complete,
   settled, valid allocation; Local Planners then republish demand. Also restart
   any Local Planner whose scale request failed in transport, since that unknown
   outcome is held until restart.

Transfers, reservations and demand live in memory; durable recovery and HA are
later slices of the DEP.

## Offline Allocation Planning (Experimental)

`plan_allocation` computes the same decision the managed controller would make,
without contacting Kubernetes. Save the policy (the `policy` block above, as
JSON) as `allocation-policy.json` and the committed GPU vector as
`allocations.json`, e.g. `{"interactive": 8, "batch": 8}`, then run:

```bash
python -m dynamo.global_planner.plan_allocation \
  --policy allocation-policy.json \
  --allocations allocations.json \
  --pool interactive --replicas 3
```

The plan assigns 12 GPUs to `interactive` and 4 to `batch` (targets 3 and 1);
asking `batch` to grow cannot reclaim from `interactive`. The output is either a
complete plan or a pending `reason`: `pool_minimum`/`pool_maximum` (demand was
clamped), `global_minimum`, `requester_busy`, `insufficient_capacity`,
`granularity` (replica sizes cannot fit), or `search_limit`. Exit status is `0`
for a plan, `1` for pending and `2` for invalid input. `--unavailable-pool <id>`
(repeatable) excludes pools from changes.

Semantics:

- Pool minima round up and maxima round down to whole replicas, so 4-GPU
  replicas with bounds `[5, 15]` become `[8, 12]`. A policy is rejected if no
  whole-replica vector satisfies the global bounds.
- The input allocation must name every pool and satisfy all bounds. Voluntary
  reductions keep enough warm replicas to preserve the global floor; growth is
  never partially granted.
- Among feasible plans, the search minimizes reclamation from the highest donor
  priority first, then excess reclamation, then the number of affected pools;
  pool IDs break ties.
- `--max-search-steps` bounds the search (default 100,000 work units); policy
  feasibility has its own fixed limit and is rejected if unproven.

## Related Documentation

- [Planner Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-guide.md) — Planner configuration and deployment workflow
- [Global Planner Deployment Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/global-planner-guide.md) — Deployment patterns for `GlobalPlanner`, including multi-model coordination and single-endpoint multi-pool workflows
- [Planner Design](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
