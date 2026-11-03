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
[DEP #16064](https://github.com/ai-dynamo/dynamo/issues/16064). Local Planners
publish demand; the Global Planner grants allocations and can reclaim replicas
from lower-priority pools above their minima. The legacy budget and pairing
behavior described above remains the default when this setting is absent.

### Supported deployment

This mode requires one Global Planner process, with no overlapping replacement
process, and exclusive ownership of replica targets. Stop the old process before
starting its replacement. Every enrolled DGD must be in `POD_NAMESPACE`, contain
one prefill-only or decode-only worker component, and use the component workload
provider with ordinary single-node Deployment workers and a scaling adapter
(`scalingAdapter: {}`). Each replica must request the fixed, positive scalar
`nvidia.com/gpu` cost in its pool policy. CPU-only frontends and planners in the
same DGD are allowed. Pools are separate DGDs; aggregated/composite workers,
Grove, multinode workers, DRA, GPU-free mockers, and changing GPU shapes are not
supported by this mode.

Every pool must retain at least one replica. The existing operator considers
zero-target Deployments ready before downstream scale reconciliation has
necessarily completed; this mode cannot prove release for zero targets using
existing Planner permissions. Offline planning still supports zero minima.

Configure every Local Planner with `environment: global-planner` and the same
`global_planner_namespace`. Use single-role scaling without retuning, worker
rollouts, HPA, manual scaling, or other replica writers. Keep node scheduling and
GPU capacity sufficient for every allowed target. The shared budget counts only
the enrolled pools, not unrelated workloads on the cluster.

Writer annotations on scaling adapters fence late writes from a stopped
controller using resource-version checks. They are cooperative ownership checks,
not leader election or authorization against other Kubernetes writers. Runtime
access must be restricted to trusted planners: caller namespace bindings validate
the claimed caller and target but do not authenticate a principal.

### Configuration

Do not combine `managed_allocation` with legacy `managed_namespaces`, budget,
priority, GPU-cost overrides, or `no_operation`. The offline command below can
preview policy decisions without executing them.

```yaml
# Start with two ready replicas in each pool: 8 + 8 GPUs.
managed_allocation:
  poll_interval_seconds: 2
  operation_timeout_seconds: 1800
  caller_namespaces:
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
`--config <path-to-config>`. `caller_namespaces` maps each pool ID to its Local
Planner's Dynamo namespace. Requests must match the enrolled Kubernetes
namespace, DGD, role, and optional component name; each request contains exactly
one replica target. Existing Planner permissions to read DGDs, adapters and Pods, and to patch
adapters and their scale subresource suffice; no new CRD or storage service is
introduced.

### Allocation and execution

The controller retains each pool's latest desired count separately from its
grant and observed workers. Valid unchanged Local Planner recommendations also
publish demand, so a hold can cancel an older unsatisfied request. Missing
metrics and scaling-only observations do not invent a replacement demand.
Demand remains until a new valid recommendation or controller restart; a lost
Local Planner does not automatically withdraw its last demand.

Growth uses available capacity first. Forced reclamation protects equal- and
higher-priority pools and every pool's minimum. A pool can voluntarily give up
capacity above its own retained demand regardless of priority. This allows the
example to move from `8/8` to `12/4` for interactive demand, then back to `8/8`
when interactive demand falls and batch still wants its previous allocation.
Pending requests are reconsidered on the bounded reconciliation cadence in
priority order. There is no fairness or starvation guarantee across priorities.

Every operation starts with a complete feasible allocation vector. All
capacity-changing operations are serialized globally:

1. Commit the plan and reserve requester capacity in memory.
2. Apply donor reductions through the scaling adapters.
3. Wait for current-generation readiness and replica status, and for the removed
   worker Pods to disappear. Pending and terminating Pods remain charged.
4. Apply requester growth and wait for the complete plan to settle.

Demand and status endpoints remain responsive while Kubernetes operations and
settlement are pending. A newer demand does not rewrite a transfer already in
progress; it is considered after that transfer settles. The global floor applies
to committed allocations and settled targets. Ready serving capacity can fall
below that floor during the release-before-start interval.

Release checks assume healthy nodes and normal graceful Pod deletion. Force
deletion, node loss/partition, and out-of-band resource or label changes are
outside the physical-release guarantee. Draining uses existing worker shutdown
behavior and configured termination grace; this change does not extend those
timeouts or guarantee that every in-flight request completes before termination.

### Status and recovery

Managed responses add `allocation` with `pool_id`, `controller_id`,
`request_id`, `transfer_id`, `phase`, `reason`, and separate desired, granted,
observed, ready, and terminating replica counts. Unknown counts are `null`, never
zero. Status is available without changing demand through
`<ns>.GlobalPlanner.allocation_status`, accepting the caller namespace, Kubernetes
namespace, and DGD name. `scaling` means accepted/pending; `success` means settled;
`error` with phase `recovery_required` means mutations have stopped. The health
endpoint also reports allocation phase and reason; transport health alone does
not imply that allocations are progressing.

Managed requests return promptly even when `blocking` is true. The updated
Local Planner connector recognizes managed pending responses and polls status
while continuing to use Kubernetes for actual worker readiness. A known managed
recovery status is an observable hold; an unavailable status RPC clears cached
status and still permits local Kubernetes observation. Older Local
Planners without request IDs are rejected. The most recent 256 request IDs per
pool are deduplicated; a retry cannot change its payload or restore an older
cached demand. Clients must not replay requests outside that window or across a
controller restart. The bundled connector retries connection establishment only,
not requests that have been sent.

Unknown observations, target/identity/shape drift, uncertain writes, and
settlement timeouts freeze mutations and retain existing reservations. There is
no automatic rollback or endpoint that clears this state. For manual recovery:

1. Stop the controller and ensure no old instance remains.
2. Inspect every enrolled DGD, adapter, and worker Pod; repair targets and wait
   for all pools to become ready and settled inside both pool and global bounds.
3. Restart the controller. It fences every adapter and reconstructs only a
   complete settled allocation. Local Planners then republish current demand.
   If a Local Planner had an ambiguous scaling-request transport failure, also
   restart it: the existing effect-submission guard retains that unknown outcome
   until restart. Read-only status outages do not trigger that guard.

Transfer records, reservations, and retained demand are in memory. A restart
that cannot establish a settled allocation within all bounds requires manual
repair. A valid settled donor-only intermediate allocation can become the new
baseline; its old transfer intent is lost and current demand must be republished.
The controller never resumes an assumed plan or spends unverified capacity. Durable
recovery and HA require a later slice of the DEP.

## Offline Allocation Planning (Experimental)

[DEP #16064](https://github.com/ai-dynamo/dynamo/issues/16064) proposes active
reclamation from lower-priority pools. The standalone allocation planner below
implements the policy and settled-allocation calculation for that proposal. It
does not contact Kubernetes or execute a transfer. The running service consumes
the same policy under `managed_allocation.policy` when managed mode is enabled.

Each pool explicitly owns one prefill-only or decode-only worker component in
its own DGD. GPU cost per replica must be a positive integer; zero-cost and
composite pools are outside this allocation model. Identities are exact, without
selectors. Higher static priorities are more important.

For example, save this as `allocation-policy.json`:

```json
{
  "min_total_gpus": 16,
  "max_total_gpus": 16,
  "pools": [
    {
      "pool_id": "interactive",
      "participant_id": "prod/interactive",
      "component_name": "DecodeWorker",
      "role": "decode",
      "gpu_per_replica": 4,
      "min_gpus": 4,
      "max_gpus": 16,
      "priority": 100
    },
    {
      "pool_id": "batch",
      "participant_id": "prod/batch",
      "component_name": "DecodeWorker",
      "role": "decode",
      "gpu_per_replica": 4,
      "min_gpus": 4,
      "max_gpus": 16,
      "priority": 10
    }
  ]
}
```

Save the complete current allocation as `allocations.json`:

```json
{"interactive": 8, "batch": 8}
```

Then inspect a request for three interactive replicas:

```bash
python -m dynamo.global_planner.plan_allocation \
  --policy allocation-policy.json \
  --allocations allocations.json \
  --pool interactive --replicas 3
```

The resulting plan assigns 12 GPUs to `interactive` and 4 to `batch`, with exact
targets of three and one replicas. No voluntary downscale request from `batch`
is needed to calculate the plan. Both pools' minima and the exact shared total
remain satisfied. Asking `batch` to grow cannot reclaim from `interactive`.

Pool minima round up and maxima round down to whole replicas. For example, a
pool costing 4 GPUs per replica with bounds `[5, 15]` has effective bounds
`[8, 12]`. A configuration is rejected if a pool's interval becomes empty or if
no actual allocation vector satisfies the exact global bounds. Scalar sums of
pool minima and maxima alone are insufficient to establish feasibility.

The supplied allocation must name every configured pool, use whole replicas,
and already satisfy all bounds. Desired replicas are clamped to the pool's
effective interval; the output retains both desired and effective targets.
Voluntary downscaling retains enough warm replicas to preserve the global
allocation floor. Growth uses spare capacity first, then considers reductions
only from strictly lower-priority pools above their minima. Among feasible
plans, the search minimizes reclamation from higher donor priority classes
first, then excess reclamation, then the number of affected pools. Pool IDs
break remaining ties deterministically.

Use `--unavailable-pool <id>` repeatedly to exclude pools from changes. This is
an offline eligibility constraint, not reservation accounting. The input
represents settled committed allocations; it cannot describe in-flight
transfers or physical holdings. The command returns a complete hypothetical
plan or a pending decision with no partial plan. Exit status is `0` for a plan,
`1` for pending, and `2` for invalid input. Reasons identify normalized pool
bounds, retained global floor, a busy requester, insufficient eligible capacity,
incompatible replica sizes, or search exhaustion. `--max-search-steps` bounds
allocation search work (default 100,000 units, accounting for candidate
transitions and score width). Policy feasibility validation has its own fixed
100,000-unit limit; an exhausted proof is rejected as unverified.

The Python entry points are `AllocationPolicy` in `allocation_policy.py` and
`plan_allocation` in `allocation.py`. They are independent of backend execution.
The allocation floor here applies to settled targets: it does not establish a
live-serving floor during a release-before-start transfer. The managed mode
above supplies serialized execution, release observation, and conservative
restart checks under its deployment restrictions. Durable transfer recovery
and broader deployment support remain outside this slice.

## Related Documentation

- [Planner Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-guide.md) — Planner configuration and deployment workflow
- [Global Planner Deployment Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/global-planner-guide.md) — Deployment patterns for `GlobalPlanner`, including multi-model coordination and single-endpoint multi-pool workflows
- [Planner Design](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-design.md) — Planner architecture and algorithms

Planners delegate to this service when planner config uses `environment: "global-planner"` and sets `global_planner_namespace`.
