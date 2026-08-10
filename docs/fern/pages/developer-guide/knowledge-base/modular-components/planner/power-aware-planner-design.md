---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Power-Aware Planner Design
subtitle: Static GPU power-cap ownership, admission, projection, and rollout safety
---

> [!NOTE]
> **Tier 3 design documentation** for contributors and architects. For configuration and deployment,
> see [Planner Examples](planner-examples.md#power-aware-budget-scaling).

> [!WARNING]
> **Experimental.** The power-aware Planner adds a projected GPU power ceiling to NVIDIA Dynamo
> autoscaling. It admits or reduces replica proposals according to requested per-GPU caps and one
> DynamoGraphDeployment (DGD) budget. It does not measure power, prove that a cap reached the hardware,
> change a cap at runtime, or remediate a deployment that is already over budget.

## Decision Summary

| Decision | Current static behavior |
| --- | --- |
| Cap owner | Worker `podTemplate` annotation in the DGD |
| Admission owner | DGD validating webhook with fail-closed admission |
| Cap enforcer | Node-local Power Agent |
| Planner reads | Settled DGD intent plus list-only Pod state |
| Projection unit | Watts per replica: cap per GPU multiplied by GPUs per logical replica |
| Budget owner | `PlannerConfig.total_gpu_power_limit` for one DGD |
| Supported modes | `disagg`, `prefill`, and `decode` |
| Enforcement point | GPU clamp, rollout hold, then power clamp |
| Cap or topology changes | Delete and recreate the DGD |
| Total-budget changes | Restart the Planner |
| Proven guarantee | Admitted scale-ups do not push an in-budget ready baseline above the static requested-cap projection |
| Excluded guarantee | Effective caps, measured draw, or facility power remain within the budget |

## Goals and Non-Goals

The static design has four goals:

- Keep cap authorship with the workload definition.
- Keep the Planner read-only with respect to Pods and GPU power controls.
- Apply the power ceiling to built-in and external-plugin proposals at one final boundary.
- Preserve partial proposals and in-progress operator rollouts.

The design does not:

- observe instantaneous GPU power or the effective hardware cap;
- coordinate budgets across DGDs, namespaces, clusters, or racks;
- lower replica counts solely because the current deployment is over budget;
- retarget caps on a live DGD;
- enable power awareness for aggregated, virtual, replay, or Global Planner environments; or
- change the existing GPU-budget unit for multinode workers.

## Ownership and Data Flow

```mermaid
flowchart TD
    A[DGD worker podTemplate annotation] --> B[DGD validating webhook]
    B -->|accepted static tuple| C[Operator-rendered worker Pods]
    C -->|annotation| D[Power Agent]
    D -->|NVML or DCGM write| E[GPU hardware cap]
    A -->|settled DGD snapshot| F[Planner startup]
    C -->|startup annotation settlement| F
    F -->|cached watts per replica| G[Worker capabilities]
    C -->|runtime terminating-Pod state| H[Rollout stability]
    I[Built-in and external proposals] --> J[GPU budget clamp]
    J --> H
    G --> K[Power budget clamp]
    H --> K
    K -->|replica targets only| L[DGD scaling interface]
```

The shared contract is `dynamo.nvidia.com/gpu-power-limit`, expressed in watts per GPU:

```yaml
podTemplate:
  metadata:
    annotations:
      dynamo.nvidia.com/gpu-power-limit: "350"
```

The operator renders the annotation onto worker Pods. Every 15 seconds, the Power Agent lists and
reconciles Pods on its node and applies the requested limit through NVML or DCGM. The Planner resolves
the requested limit from the DGD, verifies its propagation on Pods before caching it, and never writes
the annotation.

This ownership boundary prevents the Planner and Power Agent from competing over a cap. It also
creates the main limitation: the Planner knows the requested cap, while the Power Agent knows the
effective cap and whether the hardware write succeeded. No feedback path connects them.

## Admission Contract

The DGD validating webhook establishes the static inputs before the Planner starts. For every
power-annotated component, it:

- requires the annotation value to be a positive decimal integer;
- rejects GPU allocation through Dynamic Resource Allocation (DRA), including GPU Memory Service and
  consumed resource claims;
- rejects adding, removing, or changing the power annotation after creation;
- rejects changes to the effective scalar `nvidia.com/gpu` count, preferring the `main` container
  limit over its request; and
- rejects changes to `multinode.nodeCount`.

These rules apply to the power tuple that the Planner caches:

```text
T_r = (C_r, G_r, M_r)
```

Here, `C_r` is the requested per-GPU cap, `G_r` is the scalar GPUs per Pod, and `M_r` is the
multinode node count for role `r`. Delete and recreate the DGD to change any member of this tuple.

The webhook protects the DGD inputs, while startup settlement protects the transition from DGD intent
to applied Pod annotations. Both are required for a stable cached projection.

This invariant requires the validating webhook to be enabled, reachable, and configured with
`webhook.failurePolicy: Fail`, which is the platform chart default. `Ignore` is an emergency-only
bypass: if the webhook is unavailable, Kubernetes can accept DGD writes without validation. Pause DGD
writes during that window and audit every accepted change afterward. If a cached power input might
have changed, recreate the DGD with the intended tuple and restart the Planner against the settled
replacement.

## Planner Access and RBAC

The Planner needs list-only Pod access when power awareness is installed. To add `pods/list` to either
the namespace-restricted Role or the cluster-wide ClusterRole, set the following value on the platform
chart's `dynamo-operator` subchart:

```yaml
dynamo-operator:
  planner:
    powerAwareness:
      enabled: true
```

The value defaults to `false`, so power-disabled installations do not gain the permission.

The Planner uses this permission for two read paths:

- **Startup settlement:** verify that relevant non-terminal Pods carry the exact annotation string
  from the settled DGD snapshot.
- **Runtime stability:** detect terminating Pods while classifying a worker rollout.

The Planner has no Pod mutation permission for this feature. It writes only replica targets through
the existing DGD scaling interface. See [Planner Examples](planner-examples.md#power-aware-budget-scaling)
for the Helm and Planner configuration.

## Configuration Contract

Enable the feature with these Planner fields:

```json
{
  "environment": "kubernetes",
  "mode": "disagg",
  "enable_power_awareness": true,
  "total_gpu_power_limit": 5200
}
```

| Field | Default | Validation | Meaning |
| --- | ---: | --- | --- |
| `enable_power_awareness` | `false` | Boolean | Enables startup settlement, projection metrics, rollout holds, and the final power clamp |
| `total_gpu_power_limit` | unset | Integer greater than or equal to 1; required when enabled | Projected GPU power ceiling for this DGD |
| `environment` | varies | Must be `kubernetes` when enabled | Selects the connector that resolves DGD-owned caps and Pod state |
| `mode` | `disagg` | Must be `disagg`, `prefill`, or `decode` when enabled | Selects the power-relevant worker roles; `agg` is unsupported |

Per-GPU caps are not Planner configuration fields. Every required worker role must carry the
annotation. The total budget is process-static and a changed value takes effect only after a Planner
restart.

## Startup Resolution

Initialization establishes one settled snapshot before permanently caching power facts:

1. Validate that the DGD contains every required worker role.
2. Wait for the DGD observed generation and worker rollout state to settle.
3. List DGD-scoped Pods and compare the raw annotation string on every relevant non-terminal Pod with
   the DGD template value.
4. Resolve the annotation as a positive integer.
5. Resolve scalar GPUs from the `main` container limit, falling back to its request.
6. Multiply GPUs per Pod by `multinode.nodeCount`, which defaults to `1`.
7. Cache the per-GPU cap and watts per logical replica in deployment state and worker capabilities.
8. Verify that `min_endpoint` replicas of every required role fit `total_gpu_power_limit`.

Terminating Pods still consume power, so they block startup settlement even when their annotation
matches. Pods in terminal `Succeeded` or `Failed` phases are ignored. A component with zero desired
replicas and no Pods is settled because future Pods will inherit the current template.

For role `r`:

```text
G_r = GPUs per Pod_r x nodeCount_r
W_r = requested cap per GPU_r x G_r
```

For a disaggregated target:

```text
W_projected = N_p x W_p + N_d x W_d
```

Prefill-only and decode-only modes apply the same calculation to their one required role.

Initial deployment validation reports missing or ambiguous required roles with
`DeploymentValidationError`. A connector without the power-aware interface, an invalid cached GPU or
topology input, or an infeasible minimum footprint also raises `DeploymentValidationError`. During
settlement, a missing or invalid annotation raises `PowerAnnotationMissingError` or
`PowerAnnotationInvalidError`, a failed operator rollout raises `RolloutFailedError`, and convergence
that exceeds the 30-minute polling limit raises `TimeoutError`. DRA-backed power components are
normally rejected earlier by DGD admission.

## Static Snapshot Lifecycle

`refresh()` does not reread the power annotation or recalculate watts per replica. The admission
contract keeps the DGD tuple immutable for the lifetime of the DGD, and the Planner uses its cached
tuple for the lifetime of the process.

To change a per-GPU cap, scalar GPU count, or `nodeCount`:

1. Delete the existing DGD.
2. Create a replacement DGD with the new tuple.
3. Start the Planner against the settled replacement.

To change only `total_gpu_power_limit`, update the Planner configuration and restart the Planner.

## Runtime Rollout Safety

`KubernetesConnector.get_power_aware_worker_counts()` issues one DGD GET and one DGD-scoped Pod LIST
together off the event loop through a single `asyncio.to_thread` dispatch. The Pod snapshot is
partitioned by component and used only to detect terminating Pods.

The surrounding `PlannerEnvironmentImpl.refresh()` also calls synchronous connector methods to refresh
GPU counts and the model name. On the Kubernetes path, each call issues an additional DGD GET on the
event-loop thread. These runtime reads do not validate annotations or recalculate the cached power caps.

The rollout hold applies only after the DGD GET and Pod LIST succeed. An exception from either read
propagates through `refresh()` and out of the Planner run loop; the Planner does not convert the error
into an unstable snapshot or retry it inside the loop.

The connector combines Pod termination state with DGD ready counts, component stability, and the
rolling-update phase. A terminating Pod, an unstable component, or a blocking or failed rolling-update
phase marks the deployment unstable. The environment then sets the settled `expected` count of both
power-relevant roles to `None`.

If either role is unstable, the final power boundary:

- suppresses every proposed scale-up;
- continues to allow scale-down;
- logs one warning for a continuous unstable interval; and
- leaves already-issued desired counts untouched.

This deployment-wide hold is conservative. It cannot distinguish the settled target of one rolling
role from another stable role, so it blocks both roles from scaling up until the deployment settles.

## Final Budget Boundary

The plugin pipeline merges omitted roles from the ready-count baseline. Before applying budgets,
`PipelineOutcome.proposed_components` restores the explicit PROPOSE-stage mask. A role omitted by
PROPOSE is charged at its ready count for arithmetic but cannot become an adjustable or emitted target.

```text
merged proposal
  -> restore the explicit PROPOSE-stage component mask
  -> GPU budget clamp
  -> rollout scale-up hold
  -> power budget clamp
  -> suppress targets equal to known settled expected counts
  -> replica target
```

The GPU clamp runs first and the power clamp runs second. The operations are not commutative. The
power ceiling can undo an increase introduced by the GPU floor, so the ceiling wins when the two
constraints conflict.

Only a target equal to a known settled `expected` count is suppressed as a no-op. During a rollout,
`expected` is unknown. An explicit target equal to the transient ready count can intentionally cancel
an in-flight desired count and must remain observable.

### Power Clamp Cases

| Input state | Result |
| --- | --- |
| Effective proposal fits | Preserve the proposal |
| Both roles are proposed and exceed the ceiling | Apply the pair clamp within the ceiling, respect `min_endpoint`, and never exceed either post-GPU-clamp count |
| One role is proposed and exceeds the residual budget | Charge the peer at its ready count and shrink only the proposed role |
| The fixed peer leaves less than one role's minimum footprint | Suppress that role's scale-up instead of mutating the peer |
| One role grows while the other shrinks and the parallel peak exceeds the budget | Emit the scale-down leg first and defer the scale-up |
| Proportional shrinking creates an opposing rebalance with an excessive parallel peak | Apply the same scale-down-first staging |
| No role is adjustable while the baseline is over budget | Emit no power-driven remediation |

The parallel projection is:

```text
W_peak =
    max(N_p,ready, N_p,target) x W_p
  + max(N_d,ready, N_d,target) x W_d
```

This peak protects opposing prefill/decode rebalances from assuming that a scale-down completes before
a simultaneous scale-up. It is not a complete model of Kubernetes surge, pending Pods, or terminating
Pods; the deployment-wide rollout hold covers those states conservatively.

## Guarantees and Failure Boundaries

| Statement | Status | Reason |
| --- | --- | --- |
| An admitted scale-up from an in-budget ready baseline fits the static requested-cap model | Guaranteed within the cached inputs | The final power clamp runs after proposal merging and the GPU clamp |
| A partial proposal does not mutate its unproposed peer | Guaranteed | Explicit proposal provenance keeps the peer fixed while charging its ready count |
| An opposing rebalance does not intentionally exceed the modeled parallel peak | Guaranteed within the ready/target model | Scale-up legs are deferred |
| Planner power awareness does not patch Pods | Guaranteed | The connector, RBAC, and integration test expose list-only Pod access |
| Current replicas are automatically reduced when already over budget | Not guaranteed | The clamp changes only adjustable proposals and has no emergency reconciliation policy |
| A runtime DGD or Pod read failure becomes a rollout hold | Not guaranteed | The exception propagates out of the Planner run loop instead of producing an unstable snapshot |
| GPU hardware enforces the requested DGD value exactly | Not guaranteed | The Power Agent can clamp to hardware limits, apply its safe default for malformed or conflicting annotations, or fail an actuator write |
| Actual draw stays below `total_gpu_power_limit` | Not guaranteed | The Planner observes neither effective caps nor measured draw |
| A shared rack or cluster stays below a facility budget | Not guaranteed | The budget applies to one DGD and has no global allocator |
| A cap update becomes visible on the existing DGD | Not supported | Admission makes the power tuple immutable |

The distinction between requested and effective caps is safety-critical. If a requested value is below
a GPU's supported minimum, the Power Agent can apply a higher effective value. Malformed or conflicting
annotations on a GPU can select the configured safe-default cap, and an actuator failure can leave
hardware outside the Planner's model. Treat `total_gpu_power_limit` as an admission policy, not a
facility protection mechanism.

## Observability

When Prometheus and power awareness are enabled and the required per-replica values are resolved, the
Planner publishes:

| Metric | Meaning |
| --- | --- |
| `dynamo_planner_power_budget_total_watts` | Configured DGD budget |
| `dynamo_planner_power_projected_watts` | Ready replica counts multiplied by cached watts per replica |
| `dynamo_planner_power_budget_utilization` | Projected watts divided by the budget |

These metrics contain requested-cap projections. They do not report measured power, effective caps,
actuator health, the proposed target, the modeled parallel peak, or the clamp reason. Clamp and
rollout-hold details are logged.

## Test Evidence

The test suite divides the contract into explicit layers:

| Layer | Evidence |
| --- | --- |
| DGD admission | Positive cap values, DRA rejection, and annotation/GPU/node-count immutability |
| DGD parsing | Disaggregated, asymmetric, multinode, renamed, missing, duplicate, and invalid-component cases |
| Startup settlement | Generation convergence, exact annotation strings, terminating and terminal Pods, zero replicas, and missing Pods |
| Configuration | Default-off behavior, required budget, Kubernetes-only environment, supported modes, and positive budget |
| Runtime state | One off-thread Pod snapshot, terminating-Pod detection, blocking/failed rollout phases, and power-off compatibility |
| Pure budget logic | Fit, proportional shrink, partial proposals, fixed-peer residuals, no-upscale invariant, baseline over-budget behavior, and rebalance peaks |
| Adapter boundary | Explicit proposal provenance, GPU-then-power ordering, plugin proposal coverage, rollout holds, and stable no-op suppression |
| Metrics | Projection, asymmetric and multinode values, feature gates, invalid-budget guards, and unresolved-cap suppression |
| Ownership | Conditional `pods/list`, annotation-key equality, and a resolve/project/clamp integration test with no Pod patch surface |

These tests establish deterministic controller behavior. They do not constitute a live GPU
enforcement test or validate a facility-level power ceiling.

## Deferred Dynamic Control and Follow-Up Priorities

### Close the Enforcement Feedback Gap

Expose the effective applied cap, cap generation, actuator success, and data freshness to the
admission controller. Use effective caps for worst-case admission. Treat measured draw as an
optimization signal with explicit headroom, not as proof that a later workload spike will fit.

### Make Cap Updates Transactional

Define a versioned handoff before allowing cap retargeting:

- the DGD declares cap intent and a generation;
- workers report the generation and effective cap they enforce;
- the Planner admits scale-up only when every relevant replica reports the expected generation; and
- incomplete or stale generations fail closed.

This removes the static delete-and-recreate lifecycle without restoring Planner-owned Pod patching.

### Define Over-Budget Reconciliation

Specify behavior for budget decreases, effective-cap increases, failed cap application, and external
replica changes. The policy must define role priority, `min_endpoint` behavior, service-level objective
interactions, emergency cooldown rules, and recovery without oscillation.

### Replace the Rollout Approximation

Expose per-role desired, ready, pending, unavailable, and terminating counts. Include the operator's
surge behavior in the peak model. This would permit safe scale-up of an unrelated stable role and
account for transient Pods directly.

### Introduce Budget Scope and Allocation

Define a hierarchy for shared facility, rack, cluster, namespace, and DGD budgets, including
reservation, borrowing, and revocation. Global Planner integration should follow that allocation
model instead of forwarding independent local budgets.

### Reconcile Multinode Resource Units

Power projection multiplies GPUs per Pod by `nodeCount`, while the existing GPU-budget path retains
its per-Pod GPU count. Name and type both units before extending the feature so one multinode replica
cannot satisfy the two budgets under different interpretations.

### Add Decision-Level Telemetry

Add bounded counters for clamp and hold reasons, gauges for target and peak watts, cached cap
generation and age, and effective-cap health. Use this evidence before enabling automated remediation
or adaptive control.

### Optimize Power Agent API Load Separately

Use a watch or informer-backed Pod cache to reduce the Power Agent's Kubernetes API load. This
optimization does not close the Planner's effective-cap or reconciliation gaps and can proceed
independently.

## Code Reference Map

| Contract | Implementation | Tests |
| --- | --- | --- |
| Configuration and cross-field validation | [`PlannerConfig`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/config/planner_config.py) | [`test_planner_config.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_planner_config.py) |
| DGD admission and static tuple | [`dynamographdeployment.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/internal/webhook/validation/dynamographdeployment.go) and [`dynamographdeployment_helpers.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/internal/webhook/validation/dynamographdeployment_helpers.go) | [`dynamographdeployment_validation_envtest_test.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/internal/webhook/validation/dynamographdeployment_validation_envtest_test.go) |
| Annotation, GPU topology, and role resolution | [`dgd_services.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/monitoring/dgd_services.py) | [`test_dgd_power_annotation.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_dgd_power_annotation.py) |
| Startup settlement and Pod convergence | [`kubernetes_api.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/connectors/clients/kubernetes_api.py) | [`test_kube.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_kube.py) |
| Startup cache and minimum footprint | [`environment/base.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/environment/base.py) | [`test_power_environment.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_power_environment.py) |
| Runtime rollout state | [`connectors/kubernetes.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/connectors/kubernetes.py) | [`test_kubernetes_connector.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_kubernetes_connector.py) |
| Projection and clamp logic | [`budget.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/core/budget.py) | [`test_power_budget.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_power_budget.py) |
| Proposal provenance and final boundary | [`pipeline.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/plugins/orchestrator/pipeline.py) and [`engine_adapter.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/plugins/orchestrator/engine_adapter.py) | [`test_pipeline.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/plugins/orchestrator/test_pipeline.py) and [`test_power_budget.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_power_budget.py) |
| Projection metrics | [`core/base.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/core/base.py) and [`planner_metrics.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/monitoring/planner_metrics.py) | [`test_metric_publication.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/unit/test_metric_publication.py) |
| Power Agent enforcement boundary | [`power_agent.py`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/power-agent/power_agent.py) and [`actuator.py`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/power-agent/actuator.py) | [`test_apply_cap.py`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/power-agent/tests/test_apply_cap.py), [`test_multi_pod_policy.py`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/power-agent/tests/test_multi_pod_policy.py), and [`test_dcgm_actuator.py`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/power-agent/tests/test_dcgm_actuator.py) |
| Conditional Pod RBAC | [`planner.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/components/operator/templates/planner.yaml) | [`planner_rbac_test.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/tests/planner_rbac_test.yaml) |
| No Pod mutation | [`KubernetesConnector`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/connectors/kubernetes.py) | [`test_power_no_mutation.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/integration/test_power_no_mutation.py) |

## Related Documentation

- [Planner Design](planner-design.md) describes the complete Planner pipeline and scaling algorithms.
- [Planner Examples](planner-examples.md#power-aware-budget-scaling) shows the configuration and Helm
  enablement.
- [Power-aware budget example](https://github.com/ai-dynamo/dynamo/tree/main/examples/power-aware-budget)
  contains the DGD annotation and Planner configuration contract.
