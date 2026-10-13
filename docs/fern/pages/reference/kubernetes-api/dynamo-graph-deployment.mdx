---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DynamoGraphDeployment (DGD) Reference
subtitle: Field reference for the DynamoGraphDeployment custom resource — the canonical, live description of a Dynamo inference graph.
---

A `DynamoGraphDeployment` (DGD) is the canonical description of a running Dynamo inference graph. You list the components that make up the graph — frontend, workers, prefill and decode workers, planner, EPP — and the [operator](../../kubernetes/installation/install-dynamo.md) reconciles them into [`DynamoComponentDeployment`](dynamo-component-deployment.mdx) resources, pods, and Services.

A DGD is the resource that persists and serves traffic. Author one directly when you have a known-good configuration or a tuned [recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes); generate one from intent with a [`DynamoGraphDeploymentRequest`](dynamo-graph-deployment-request.mdx) (DGDR).

This page documents the `nvidia.com/v1beta1` API — the served, current version.

LPX components are experimental and require the operator's `lpx.enabled` setting. When disabled,
admission rejects new or changed LPX components but permits otherwise valid updates that leave them
unchanged, and deletion. Existing LPX DGDs report a disabled status without further workload
reconciliation; installed CRDs and workloads are preserved.

<Info>
  Each entry in `spec.components` is a `DynamoComponentDeploymentSharedSpec` — the same fields documented on the [DCD Reference](dynamo-component-deployment.mdx#shared-component-spec). This page covers the **graph-level** fields; per-component fields live there.
</Info>

## Minimal example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-graph
spec:
  backendFramework: vllm
  components:
    - name: frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.4.0
    - name: worker
      type: worker
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0
```

## Grove update strategy

Set `metadata.annotations["nvidia.com/grove-update-strategy"]` on the DGD to select the generated Grove `PodCliqueSet` update strategy. Accepted values are `Coherent`, `RollingRecreate`, and `OnDelete`. This annotation belongs to the DGD's metadata; `spec.annotations` propagates annotations to child resources instead.

Grove uses `RollingRecreate` unless this annotation explicitly selects another strategy. This applies to aggregated, disaggregated, and LPX deployments, regardless of operator origin version. Neither the deprecated component `minAvailable` nor a native minimum in `providerOverride` enables Coherent.

To opt into Coherent, add:

```yaml
metadata:
  annotations:
    nvidia.com/grove-update-strategy: Coherent
```

Starting with Dynamo 1.6.0, new Grove-backed DGDs that omit all deprecated component minima default provider-native `minAvailable` to `1`. Existing DGDs retain their persisted legacy fields. This availability defaulting is independent of the rollout strategy: both forms use RollingRecreate without an annotation. Mixing the old and native minimum forms anywhere in the DGD is rejected; migrate all components in one update. An explicit strategy annotation applies to both PodCliqueSets in a hybrid graph.

Grove v0.1.0-alpha.14 or later and its matching CRDs are required for Coherent updates. Upgrade externally managed Grove controllers and CRDs together. The operator writes the requested strategy and reports API validation or admission rejections through its normal failure condition and reconciliation retry path; it does not substitute a different strategy.

Coherent recovery can stall when already-unavailable replicas exhaust the disruption budget, as tracked in [Grove issue #873](https://github.com/ai-dynamo/grove/issues/873). Review this limitation before opting in. Changing or removing the strategy annotation cannot unblock a stalled active update. See [Recover from a stalled Coherent update](../../kubernetes/model-deployment/deploy-with-dgd.md#recover-from-a-stalled-coherent-update) for the DGD recreation procedure and the limitations of pod deletion workarounds.

Coherent coordinates compatible replacement capacity within one PodCliqueSet (PCS). Grove selects components whose rendered pod templates change and gang-schedules a minimum viable unit of their replacements. Separate PCSes, including the ordinary and LPX PCSes in a hybrid graph, roll independently.

Dynamo includes a shared worker-generation hash in rendered worker templates. A pod-template edit to one worker can therefore roll every worker component, including workers whose authored pod template did not change. Non-worker components such as the frontend participate when their own rendered template changes. Coherent does not enforce traffic isolation between versions; Dynamo's discovery and routing must keep requests within compatible worker generations.

### Coherent capacity and disruption

| Setting | Meaning under Coherent |
|---|---|
| `minAvailable` | Size of this component's minimum viable unit (MVU): the number of pods or complete scaling-group replicas replaced together in the coordinated unit. It also retains its gang scheduling and availability role. |
| `rollingUpdate.maxUnavailable` | Per-component ceiling on unavailable pods or scaling-group replicas during the rollout. It must be at least `minAvailable` and defaults to that value. Existing unavailable capacity consumes the budget. |

Grove alpha.14 has no surge support: it removes old capacity before replacements become available. A larger `minAvailable` therefore requires a larger disruption budget; it is not a promise that this many replicas will remain serving throughout the update. Increasing `maxUnavailable` permits more unavailable capacity without reducing the MVU size. The budget gates further rollout deletions; unrelated failures can independently reduce availability.

For a component with eight replicas:

| `minAvailable` | `maxUnavailable` | Capacity implication |
|---|---|---|
| `1` | `1` (default) | The minimum replacement unit contains one replica, with a one-replica disruption budget. |
| `4` | `4` (default) | A coordinated unit can remove four replicas before replacements become available, temporarily losing half the component's capacity. |
| `4` | `6` | The minimum unit still contains four replicas; the larger budget permits up to six unavailable replicas. |

Grove can roll replicas beyond the minimum unit through additional tail steps. Progress depends on scheduling and remaining availability budget; it does not always wait for every replica in the previous batch to become Ready. Coherent does not guarantee an exact P/D ratio at every instant or zero downtime. For the detailed step plan, see [Grove's coherent update design](https://github.com/ai-dynamo/grove/blob/v0.1.0-alpha.14/docs/proposals/393-coherent-rolling-updates/README.md).

<Warning>
  If `minAvailable == replicas`, the minimum unit can take the entire component offline. This also applies to a single-replica worker or frontend whose template changes. Unchanged frontends are not co-rolled solely because workers change. Replica counts alone do not establish whether the remaining workers can meet your latency and throughput service-level objectives (SLOs). Admission warns on creation and relevant rollout edits when an explicitly selected Coherent minimum could take the entire component offline. Replica-only edits warn when they introduce this risk; unrelated metadata updates and replica-only edits that leave existing risk unchanged do not repeat the warning. Pod-template edits still warn before a rollout that could take the component offline.
</Warning>

For new DGDs, use the provider-native `minAvailable: 1` default unless the application requires a larger viable unit. Keep the disruption budget at its default until the remaining capacity has been validated under representative traffic. Size spare serving capacity before a rollout; adding a larger budget does not add surge capacity. `minAvailable` is immutable after DGD creation, so review it before creating a deployment or opting an existing deployment into Coherent.

Grove rejects replica changes throughout an active Coherent rollout. Before changing replicas, the Dynamo operator synchronizes the desired PodCliqueSet configuration and, after a write, waits for its cache to observe that configuration. For Coherent, it also waits until Grove acknowledges the current configuration (`status.observedGeneration` matches `metadata.generation`) and any active rollout completes. Missing progress or completed progress from an older generation does not acknowledge a new configuration. Initial configuration acknowledgement does not require pods to be Ready or LPX scheduler requests to exist. During this wait the operator continues observing workload readiness and component status and reports `ScalingDeferred` when a replica change is pending. PodCliqueSet watch events resume scaling. RollingRecreate rollouts do not block replica changes once the configuration is observed, even if Grove's observed generation lags. Unexpected API or admission failures use the normal reconciliation retry path and report `Failed` with `Ready=False` while the failure persists. A rejected scale write caused by lag between the Grove and Dynamo caches can therefore briefly report a failure until a retry succeeds. External scalers writing directly to Grove's scale subresources still receive the admission rejection; do not rely on autoscaling to supply capacity during an active Coherent rollout.

For LPX workloads with explicit replica counts, scheduling deadline cleanup also waits until capacity can be reduced before deleting expired pipeline requests. Workloads with omitted replica counts retain external capacity ownership, and their expired requests can still be cleaned up during a rollout.

For RollingRecreate, `maxUnavailable` defaults to `1`; rolling update configuration is rejected with OnDelete. See [Grove's defaulting implementation](https://github.com/ai-dynamo/grove/blob/v0.1.0-alpha.14/operator/internal/webhook/admission/pcs/defaulting/podcliqueset.go).

DGD does not currently expose `maxUnavailable`. The Dynamo operator leaves it unset in generated PCS templates, so Grove supplies the defaults above. Upgrading Grove alone does not add this field to the DGD API.

<Note>
  An operator upgrade does not opt any DGD into Coherent. To opt an existing DGD into Coherent, set `metadata.annotations["nvidia.com/grove-update-strategy"]: Coherent`. A strategy-only change leaves pod templates and worker hashes unchanged and does not itself trigger a workload rollout. All strategy transitions, including annotation changes or removal, wait until an active Grove update finishes (`status.updateProgress.updateEndedAt` is set). For LPX deployments, editing the annotation also changes the LPX input revision and causes child reconciliation. See the [platform upgrade notes](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/README.md#v160).
</Note>

<Note>
  Adding, removing, or migrating a minimum availability field does not select Coherent. Removing the Coherent annotation returns to RollingRecreate after any active Grove update finishes.
</Note>

### Configure native minimum availability

Set an explicit minimum at component scope. Dynamo resolves the target when omitted, so generated PCS names are not required. An omitted minimum resolves to `1` during rendering; admission leaves an absent `providerOverride` absent and does not add availability fields to a topology-only override.

| Component shape | Provider target | Native minimum path inside `value` |
|---|---|---|
| Standalone single-node component | `PodCliqueTemplateSpec` | `spec.minAvailable` |
| Multinode, inter-pod GPU memory service, or forced scaling group | `PodCliqueScalingGroupConfig` | `minAvailable` |
| LPX component with a conductor role | `PodCliqueScalingGroupConfig` | `minAvailable` |

A standalone prefill component can specify:

```yaml
providerOverride:
  apiVersion: grove.io/v1alpha1
  value:
    spec:
      minAvailable: 1
```

For a scaling-group component, use `value.minAvailable: 1` instead. A shared LPX draft has no independent availability override; configure its target component. Root and multinode role overrides remain topology-only. LPX topology overrides remain unsupported.

The effective minimum must be a positive integer and fit a positive replica count; ordinary Grove components may scale to zero without clearing their minimum. With omitted LPX replicas, the minimum seeds the initial scaling-group capacity.

Admission returns a deprecation warning when a component introduces or changes the old `minAvailable` field, independently of the rollout strategy. Updates that leave the legacy value unchanged, including replica-only and metadata edits, do not repeat the warning. To migrate an existing DGD, remove every deprecated component `minAvailable` and put the same values in the corresponding component provider overrides in one update, including defaulted values on frontends and other components. Admission enforces effective-value immutability across both forms. Migration leaves the rollout strategy unchanged; select Coherent separately with the annotation. Moving the same values leaves worker hashes and rendered pod templates unchanged; subsequent workload updates use the selected strategy. Review the disruption implications before opting in.

A shared LPX draft has a legacy minimum of `1` but no independent scaling group. Remove that legacy field without adding a draft override; its effective default remains `1`, and the target component owns the native workload minimum.

Both fields retain their native hub/spoke conversion mapping. Converting between `v1alpha1` and `v1beta1` preserves the chosen form and does not migrate a deployment or select a strategy.

## Spec reference

<ParamField path="components" type="[]DynamoComponentDeploymentSharedSpec">
  Components deployed as part of this graph. Each entry carries its own stable logical `name` (unique within the list, case-insensitively). Component types are repeatable except `type: epp`, which may appear at most once. Maximum 25 components. For the full per-component field set, see [DCD Reference — Shared component spec](dynamo-component-deployment.mdx#shared-component-spec).
</ParamField>

<Note>
  Each new non-LPX component must supply `spec.components[*].podTemplate.spec.containers[name=main]` with a non-empty `image`. The runtime image is `spec.components[*].podTemplate.spec.initContainers[name=runtime].image` in Dynamo sidecar mode, or `spec.components[*].podTemplate.spec.containers[name=main].image` otherwise. If that image has no parseable semantic-version tag, set `spec.components[*].runtimeVersionOverride` to its Dynamo runtime version. Also set it when the tag identifies a different version, such as the inference engine's version. See [Runtime Version Compatibility](dynamo-component-deployment.mdx#runtime-version-compatibility). DGD has no graph-level `runtimeVersionOverride`; the DGDR-level field supplies a default when generating components.

  LPX components use `roles[].podTemplate` instead and do not require `runtimeVersionOverride`. See [Shared component spec](dynamo-component-deployment.mdx#shared-component-spec).
</Note>

<ParamField path="providerOverride" type="ProviderOverride">
  Customizes the root resource generated by the selected workload provider. Currently supported only for Grove, with `apiVersion: grove.io/v1alpha1` and target `PodCliqueSet`. The `value` may set only `spec.template.topologyConstraint`. This field cannot select or change the provider. See [ProviderOverride](#provideroverride).
</ParamField>

<ParamField path="backendFramework" type="string">
  GPU backend framework for worker components. When omitted, the operator infers the framework from each worker's command and arguments. When set, it must match the detected worker framework. LPX components omit this field. Backend-specific defaults apply to `spec.components[*].podTemplate.spec.containers[name=main]`.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>sglang</Badge> <Badge intent="note" minimal>vllm</Badge> <Badge intent="note" minimal>trtllm</Badge></span>
</ParamField>

<ParamField path="env" type="[]core/v1.EnvVar">
  Environment variables prepended to each component's `main` container environment and, in Dynamo sidecar mode, its `runtime` init container environment. A container-specific `env` entry with the same name takes precedence and may reference values from this list.

  <a href="https://pkg.go.dev/k8s.io/api/core/v1#EnvVar" target="_blank">core/v1.EnvVar</a>
</ParamField>

<ParamField path="annotations" type="map[string]string">
  Annotations propagated to all child resources (scaling adapters, DCDs, Deployments, and pod templates). Values on the selected component or role `podTemplate` take precedence on conflict.
</ParamField>

<ParamField path="labels" type="map[string]string">
  Labels propagated to all child resources. Same precedence rules as `annotations`.

  For LPX, metadata synchronization keeps the PCS labels and clique templates consistent after queue edits. Moving an existing KAI PodGroup to another queue depends on the Grove backend; metadata synchronization does not recreate existing PodGroups.
</ParamField>

<ParamField path="priorityClassName" type="string">
  Name of the `PriorityClass` to use for Grove `PodCliqueSet`s. Requires the Grove pathway. See [Multinode Orchestration](../../kubernetes/installation/multinode-orchestration.md).
</ParamField>

<ParamField path="restart" type="Restart">
  Restart policy for the graph. Must be unset on creation; set or change `restart.id` on an existing DGD to trigger a restart.
  Graphs containing LPX components require explicit `restart.strategy.type: Parallel`; Sequential and omitted strategies are rejected.
</ParamField>

<Indent>
  <ParamField path="id" type="string" required={true}>
    Arbitrary string; any change to it initiates a restart of the graph according to the configured strategy.
  </ParamField>
  <ParamField path="strategy" type="RestartStrategy">
    How components are restarted.
  </ParamField>
  <Indent>
    <ParamField path="type" type="RestartStrategyType" default="Sequential">
      Whether components restart one at a time or all at once.

      <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Sequential</Badge> <Badge intent="note" minimal>Parallel</Badge></span>
    </ParamField>
    <ParamField path="order" type="[]string">
      Complete ordered set of component names for a sequential restart. Omit to use the controller's default order. Must not be set for parallel restarts.
    </ParamField>
  </Indent>
</Indent>

<ParamField path="topologyConstraint" type="SpecTopologyConstraint">
  Deployment-level topology constraint. Components without their own `topologyConstraint` inherit this value. See the [SpecTopologyConstraint API](full-api-reference.mdx#spectopologyconstraint).
</ParamField>

<Indent>
  <ParamField path="clusterTopologyName" type="string" required={true}>
    Name of the `ClusterTopology` resource defining the topology hierarchy for this deployment.
  </ParamField>
  <ParamField path="packDomain" type="string">
    Default topology domain to pack pods within. Optional at this level; omit when only components carry constraints.
  </ParamField>
</Indent>

<ParamField path="experimental" type="DynamoGraphDeploymentExperimentalSpec">
  Graph-level opt-in preview features whose API shape may change in breaking ways between `v1beta1` releases. Component-level experimental features live under `spec.components[*].experimental` instead — see [DCD Reference — ExperimentalSpec](dynamo-component-deployment.mdx#experimentalspec).
</ParamField>

<Indent>
  <ParamField path="kvTransferPolicy" type="KvTransferPolicy">
    Topology-aware routing for KV-cache transfers between prefill and decode workers. Set exactly one of `labelKey` or `clusterTopologyName`. See the [KvTransferPolicy API](full-api-reference.mdx#kvtransferpolicy).
  </ParamField>
  <Indent>
    <ParamField path="clusterTopologyName" type="string">
      References a Grove `ClusterTopology` CR. The operator reads the CR's topology levels and projects them through Dynamo-owned pod labels for worker topology metadata. Mutually exclusive with `labelKey`.
    </ParamField>
    <ParamField path="labelKey" type="string">
      A Kubernetes node label key (for example `topology.kubernetes.io/zone`) whose value identifies the topology domain for each worker. The operator copies the node label onto worker pods so the runtime can publish it as worker metadata. Should correspond to the topology level named in `domain`. Mutually exclusive with `clusterTopologyName`.
    </ParamField>
    <ParamField path="domain" type="TopologyDomain">
      Logical name for the topology level to enforce (for example `zone`, `rack`). The router uses this to match workers that share the same value for the label identified by `labelKey`. Free-form string matching `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`.
    </ParamField>
    <ParamField path="enforcement" type="KvTransferEnforcement" default="required">
      How the selected prefill worker's topology is applied to decode routing. `required` allows only decode workers in the same topology domain as the selected prefill worker; `preferred` keeps all decode workers eligible but biases selection toward the same domain.

      <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>required</Badge> <Badge intent="note" minimal>preferred</Badge></span>
    </ParamField>
    <ParamField path="preferredWeight" type="float">
      Required and used only when `enforcement` is `preferred`. Higher values create a stronger same-domain routing preference but do not guarantee same-domain selection; the value is not a probability. `0` disables the topology preference; `1` is the strongest supported preference. Minimum `0`, maximum `1`.
    </ParamField>
  </Indent>
</Indent>

## Status

The operator maintains observed state under `status`.

<ParamField path="status.state" type="string" default="initializing">
  High-level textual status of the graph deployment lifecycle.
</ParamField>

<ParamField path="status.conditions" type="[]metav1.Condition">
  Latest observed conditions, merged by type. Includes `Available` and `DynamoComponentReady`.

  <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Condition" target="_blank">metav1.Condition</a>
</ParamField>

<ParamField path="status.components" type="map[string]ComponentReplicaStatus">
  Per-component replica status, keyed by component name.

  See: [ComponentReplicaStatus](#componentreplicastatus)
</ParamField>

<ParamField path="status.observedGeneration" type="integer">
  Most recent `metadata.generation` observed by the controller.
</ParamField>

<ParamField path="status.restart" type="RestartStatus">
  Status of a graph-level restart in progress.
</ParamField>

<Indent>
  <ParamField path="observedID" type="string">
    The restart ID currently being processed. Matches `spec.restart.id`.
  </ParamField>
  <ParamField path="phase" type="RestartPhase">
    Phase of the restart.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Pending</Badge> <Badge intent="note" minimal>Restarting</Badge> <Badge intent="note" minimal>Completed</Badge> <Badge intent="note" minimal>Failed</Badge> <Badge intent="note" minimal>Superseded</Badge></span>
  </ParamField>
  <ParamField path="inProgress" type="[]string">
    Names of the components currently being restarted.
  </ParamField>
</Indent>

<ParamField path="status.checkpoints" type="map[string]ComponentCheckpointStatus">
  Per-component checkpoint status, keyed by component name. See the [ComponentCheckpointStatus API](full-api-reference.mdx#componentcheckpointstatus).

  See: [ComponentCheckpointStatus](#componentcheckpointstatus)
</ParamField>

<ParamField path="status.rollingUpdate" type="RollingUpdateStatus">
  Progress of operator-managed rolling updates. Currently supported only for single-node, non-Grove deployments. See the [RollingUpdateStatus API](full-api-reference.mdx#rollingupdatestatus).
</ParamField>

<Indent>
  <ParamField path="phase" type="RollingUpdatePhase">
    Current phase of the rolling update.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Pending</Badge> <Badge intent="note" minimal>InProgress</Badge> <Badge intent="note" minimal>Completed</Badge> <Badge intent="note" minimal>Failed</Badge></span>
  </ParamField>
  <ParamField path="startTime" type="metav1.Time">
    When the rolling update began.

    <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Time" target="_blank">metav1.Time</a>
  </ParamField>
  <ParamField path="endTime" type="metav1.Time">
    When the rolling update completed, successfully or failed.

    <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Time" target="_blank">metav1.Time</a>
  </ParamField>
  <ParamField path="updatedComponents" type="[]string">
    Components that have completed the rolling update.
  </ParamField>
</Indent>

<ParamField path="status.placement" type="PlacementStatus">
  Scheduler placement score and reporting state. The schema is available, but the controller does not yet populate this field.
</ParamField>

<Indent>
  <ParamField path="score" type="number">
    Worst placement score across relevant scheduler placement units. Minimum `0`, maximum `1`; higher is better. Scores are comparable only between DGDs using the same scheduler scoring contract and version.
  </ParamField>
  <ParamField path="state" type="PlacementScoreState">
    `Reported` means every scored placement unit has a score; `Partial` means only some do. `Unsupported` means the backend exposes no score. `Unknown` means scoring is supported but the current value is indeterminate, and `score` must be absent.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Reported</Badge> <Badge intent="note" minimal>Partial</Badge> <Badge intent="note" minimal>Unsupported</Badge> <Badge intent="note" minimal>Unknown</Badge></span>
  </ParamField>
</Indent>

## Additional Types

Nested struct types referenced by the fields above, broken out here to keep the field lists shallow. Types prefixed `core/v1.` or `metav1.` are standard Kubernetes types and link to their Go package documentation instead of being expanded.

### ProviderOverride

A sparse provider-native fragment, supported only for Grove-backed DGD resources. The same shape is used at graph, component, and role scope; the location determines the allowed target. Standalone DCDs do not support provider overrides.

<ParamField path="apiVersion" type="string" required={true}>
  Provider schema version. Must be `grove.io/v1alpha1`.
</ParamField>

<ParamField path="target" type="string">
  Provider resource or embedded schema. Admission resolves and persists this value when omitted. Graph scope uses `PodCliqueSet`; component scope uses `PodCliqueTemplateSpec` or `PodCliqueScalingGroupConfig` according to the component shape; multinode leader and worker roles use `PodCliqueTemplateSpec`. Other targets are rejected.
</ParamField>

<ParamField path="value" type="object" required={true}>
  Sparse fragment of the target schema. `PodCliqueSet` accepts only `spec.template.topologyConstraint`; embedded targets accept `topologyConstraint`. At component scope, `PodCliqueTemplateSpec` also accepts `spec.minAvailable`, and `PodCliqueScalingGroupConfig` accepts `minAvailable`. Root and role availability overrides and other fields are rejected. See [Configure native minimum availability](#configure-native-minimum-availability).
</ParamField>

### ComponentReplicaStatus

Replica information for a single component, keyed by component name in `status.components`.

<ParamField path="componentKind" type="ComponentKind">
  Underlying Kubernetes resource kind backing the component.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>PodClique</Badge> <Badge intent="note" minimal>PodCliqueScalingGroup</Badge> <Badge intent="note" minimal>Deployment</Badge> <Badge intent="note" minimal>LeaderWorkerSet</Badge></span>
</ParamField>

<ParamField path="componentNames" type="[]string">
  Underlying Kubernetes resource names for this Dynamo component. During normal operation this contains a single name; during rolling updates it contains both the old and new resource names.
</ParamField>

<ParamField path="runtimeNamespace" type="string">
  Effective Dynamo runtime namespace for the component. Workers may include a generation suffix; non-workers use the base namespace. During a rolling update, worker status retains the old active revision's namespace until cutover completes.
</ParamField>

<ParamField path="servedModelName" type="string">
  Effective primary model identity resolved from the fully rendered serving Pod. The operator checks `--served-model-name`, `--model-name`, `--model`, and `--model-path` in that order. Omitted when the rendered command declares none of those flags. During a rolling update, worker status retains the old active revision's value until cutover completes.
</ParamField>

<ParamField path="runtimeComponentName" type="string">
  Explicit Dynamo runtime component identity resolved from the serving Pod's `--endpoint` value. Omitted when the rendered command does not declare a component override. During a rolling update, worker status retains the old active revision's value until cutover completes.
</ParamField>

<ParamField path="gpuPowerLimitWatts" type="integer">
  Effective per-GPU power limit configured for the component. Omitted when no positive component-level power limit is configured. Minimum `1`.
</ParamField>

<ParamField path="gpusPerEngine" type="integer">
  GPUs assigned to one inference engine across all nodes of a component replica, excluding independent auxiliary GPU allocations. A reported `0` means the engine has no GPUs; consult `gpusPerReplica` for auxiliary allocations. Minimum `0`.
</ParamField>

<ParamField path="gpusPerReplica" type="integer">
  Unique GPU allocation added by scaling the component up by one replica, across all nodes, application and initialization phases, and provider-owned pods. Scalar GPUs use the effective Kubernetes pod scheduling footprint; shared DRA claims count once. A reported `0` means a resolved non-GPU allocation; omission means no current shape is available. Minimum `0`.
</ParamField>

<ParamField path="replicas" type="integer">
  Total number of non-terminated replicas. Minimum `0`.
</ParamField>

<ParamField path="updatedReplicas" type="integer">
  Number of replicas at the current/desired revision. Minimum `0`.
</ParamField>

<ParamField path="readyReplicas" type="integer">
  Number of ready replicas. Populated for `PodClique`, `Deployment`, and `LeaderWorkerSet`; not available for `PodCliqueScalingGroup`. Minimum `0`.
</ParamField>

<ParamField path="availableReplicas" type="integer">
  Number of available replicas. Populated for `Deployment` and `PodCliqueScalingGroup`; not available for `PodClique` or `LeaderWorkerSet`. Minimum `0`.
</ParamField>

<ParamField path="scheduledReplicas" type="integer">
  Replicas scheduled by the backend, in complete Dynamo component-replica units rather than raw pod counts. Helps distinguish scheduling shortfalls from runtime readiness. Omitted when the backend cannot report a reliable count; absence means not reported, not zero scheduled. Minimum `0`.
</ParamField>

### ComponentCheckpointStatus

Checkpoint information for a single component, keyed by component name in `status.checkpoints`.

<ParamField path="checkpointName" type="string">
  Name of the active `PodSnapshot`.
</ParamField>

<ParamField path="checkpointID" type="string" deprecated={true}>
  Deprecated legacy Dynamo artifact ID. Standalone snapshots leave this field empty.
</ParamField>

<ParamField path="identityHash" type="string" deprecated={true}>
  Deprecated legacy checkpoint identity hash. Standalone snapshots leave this field empty.
</ParamField>

<ParamField path="ready" type="boolean">
  Whether the checkpoint artifact is ready for future pods to restore.
</ParamField>

## Inspect a DGD

```bash
# List graph deployments (short name: dgd)
kubectl get dgd -n $NAMESPACE

# Detailed status, conditions, and events
kubectl describe dgd my-graph -n $NAMESPACE

# Per-component readiness
kubectl get dgd my-graph -n $NAMESPACE -o jsonpath='{.status.components}'
```

## Related pages

<CardGroup cols={2}>
  <Card title="DCD Reference" icon="cube" href="dynamo-component-deployment.mdx">
    Per-component fields embedded in `spec.components`.
  </Card>
  <Card title="DGDR Reference" icon="wand-magic-sparkles" href="dynamo-graph-deployment-request.mdx">
    Generate a DGD by intent instead of authoring it by hand.
  </Card>
  <Card title="Deploy with DGD" icon="rocket" href="../../kubernetes/model-deployment/deploy-with-dgd.md">
    Task-oriented walkthrough of authoring a graph deployment.
  </Card>
  <Card title="Install Dynamo" icon="gears" href="../../kubernetes/installation/install-dynamo.md">
    Install and configure the Dynamo platform and operator.
  </Card>
</CardGroup>
