---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DynamoComponentDeployment (DCD) Reference
subtitle: Field reference for the DynamoComponentDeployment custom resource — the single Dynamo component that a DynamoGraphDeployment reconciles into pods.
---

A `DynamoComponentDeployment` (DCD) describes one component of a Dynamo inference graph — a frontend, a worker, a prefill or decode worker, a planner, or an EPP. The [operator](../../kubernetes/installation/install-dynamo.md) reconciles each DCD into a Kubernetes Deployment (or Grove workload) plus its Services, ConfigMaps, and pods.

A DCD is rarely authored on its own. Each entry in a [`DynamoGraphDeployment`](dynamo-graph-deployment.mdx) `spec.components` list is a `DynamoComponentDeploymentSharedSpec`, the same shape documented here under [Shared component spec](#shared-component-spec). The operator creates one child DCD per component. Author a standalone DCD only when you want to manage a single component's lifecycle independently; otherwise define components inside a DGD.

This page documents the `nvidia.com/v1beta1` API — the served, current version.

<Info>
  This reference covers user-configurable spec fields. For platform installation and operator configuration, see [Install Dynamo](../../kubernetes/installation/install-dynamo.md).
</Info>

## Minimal example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoComponentDeployment
metadata:
  name: my-worker
spec:
  backendFramework: vllm
  type: worker
  replicas: 1
  podTemplate:
    spec:
      containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0
```

## Spec reference

The DCD spec is `backendFramework` plus the [shared component spec](#shared-component-spec) inlined at the same level.

<ParamField path="backendFramework" type="string">
  Inference backend framework for this component. Drives backend-specific defaults the operator injects into the `spec.podTemplate.spec.containers[name=main]` container.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>sglang</Badge> <Badge intent="note" minimal>vllm</Badge> <Badge intent="note" minimal>trtllm</Badge></span>

</ParamField>

### Shared component spec

These fields are shared between a standalone DCD and each entry of a DGD `spec.components` list, except where a field is marked DGD-only. In a DGD, prefix them with `spec.components[*]`. Container paths below use the standalone DCD layout; replace the leading `spec.podTemplate` with `spec.components[*].podTemplate` for DGD components.

<ParamField path="providerOverride" type="ProviderOverride">
  Customizes the primary Grove unit for a component embedded in a DGD. Standalone DCD OpenAPI omits this field. Use `apiVersion: grove.io/v1alpha1`; the target is `PodCliqueTemplateSpec` for a component backed by a PodClique or `PodCliqueScalingGroupConfig` for one backed by a scaling group. The `value` accepts `topologyConstraint` and, at component scope, native `spec.minAvailable` for a standalone clique or `minAvailable` for a scaling group. LPX conductor components support the native minimum but not topology overrides. See [ProviderOverride](dynamo-graph-deployment.mdx#provideroverride).
</ParamField>

<ParamField path="name" type="string" required={true}>
  Stable logical identifier for the component, unique within its parent DGD's `spec.components` list. Must match `^[A-Za-z0-9]([-A-Za-z0-9]*[A-Za-z0-9])?$`, 1–63 characters. For a standalone DCD the defaulting webhook populates `name` from `metadata.name`, so you rarely set it explicitly. The name is decoupled from the underlying workload name so the operator can rename child workloads (for example, hash-suffixing worker DCDs during a rolling update) without losing the identity that labels, status maps, scaling adapters, planner RBAC, and EPP filters depend on.
</ParamField>

<ParamField path="type" type="string">
  Role of the component within the graph. Drives port mapping, frontend detection, planner RBAC, and the pod label `nvidia.com/dynamo-component-type`. `prefill` and `decode` are first-class values for disaggregated serving and can be set directly. At most one component per graph may be `epp`. Immutable after it is set. Standalone DCDs cannot use `lpx`.

  The `lpx` type is experimental, requires the operator's `lpx.enabled` setting, and may change incompatibly.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>frontend</Badge> <Badge intent="note" minimal>worker</Badge> <Badge intent="note" minimal>prefill</Badge> <Badge intent="note" minimal>decode</Badge> <Badge intent="note" minimal>planner</Badge> <Badge intent="note" minimal>epp</Badge> <Badge intent="note" minimal>lpx</Badge></span>
</ParamField>

<ParamField path="lpx" type="LPXConfig">
  Experimental DGD-only compiled-model and scheduling configuration; requires the operator's `lpx.enabled` setting and may change incompatibly. Set `buildId`; placement comes from the build manifest. Runtime options belong in role pod templates. See [LPXConfig](full-api-reference.mdx#lpxconfig).

  When the operator's `lpx.modelRegistryURL` is configured, `buildId` must be relative to that registry; absolute paths and URLs require an empty registry URL.
</ParamField>

<Indent>
  <ParamField path="scheduling" type="SchedulingSpec">
    Configures this LPX component's scheduling attempts. Components may use different deadlines, including when they share a workload.
  </ParamField>

  <Indent>
    <ParamField path="attemptDeadlineSeconds" type="int64">
      Best-effort deadline for each of this component's LPU pipeline requests to leave Pending, including waiting for the scheduler to start. The timer uses a nonzero `status.schedulingStartedAt` when available; otherwise it uses `metadata.creationTimestamp`. A scheduler-provided timestamp for a later scheduling cycle takes precedence over the request's original creation time. Valid values are `1` through `9,223,372,036`; omission means unlimited. This field is not a solver budget or an overall deployment readiness timeout.

      Current-generation `Bound`, `NoFit`, and `Unsupported` receipts are exempt. `Degraded`, `Releasing`, and `Released` receipts are also exempt when their committed execution was accepted for the current request generation. `NoFit` can therefore leave the deployment Pending indefinitely, and a `Bound` request can still wait indefinitely for runtime readiness.

      Deadline handling runs after download checks, compiler-registry resolution, and desired-state validation. An outage or invalid desired configuration can delay failure reporting and cleanup indefinitely, even after the deadline has elapsed. The timer does not bound asynchronous cleanup or finalization.

      Expiry records `LPXSchedulingDeadlineExceeded` before cleanup. Cleanup removes complete engine replicas only from a trailing suffix: an expired interior replica blocks deadline cleanup to preserve healthy higher ordinals. Explicit `replicas` authorizes lowering the Grove replica count to the surviving prefix; externally managed capacity remains unchanged. Grove scaling and owner garbage collection handle pod cleanup; the LPX graph controller does not delete pods directly. A later LPX input revision authorizes republication after cleanup. An existing failure remains cleanup authority for cycles that started before it, so retiring those cycles does not consume the retry edit.
    </ParamField>
  </Indent>

  <ParamField path="experimental" type="LPXExperimentalSpec">
    Groups opt-in LPX options whose shape may change incompatibly between v1beta1 releases. These fields are not covered by the v1beta1 deprecation policy.
  </ParamField>

  <Indent>
    <ParamField path="localPartitions" type="LPXLocalPartitions">
      Selects partitions of a hybrid build that the Cyborg conductor runs on its own GPU. Set `mode: All` to run every partition on the GPU, or `mode: IDs` with `ids` to run the listed partitions. `ids` is required with `IDs` and forbidden with `All`. Omitting `localPartitions` runs every partition on LPUs. Hybrid XT (LP20) and HX (LP30) builds support this field. LPU-only and speculative builds reject it.

      The operator schedules Agent Pods and LPU pipeline requests only for the remaining partitions, and lists only those partitions in the Agents' partition configuration and, for XT builds, the conductor's `lpu_servers`. For HX builds, the request's allocation metadata also lists only the remaining partitions. When every partition is local, it renders no Agent PodClique and creates no LPU pipeline request. It publishes the resolved local partitions to the Cyborg `main` container as `LPX_LOCAL_PARTITION_IDS`, a comma-separated list of compiler partition IDs.

      `ids` lists compiler partition IDs of the build's runtime partitions. A selected prop-sync chain is one runtime partition: select its first partition, which keeps the whole chain on the GPU. Unknown IDs and non-first chain members are rejected during reconciliation. Because a chain moves as a unit, a local selection never splits a chain. The operator keeps a prop-sync connector only when both of its partitions remain on LPUs, and renumbers it to their positions among the remaining partitions. Changing the selection changes the workload digest and rolls the workload.
    </ParamField>
  </Indent>
</Indent>

<ParamField path="runtimeVersionOverride" type="string">
  Declares the Dynamo runtime version in `spec.podTemplate.spec.containers[name=main].image` by default, or `spec.podTemplate.spec.initContainers[name=runtime].image` when the dynamo sidecar is present, taking precedence over the image tag. Use canonical `MAJOR.MINOR.PATCH`, for example `"1.4.0"`, with each part between `0` and `9999` and no leading zeros, `v` prefix, prerelease suffix, or build metadata. This field does not change the image. See [Runtime Version Compatibility](#runtime-version-compatibility) for admission requirements and rollout behavior.
</ParamField>

<ParamField path="replicas" type="integer">
  Desired number of complete component instances. For a single-node component, each instance is one pod. For a multinode component, each instance contains all of its roles. Minimum `0`. When `scalingAdapter` is set, this field is owned by the [DynamoGraphDeploymentScalingAdapter](full-api-reference.mdx#dynamographdeploymentscalingadapter) and should not be modified directly.

  For a single `type: lpx` component in a DGD, this counts complete engine replicas, not individual role Pods. Hybrid and LPU-only engines allow up to `2496` replicas; generated Pod hostname limits may lower that maximum. In a speculative pair, draft replicas count model instances (`1–8`), and target replicas must be `1`. LPX does not support scaling adapters or scale-to-zero.
</ParamField>

<ParamField path="minAvailable" type="integer" deprecated={true}>
  Minimum complete component replicas guaranteed to be gang-scheduled before a shortfall triggers gang termination. Deprecated: use the native minimum through the component `providerOverride`. Supported only for Grove-backed DGD components; rejected for non-Grove deployments. Existing legacy deployments retain this field and its default of `1`. New Grove DGDs without any legacy minimum resolve an omitted native minimum to `1` during rendering, without creating a provider override. Minimum `1`; the effective value is immutable after creation, including when migrating to the native form. Positive `replicas` must be at least this value. Scaling to `0` is allowed and retains `minAvailable` for the next scale-up.

  Under Coherent updates, this value also defines the component's minimum viable replacement unit. With no surge capacity, replacing that unit can make this many replicas unavailable together. Both minimum forms use RollingRecreate unless the DGD strategy annotation explicitly selects Coherent. Migrating component minima to provider overrides leaves the update strategy unchanged. See [Configure native minimum availability](dynamo-graph-deployment.mdx#configure-native-minimum-availability) and [Coherent capacity and disruption](dynamo-graph-deployment.mdx#coherent-capacity-and-disruption).

  For LPX components with omitted `replicas`, `minAvailable` sets the initial Grove scaling-group size; subsequent capacity is externally managed.
</ParamField>

<ParamField path="podTemplate" type="core/v1.PodTemplateSpec">
  Complete Pod template shared by all roles of the component. It is mutually exclusive with `roles[*].podTemplate`. Every component must include `spec.podTemplate.spec.containers[name=main]` with a non-empty `image`. By default the operator injects command, environment, port, probe, resource, and volume-mount defaults into `spec.podTemplate.spec.containers[name=main]`, merging your overrides by name. Declaring `spec.podTemplate.spec.initContainers[name=runtime]` activates [Dynamo Sidecar Mode](#dynamo-sidecar-mode). Existing standard-mode components created without `spec.podTemplate.spec.containers[name=main].image` may retain that omission on unrelated updates. For DGD components whose runtime image has no parseable semantic-version tag, set `runtimeVersionOverride`. Other containers are user-managed and must specify their required fields, including `image`; `frontendSidecar` can explicitly select a regular container for frontend defaults. Replaces the ten separate per-component fields (`resources`, `envs`, `livenessProbe`, and so on) that existed in `v1alpha1`.

  For `type: lpx`, omit this field and use `roles[].podTemplate`. Every LPX role requires its own template with an explicit `main` container image.

  <a href="https://pkg.go.dev/k8s.io/api/core/v1#PodTemplateSpec" target="_blank">core/v1.PodTemplateSpec</a>
</ParamField>

<ParamField path="multinode" type="MultinodeSpec">
  Configures a `worker`, `prefill`, or `decode` component that spans multiple pods. Other component types reject this field. A pre-existing unsupported combination can remain unchanged during unrelated updates or remove `multinode`, but cannot change or reintroduce it. See [Multinode Orchestration](../../kubernetes/installation/multinode-orchestration.md).
</ParamField>

<Indent>
  <ParamField path="nodeCount" type="integer" default="2">
    Number of nodes to deploy. Minimum `2` and immutable after creation. Total GPUs used is `nodeCount × container GPU request`.
  </ParamField>
</Indent>

  <ParamField path="roles" type="[]ComponentRoleSpec">
  Named Pod-producing parts of a compound component, keyed by `name`. Multinode components must contain exactly one `leader` and one `worker`; their cardinality is fixed by `multinode.nodeCount`, and omitted role counts default to `1` and `multinode.nodeCount - 1`, respectively. Omit the field to retain the implicit leader/worker layout. If either multinode role sets `podTemplate`, both roles must set one and the component-level `podTemplate` must be absent.

  LPX components use `agent` for Agent Pods and `conductor` for the serving runtime. Independent LPX components each declare a conductor role with an explicit template. In a speculative pair, the draft has only an agent role and the target declares both roles. A hybrid engine's GPU runtime uses the conductor template.

  Standalone DCD roles cannot use LPX templates or DGD-only provider overrides. See [ComponentRoleSpec](full-api-reference.mdx#componentrolespec).
</ParamField>

<Indent>
  <ParamField path="name" type="string" required={true}>
    Semantic role name: `leader` or `worker` for ordinary multinode components, `conductor` or `agent` for LPX components. Names must be unique within the component.
  </ParamField>
  <ParamField path="replicas" type="integer">
    Logical cardinality of this role in one complete component instance; minimum `1`. For ordinary multinode components, admission defaults and persists an omitted value as `leader: 1` or `worker: multinode.nodeCount - 1`; an explicit value must match that fixed shape. For LPX, omitted conductor replicas use `1`, subject to runtime-specific limits. Agent replicas are derived from the model build and `lpx.experimental.localPartitions`; an explicit count must match that derived count.
  </ParamField>
  <ParamField path="podTemplate" type="core/v1.PodTemplateSpec">
    Complete Pod template for this role. It must include a container named `main` with a non-empty `image`. Role templates do not inherit from each other or from the component-level template.

    An init container named `runtime` is rejected in role templates because Dynamo sidecar mode does not yet support multinode or LPX components.

    For multinode components, supplying complete role templates transfers ownership of backend-specific leader and worker commands and topology arguments to the user; Dynamo skips its vLLM, SGLang, and TensorRT-LLM role-dependent launch rewrites. Common Pod and workload wiring remains operator-owned. For vLLM multiprocessing, Dynamo still injects the shared coordination port `29500`. Multinode role templates cannot currently be combined with GMS or failover. The component remains the lifecycle, rollout, scaling, service, and status boundary.

    For LPX components, every declared role requires its own template. Agent and conductor templates are independent.

    <a href="https://pkg.go.dev/k8s.io/api/core/v1#PodTemplateSpec" target="_blank">core/v1.PodTemplateSpec</a>
  </ParamField>
  <ParamField path="providerOverride" type="ProviderOverride">
    Provider configuration for the workload unit generated for this role. Available only on ordinary multinode components embedded in a DGD. LPX roles do not support it; standalone DCD OpenAPI omits this field. For Grove multinode leader and worker roles, use target `PodCliqueTemplateSpec`; only `topologyConstraint` is allowed in `value`. Role overrides are not supported for inter-pod GMS components. See [ProviderOverride](dynamo-graph-deployment.mdx#provideroverride).
  </ParamField>
</Indent>

<ParamField path="sharedMemorySize" type="resource.Quantity">
  Size of the tmpfs mounted at `/dev/shm`. Omit to use the operator default (`8Gi`); set a positive quantity for a custom size; set `"0"` to disable the shared-memory volume entirely.

  <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/api/resource#Quantity" target="_blank">resource.Quantity</a>
</ParamField>

<ParamField path="globalDynamoNamespace" type="boolean" default="false">
  Places the component in the global Dynamo namespace rather than the per-deployment namespace derived from the DGD name.
</ParamField>

<ParamField path="modelRef" type="ModelReference">
  References a model served by this component. When set, a headless service is created for endpoint discovery.
</ParamField>

<Indent>
  <ParamField path="name" type="string" required={true}>
    Base model identifier, for example `llama-3-70b-instruct-v1`.
  </ParamField>
  <ParamField path="revision" type="string">
    Model revision or version.
  </ParamField>
</Indent>

<ParamField path="scalingAdapter" type="ScalingAdapter">
  Opts the component into a [DynamoGraphDeploymentScalingAdapter](full-api-reference.mdx#dynamographdeploymentscalingadapter) (DGDSA). Set it — even as an empty object, `scalingAdapter: {}` — to create a DGDSA that owns `replicas` so external autoscalers (HPA, KEDA, Planner) can drive scaling through the Scale subresource. Omit the field to opt out.
</ParamField>

<ParamField path="frontendSidecar" type="string">
  Designates a container in each selected complete Pod template as the frontend sidecar. The value must match a container `name` in the component template or every role template; the operator merges its frontend-sidecar defaults (Dynamo env vars, ports, health probes) into that container the same way it merges into `main`.

  Strict namespace-prefix matching uses the sidecar's own image version, independently of the component's runtime container and `runtimeVersionOverride`. If the sidecar image tag does not identify the runtime version, set `DYN_NAMESPACE_PREFIX_STRICT=true` in that container's `env` when its image contains the strict-prefix implementation.
</ParamField>

<ParamField path="compilationCache" type="CompilationCacheConfig">
  Configures a PVC-backed compilation cache. The operator handles backend-specific mount paths and environment variables, so you do not hand-wire them into `podTemplate`.
</ParamField>

<Indent>
  <ParamField path="pvcName" type="string" required={true}>
    Name of a user-created PVC, which must exist in the same namespace as the deployment.
  </ParamField>
  <ParamField path="mountPath" type="string">
    Overrides the backend-specific default mount path. When empty, the operator selects a default appropriate for the backend framework.
  </ParamField>
</Indent>

<Warning>
  Cached artifacts (for example vLLM's `torch.compile`/CUDA graph cache) are specific to the
  backend, torch, and CUDA versions that produced them. After upgrading to a new Dynamo release —
  or any change to the underlying backend image — clear the PVC's contents before redeploying.
  Stale entries from an older build have caused worker crashes at startup (for example an NVML
  assertion failure during CUDA graph capture) that disappear once the cache is emptied.
</Warning>

<ParamField path="eppConfig" type="EPPConfig" deprecated={true}>
  Deprecated: omit `eppConfig` and use the native Rust EPP, which is configured through environment variables and takes no config file. Presence of this field selects the legacy Go EPP pod contract, so existing deployments keep running across an operator upgrade until you clear it.

  Only valid when `type` is `epp`, and its use is decided by the component's resolved runtime version: required below 1.5.0 (legacy Go EPP image), forbidden at 1.5.0 and later (native Rust EPP image). Migrate by clearing `eppConfig` and moving to a 1.5.0+ image in the same update. Exactly one of `configMapRef` or `config` must be set. See the [Gateway API Routing Reference](../components/gateway-api-routing.mdx).
</ParamField>

<Indent>
  <ParamField path="configMapRef" type="core/v1.ConfigMapKeySelector">
    References a user-provided ConfigMap key containing EPP configuration. Mutually exclusive with `config`.

    <a href="https://pkg.go.dev/k8s.io/api/core/v1#ConfigMapKeySelector" target="_blank">core/v1.ConfigMapKeySelector</a>
  </ParamField>
  <ParamField path="config" type="EndpointPickerConfig">
    EPP `EndpointPickerConfig` supplied inline. The operator marshals it to YAML and creates the ConfigMap for you. Mutually exclusive with `configMapRef`.
  </ParamField>
</Indent>

<ParamField path="topologyConstraint" type="TopologyConstraint">
  Component-level topology placement. See the [SpecTopologyConstraint API](full-api-reference.mdx#spectopologyconstraint).
</ParamField>

<Indent>
  <ParamField path="packDomain" type="string" required={true}>
    Topology domain to pack pods within. Must match a domain defined in the referenced `ClusterTopology`. When the parent DGD also sets `spec.topologyConstraint.packDomain`, this value must be narrower than or equal to it.
  </ParamField>
</Indent>

<ParamField path="experimental" type="ExperimentalSpec">
  Opt-in preview features whose API shape may change in breaking ways between `v1beta1` releases. Fields here are **not** covered by the normal `v1beta1` deprecation policy — do not rely on them for production workloads.

  See: [ExperimentalSpec](#experimentalspec)
</ParamField>

## Dynamo Sidecar Mode

Declaring `spec.podTemplate.spec.initContainers[name=runtime]` activates Dynamo sidecar mode. You must declare this init container yourself; the operator merges defaults into it and does not create it. It must specify a non-empty `image` and `restartPolicy: Always`. The name alone selects the mode; a missing restart policy is rejected. `spec.podTemplate.spec.containers[name=runtime]` or an init container with another name does not activate it.

The operator injects Dynamo environment variables, identity, transport TLS, system port, and startup, liveness, and readiness probes into `spec.podTemplate.spec.initContainers[name=runtime]`. The `spec.podTemplate.spec.containers[name=main]` container runs the engine with user-provided launch configuration and probes. Graph-level environment variables apply to both containers; compilation cache and shared-memory configuration remain on `spec.podTemplate.spec.containers[name=main]`. A separate `frontendSidecar` is supported.

Both containers use the existing environment merge behavior: merged entries are sorted by name and de-duplicated, with container-level values overriding graph-level values and runtime defaults. The operator origin version does not change this behavior. Backend-specific additions retain their existing placement; for example, `VLLM_CACHE_ROOT` is appended to the engine environment after merging.

This mode supports `worker`, `prefill`, and `decode` components. Multinode, enabled checkpoint, GPU memory service, and failover are rejected because they are not currently supported in this mode. Support for these features is planned for a future release. Other component types cannot declare `spec.podTemplate.spec.initContainers[name=runtime]`.

The same convention applies to `v1alpha1` through `spec.extraPodSpec.initContainers[name=runtime]`. Conversion preserves the live init-container list. Adding, renaming, or removing `spec.podTemplate.spec.initContainers[name=runtime]` changes the mode and can trigger a worker rollout. Reserve this name for the Dynamo runtime; rename unrelated init containers before upgrading the operator.

```yaml
podTemplate:
  spec:
    containers:
      - name: main
        image: <engine-image>
    initContainers:
      - name: runtime
        image: <dynamo-sidecar-image>
        restartPolicy: Always
```

Set the engine launch configuration and sidecar command for your backend. See the [vLLM sidecar manifests](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/vllm/deploy) for complete examples.

## Runtime Version Compatibility

DGD admission requires `runtimeVersionOverride` for non-LPX components when the runtime image has no parseable semantic-version tag. The runtime image is `spec.components[*].podTemplate.spec.initContainers[name=runtime].image` in Dynamo sidecar mode, or `spec.components[*].podTemplate.spec.containers[name=main].image` otherwise. Also set it when a parseable tag does not represent the Dynamo runtime version in the image. The override must describe the Dynamo runtime actually contained in the image.

When updating an image to a different Dynamo runtime version, update any configured `runtimeVersionOverride` in the same change, or remove it if the new image tag correctly identifies the runtime version. The operator uses the resolved runtime version to select feature gates that control how it renders the component's PodSpec, including flags, environment variables, and probes. For Dynamo 1.5.0 and later, changing the override alone may change the rendered PodSpec or worker revision and trigger a rollout, even when the image reference is unchanged. Keep the override aligned with the Dynamo runtime actually contained in the image.

For components generated by a DGDR, an explicit component override takes precedence over the DGDR-level default. See [Profiler image version compatibility](dynamo-graph-deployment-request.mdx#profiler-image-version-compatibility).

## Status

The operator maintains observed state under `status`.

<ParamField path="status.conditions" type="[]metav1.Condition">
  Standard Kubernetes conditions. `Available` reports whether the component is serving traffic; `DynamoComponentReady` reports whether the underlying Dynamo component is ready.

  <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Condition" target="_blank">metav1.Condition</a>
</ParamField>

<ParamField path="status.observedGeneration" type="integer">
  Most recent `metadata.generation` the controller has reconciled. A component is up to date when this equals `metadata.generation`.
</ParamField>

<ParamField path="status.component" type="ComponentReplicaStatus">
  Replica status for this component: desired, ready, and available counts. Shares the shape documented on the [DGD Reference](dynamo-graph-deployment.mdx#componentreplicastatus).
</ParamField>

## Additional Types

Nested struct types referenced by the fields above, broken out here to keep the field lists shallow. Types prefixed `core/v1.`, `metav1.`, `resource.`, or `runtime.` are standard Kubernetes types and link to their Go package documentation instead of being expanded.

### ExperimentalSpec

Groups opt-in preview features for a component. Referenced by `experimental`. Nested types (`GMSClientPodSpec`, `ComponentCheckpointJobConfig`, `DynamoCheckpointIdentity`) are expanded inline under the field that references them.

<Warning>
  These preview features can change or disappear between `v1beta1` releases without a name-preserving graduation path. They are excluded from the `v1beta1` deprecation policy — do not rely on them for production workloads.
</Warning>

<ParamField path="gpuMemoryService" type="GPUMemoryServiceSpec">
  Configures the GPU Memory Service (GMS). When set, GPU access for GMS clients is managed through Dynamic Resource Allocation (DRA), and the operator replaces the `spec.podTemplate.spec.containers[name=main]` container's GPU resources with a DRA `ResourceClaim`.
</ParamField>

<Indent>
  <ParamField path="mode" type="GPUMemoryServiceMode" default="IntraPod">
    Selects the GMS deployment topology.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>IntraPod</Badge> <Badge intent="note" minimal>InterPod</Badge></span>
  </ParamField>
  <ParamField path="deviceClassName" type="string" default="gpu.nvidia.com">
    The DRA `DeviceClass` to request GPUs from.
  </ParamField>
  <ParamField path="extraClientContainers" type="[]string">
    Additional user-declared containers that should be wired as GMS clients in service pods. `SnapshotJob` capture Pod clients are declared under `checkpoint.job.gmsClientContainers` instead. In each rendered pod, only matching container names are wired; absent names are ignored. Each name must match `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, 1–63 characters.
  </ParamField>
  <ParamField path="extraClientPods" type="[]GMSClientPodSpec">
    Additional GMS client pods for inter-pod GMS. Reserved for future use and rejected until inter-pod client orchestration is wired.
  </ParamField>
  <Indent>
    <ParamField path="name" type="string" required={true}>
      Identifies this client pod. Must match `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, 1–63 characters.
    </ParamField>
    <ParamField path="podTemplate" type="core/v1.PodTemplateSpec" required={true}>
      Configures the pod to run as a GMS client.

      <a href="https://pkg.go.dev/k8s.io/api/core/v1#PodTemplateSpec" target="_blank">core/v1.PodTemplateSpec</a>
    </ParamField>
  </Indent>
</Indent>

<ParamField path="failover" type="FailoverSpec">
  Configures active-passive GPU failover for a worker component. The `spec.podTemplate.spec.containers[name=main]` container is cloned into two engine containers (active + standby) sharing GPUs via DRA, and the standby acquires the flock when the active engine fails. Requires `gpuMemoryService` to be set, `failover.mode` to match `gpuMemoryService.mode`, and the `nvidia.com/dynamo-kube-discovery-mode: container` annotation on the DGD.
</ParamField>

<Indent>
  <ParamField path="mode" type="GPUMemoryServiceMode" default="IntraPod">
    Failover deployment topology. Must match the GMS `mode` on the same component.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>IntraPod</Badge> <Badge intent="note" minimal>InterPod</Badge></span>
  </ParamField>
  <ParamField path="numShadows" type="integer" default="1">
    Number of shadow (standby) engine containers per rank. Reserved for future use; the operator currently creates exactly one shadow. Minimum `1`, maximum `1`.
  </ParamField>
</Indent>

<ParamField path="grove" type="GroveSpec">
  Configures experimental Grove workload rendering for a component in a DGD. Requires the Grove pathway.
</ParamField>

<Indent>
  <ParamField path="forceScalingGroup" type="boolean">
    When `true`, renders a single-node component as a `PodCliqueScalingGroup` with one single-pod PodClique per replica. The first `minAvailable` replicas join the deployment's base PodGang; each additional replica gets a separate PodGang. `false` or omission retains automatic selection: multinode and inter-pod GMS components use scaling groups, while other single-node components use a PodClique. Immutable after creation.
  </ParamField>
</Indent>

<ParamField path="checkpoint" type="ComponentCheckpointConfig">
  Experimental process checkpoint/restore through standalone Snapshot. The API retains the name `checkpoint`; configure it under `spec.components[*].experimental` in a DGD. Enable checkpoint support in the Dynamo operator and install Snapshot before using this field.

  Set `enabled: true` without `checkpointRef` for DGD-managed capture through a `SnapshotJob`. Set `checkpointRef` to restore an existing compatible `PodSnapshot` in the same namespace. Omit the deprecated `mode` and `identity` fields.
</ParamField>

<Warning>
Dynamo v1.5.0 is a hard compatibility boundary for checkpoint resources. Dynamo creates automatic captures through the standalone Snapshot operator's `SnapshotJob` API, and `checkpointRef` names a standalone `PodSnapshot`. Legacy `DynamoCheckpoint` objects and their artifacts cannot be restored.

The upgrade does not delete the legacy `DynamoCheckpoint` CRD, instances, `PodSnapshot` objects, `PodSnapshotContent` objects, PVC data, or stored artifacts. Before upgrading, list retained `DynamoCheckpoint` objects and legacy `PodSnapshot` objects labeled `nvidia.com/snapshot-owner`, recreate any required snapshots through the standalone Snapshot APIs, and delete unneeded legacy checkpoints while the old controller can still run their finalizers. After upgrading, review and remove unsupported resources manually. Do not delete the shared `PodSnapshot` or `PodSnapshotContent` CRDs because the standalone Snapshot operator uses them.
</Warning>

<Indent>
  <ParamField path="enabled" type="boolean" required={true}>
    Whether checkpointing is enabled for this component. When `true`, omit `checkpointRef` for a DGD-managed automatic checkpoint, or set `checkpointRef` to restore a `PodSnapshot` in the same namespace. Omit the `checkpoint` block, or set `enabled: false`, to disable checkpointing.
  </ParamField>
  <ParamField path="startupPolicy" type="CheckpointStartupPolicy" default="Immediate">
    When serving workers start relative to automatic capture. `Immediate` starts them cold while the `SnapshotJob` runs; only Pods created after capture completes restore. Snapshot readiness does not restart existing workers. `WaitForCheckpoint` keeps serving replicas at zero until capture completes and the `PodSnapshot` is Ready, then starts them from it. Capture completion includes any helper containers that persist state.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Immediate</Badge> <Badge intent="note" minimal>WaitForCheckpoint</Badge></span>
  </ParamField>
  <ParamField path="deletionPolicy" type="CheckpointDeletionPolicy" default="Delete">
    Whether DGD-managed capture resources and snapshot artifacts are deleted or retained when the owning DGD is deleted. Explicit `checkpointRef` PodSnapshots are never owned or deleted by the DGD, and retained automatic checkpoints are not valid `checkpointRef` targets.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Delete</Badge> <Badge intent="note" minimal>Retain</Badge></span>
  </ParamField>
  <ParamField path="checkpointRef" type="string">
    Names an existing `PodSnapshot` in the same namespace. Before workers can restore, the snapshot must be Ready and carry matching Dynamo compatibility metadata for the target worker configuration. If the named snapshot does not exist, its capture has failed, or its compatibility metadata does not match, the operator reports an error and does not restore workers from it. Both startup policies perform these checks.

    Standalone worker-class (`worker`, `prefill`, or `decode`) `DynamoComponentDeployment` resources cannot set this field; configure it on the component in the owning DGD. Do not set `job` with `checkpointRef`.
  </ParamField>
  <ParamField path="targetContainerName" type="string" default="main">
    The workload container to snapshot and restore. Must match `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, 1–63 characters.

    Automatic captures run this container under Snapshot's cuInterpose launcher, so it must set `command`. The `main` container always has one, because the operator defaults it to `/bin/sh -c`. To opt out, set the pod template annotation `nvidia.com/cuda-shared-memory-support: disabled`. Multi-GPU captures can then fail when NCCL uses cuMem or NVLS memory.
  </ParamField>
  <ParamField path="job" type="ComponentCheckpointJobConfig">
    Customizes the DGD-managed `SnapshotJob` capture Pod.
  </ParamField>
  <Indent>
    <ParamField path="gmsClientContainers" type="[]string">
      `SnapshotJob` capture Pod containers that should receive GMS client wiring. Requires `gpuMemoryService` on the component. Each name must match `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, 1–63 characters.
    </ParamField>
    <ParamField path="podTemplate" type="core/v1.PodTemplateSpec">
      Customizes the `SnapshotJob` capture Pod. The operator starts from the selected workload container and merges this template, so you can add helper containers such as `gms-saver`.

      <a href="https://pkg.go.dev/k8s.io/api/core/v1#PodTemplateSpec" target="_blank">core/v1.PodTemplateSpec</a>
    </ParamField>
  </Indent>
  <ParamField path="mode" type="CheckpointMode" deprecated={true}>
    Deprecated: omit `mode`. Use `enabled: true` without `checkpointRef` for a DGD-managed automatic checkpoint, or use `checkpointRef` to restore a named PodSnapshot.

    <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>Auto</Badge> <Badge intent="note" minimal>Manual</Badge></span>
  </ParamField>
  <ParamField path="identity" type="DynamoCheckpointIdentity" deprecated={true}>
    Deprecated: omit for DGD-managed checkpoints; the operator ignores this field. Use `checkpointRef` to restore an existing `PodSnapshot`.
  </ParamField>
  <Indent>
    <ParamField path="model" type="string" required={true} deprecated={true}>
      Model identifier, for example `meta-llama/Llama-3-70B`.
    </ParamField>
    <ParamField path="backendFramework" type="string" required={true} deprecated={true}>
      Runtime framework.

      <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>vllm</Badge> <Badge intent="note" minimal>sglang</Badge> <Badge intent="note" minimal>trtllm</Badge></span>
    </ParamField>
    <ParamField path="dynamoVersion" type="string" deprecated={true}>
      Dynamo platform version.
    </ParamField>
    <ParamField path="tensorParallelSize" type="integer" default="1" deprecated={true}>
      Tensor parallel configuration. Deprecated: checkpoint launch uses the pod template instead. Minimum `1`.
    </ParamField>
    <ParamField path="pipelineParallelSize" type="integer" default="1" deprecated={true}>
      Pipeline parallel configuration. Deprecated: checkpoint launch uses the pod template instead. Minimum `1`.
    </ParamField>
    <ParamField path="dtype" type="string" deprecated={true}>
      Data type, for example `fp16`, `bf16`, or `fp8`.
    </ParamField>
    <ParamField path="maxModelLen" type="integer" deprecated={true}>
      Maximum sequence length. Minimum `1`.
    </ParamField>
    <ParamField path="extraParameters" type="map[string]string" deprecated={true}>
      Legacy identity parameters. Ignored for DGD-managed automatic capture.
    </ParamField>
  </Indent>
</Indent>

## Inspect a DCD

```bash
# List component deployments (short name: dcd)
kubectl get dcd -n $NAMESPACE

# Detailed status and conditions
kubectl describe dcd my-worker -n $NAMESPACE

# Readiness at a glance
kubectl get dcd my-worker -n $NAMESPACE \
  -o jsonpath='{.status.conditions[?(@.type=="Available")].status}'
```

## Related pages

<CardGroup cols={2}>
  <Card title="DGD Reference" icon="diagram-project" href="dynamo-graph-deployment.mdx">
    The graph resource whose `components` embed this spec.
  </Card>
  <Card title="DGDR Reference" icon="wand-magic-sparkles" href="dynamo-graph-deployment-request.mdx">
    Generate a DGD by intent instead of authoring components.
  </Card>
  <Card title="Deploy with DGD" icon="rocket" href="../../kubernetes/model-deployment/deploy-with-dgd.md">
    Task-oriented walkthrough of authoring a graph deployment.
  </Card>
  <Card title="Install Dynamo" icon="gears" href="../../kubernetes/installation/install-dynamo.md">
    Install and configure the Dynamo platform and operator.
  </Card>
</CardGroup>
