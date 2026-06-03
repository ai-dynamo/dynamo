# NVSnapshot v1alpha1 API

**Date:** 2026-05-25
**Status:** Draft
**Scope:** API surface only. Controller behavior, agent internals, RBAC, admission webhook configuration, and deployment concerns are out of scope.

---

## 1. Overview

NVSnapshot is a Kubernetes API for container checkpoint/restore (c/r). It is modeled on `VolumeSnapshot` but adapted to active-workload c/r — the snapshot subject is a running process tree, not a static volume.

### 1.1 The four APIs

| Kind | Scope | Role |
|---|---|---|
| `Snapshot` | namespaced | **Primary primitive.** Identifies a captured artifact; produces or binds a `SnapshotContent`; consumed by restore. Can be created directly by the user against an existing pod. |
| `SnapshotContent` | cluster-scoped | Artifact-of-record. Describes where the dump lives and what runtime environment produced it. |
| `SnapshotJob` | namespaced | **Convenience wrapper over `Snapshot`.** Adds pod creation: runs a fresh pod from a PodTemplate, waits for quiesce, dumps. Produces a `Snapshot` + `SnapshotContent`. Ergonomic shortcut when the user doesn't want to manage the source pod themselves. |
| `RestoreSnapshot` | namespaced | Operator-driven restore. Alternative to Mode A (annotation-on-pod); v1alpha1 will pick one of the two (see section 8 Q2). Targets an existing pod and drives CRIU replay against a referenced `Snapshot`. |

### 1.2 Relationships

```
                ┌────────────────────┐
                │    SnapshotJob     │ (namespaced)
                │     producer       │
                └──────────┬─────────┘
                           │ runs PodTemplate; on success
                           │ produces (no ownership)
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐         ┌─────────────────┐
     │    Snapshot    │ ◀─────▶ │ SnapshotContent │
     │  (namespaced)  │  bound  │  (cluster-scoped)│
     │    binding     │         │     artifact    │
     └────────┬───────┘         └─────────────────┘
              │
              │ referenced by pod annotation
              ▼
     ┌──────────────────────────────┐
     │  Restore (Mode A) — pod      │
     │  carries nvsnapshot.io/      │
     │  restore-from annotation     │
     └──────────────────────────────┘
```

**Key facts:**

- **`SnapshotJob` does not own its outputs.** Deleting the SnapshotJob leaves the produced `Snapshot` and `SnapshotContent` in place. Artifact lifecycle follows the backing PV's `persistentVolumeReclaimPolicy`.
- **`Snapshot` ↔ `SnapshotContent` is bidirectional binding** (mirrors `VolumeSnapshot` ↔ `VolumeSnapshotContent`):
  - `SnapshotContent.spec.snapshotRef` is the **back-pointer** (cluster-scoped → namespaced). Set by the controller at creation time.
  - `Snapshot.status.boundSnapshotContentName` is the **forward-pointer**. Set by the controller once binding is verified.
  - Consumers MUST verify the binding by checking that `Snapshot.status.boundSnapshotContentName` and `SnapshotContent.spec.snapshotRef.{namespace,name,uid}` agree before treating the artifact as usable. UID detects stale references.
- **`Snapshot` is the primary primitive; `SnapshotJob` is convenience.** Users can create a `Snapshot` directly against an existing pod (low-level path). Or they can create a `SnapshotJob` with a PodTemplate; the SnapshotJob controller creates the pod and then a Snapshot on top of it (ergonomic path). The result — a `Snapshot` + `SnapshotContent` — is identical either way.
- **Restore consumes via Snapshot or SnapshotContent reference**, never via SnapshotJob.

### 1.3 Quick start examples

Minimal YAML for the three most common operations. Field semantics are in section 2; these are anchors only.

**Produce a snapshot via SnapshotJob (ergonomic):**

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: SnapshotJob
metadata:
  namespace: default
  name: worker-snap
spec:
  storage:
    pvc: { claimName: nvsnapshot-artifacts }
  podTemplate:
    spec:
      containers:
        - name: main
          image: my-app:v1.2.3@sha256:abc...
  targetContainers: [main]
  quiesceProbe:
    action:
      httpGet: { path: /quiesce, port: 9090 }
```

**Produce a snapshot directly against a live pod (low-level):**

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: Snapshot
metadata:
  namespace: default
  name: worker-snap
spec:
  source:
    podRef: { name: worker-pod-7f9 }
  storage:
    pvc: { claimName: nvsnapshot-artifacts }
  # quiesceProbe omitted → agent uses pod's Ready condition
```

**Restore a pod from a snapshot (Mode A, annotation):**

```yaml
apiVersion: v1
kind: Pod
metadata:
  namespace: default
  name: worker-restored
  annotations:
    nvsnapshot.io/restore-from: worker-snap
spec:
  containers:
    - name: main
      image: my-app:v1.2.3@sha256:abc...
```

---

## 2. The four APIs

Each section below presents the API as Go types in package `apis/nvsnapshot/v1alpha1`. Field-level docstrings double as the canonical field semantics. Validation markers (`+kubebuilder:...`) are the source of truth for CRD generation.

> **Multi-version note**: v1alpha1 is the only version of this API.
> `+kubebuilder:storageversion` is intentionally absent from all three
> root types — adding it on a single-version CRD provides no value and
> sends a misleading signal that hub/spoke conversion is wired up.
> When v1beta1 is introduced, add the marker, declare a hub type, and
> implement a conversion webhook in the same change.

### 2.1 Snapshot

**Role.** The primary primitive for container checkpoint. Identifies *what* was captured (always an existing pod). The unit of capture is one or more containers within a pod.

Two ways to create one:

1. **Directly** — user creates a Snapshot with `Spec.Source.PodRef` pointing at an existing pod and `Spec.QuiesceProbe` + `Spec.Storage` populated. The pod must already satisfy the snapshot contract (control volume, target-container labels, NRI-injected binaries). This is the canonical low-level path.
2. **Via SnapshotJob** (convenience) — user creates a `SnapshotJob` with a PodTemplate. The SnapshotJob controller creates the source pod, then creates a Snapshot on top of it (with `QuiesceProbe` and `Storage` copied from the SnapshotJob and `PodRef` set to the new pod). The produced Snapshot is identical to one created directly.

**Cross-references.**

- **Binds to** a `SnapshotContent` via `Status.BoundSnapshotContentName`. Set by the controller once the dump completes and the binding is verified against `SnapshotContent.spec.snapshotRef`.

```go
// Snapshot is the user-facing binding for a container checkpoint.
// It identifies what was captured and is consumed by restore paths.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=snap
type Snapshot struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   SnapshotSpec   `json:"spec,omitempty"`
    Status SnapshotStatus `json:"status,omitempty"`
}
```

#### Spec

```go
// SnapshotSpec describes what this snapshot is OF and how it should be
// captured. Two creation flows:
//   1. SnapshotJob-produced: the SnapshotJob controller creates a
//      Snapshot, copying its quiesceProbe and storage into Snapshot.spec.
//   2. Direct user creation (live-pod): the user creates a Snapshot
//      with PodRef pointing at an existing pod that already satisfies
//      the snapshot contract (control volume, target-container labels,
//      etc.). The controller drives the dump.
// +kubebuilder:validation:XValidation:rule="self.source == oldSelf.source",message="source is immutable after binding"
type SnapshotSpec struct {
    // Source identifies the origin of this snapshot. Immutable after binding.
    // +kubebuilder:validation:Required
    Source SnapshotSource `json:"source"`

    // QuiesceProbe defines how the agent detects that the target
    // containers are safe to dump. When omitted, the agent uses the
    // pod's standard Ready condition (status.conditions[type=Ready]==True)
    // as the quiesce gate — i.e., the workload's own readinessProbe
    // doubles as the quiesce signal.
    // +optional
    QuiesceProbe *QuiesceProbe `json:"quiesceProbe,omitempty"`

    // Storage describes where the artifact will be written.
    // Required for direct creation; for SnapshotJob-produced Snapshots,
    // copied from SnapshotJob.spec.storage at creation time.
    // +kubebuilder:validation:Required
    Storage SnapshotStorage `json:"storage"`
}

// SnapshotSource identifies the captured workload. Kept as a struct
// (rather than inlined PodRef) so future variants can be added additively.
type SnapshotSource struct {
    // PodRef references the pod whose containers were (or will be)
    // captured. For direct creation, the pod must exist at Snapshot
    // creation time and satisfy the snapshot contract.
    // +kubebuilder:validation:Required
    PodRef PodReference `json:"podRef"`
}

// PodReference identifies the pod that was (or will be) captured
// and optionally narrows which of its containers are in scope.
//
// The pod's UID is intentionally NOT carried here: it is runtime
// state, not desired state, and the user cannot know it at Snapshot
// creation time. If the controller wrote UID back into the spec, the
// `self.source == oldSelf.source` immutability CEL on SnapshotSpec
// would reject the controller's own update.
type PodReference struct {
    // Name of the source pod, in the same namespace as the Snapshot.
    // +kubebuilder:validation:Required
    Name string `json:"name"`

    // Containers narrows the snapshot scope to specific containers
    // within the pod. If empty, all containers in the pod are captured.
    // +optional
    Containers []string `json:"containers,omitempty"`
}
```

#### Status

```go
// SnapshotStatus is the observed state of a Snapshot.
type SnapshotStatus struct {
    // BoundSnapshotContentName is the name of the SnapshotContent object
    // this Snapshot is bound to. nil until the binding is established.
    //
    // Consumers MUST verify binding by checking that both Snapshot and
    // SnapshotContent point at each other before treating this object
    // as usable for restore.
    // +optional
    BoundSnapshotContentName *string `json:"boundSnapshotContentName,omitempty"`

    // CreationTime is the timestamp at which the artifact was created
    // (CRIU dump complete, manifest finalized). Mirrors the bound
    // SnapshotContent's CreationTime.
    // +optional
    CreationTime *metav1.Time `json:"creationTime,omitempty"`

    // Conditions reflect the latest observations of the Snapshot's state.
    // Standard condition types:
    //   Ready  — True when the bound SnapshotContent is Ready and the
    //            artifact is usable for restore.
    //   Failed — True if creation or binding failed terminally; reason
    //            and message carry the detail.
    // Each condition carries Status, Reason, Message, LastTransitionTime,
    // and ObservedGeneration per the standard metav1.Condition shape.
    // +optional
    // +patchStrategy=merge
    // +patchMergeKey=type
    // +listType=map
    // +listMapKey=type
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}
```

---

### 2.2 SnapshotContent

**Role.** The artifact-of-record. Cluster-scoped so it survives Snapshot deletion (under `Retain` policy) and may be re-bound across namespaces in the future. Carries enough metadata for a consumer to (a) locate the artifact and (b) understand the runtime environment that produced it.

**Cross-references.**

- **Back-pointer** to its bound `Snapshot` via `Spec.SnapshotRef`. Required and immutable after binding; populated by the controller at SnapshotContent creation, naming the produced Snapshot.

```go
// SnapshotContent is the cluster-scoped artifact-of-record for a
// captured container checkpoint. Equivalent to VolumeSnapshotContent.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster,shortName=snapcontent
type SnapshotContent struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   SnapshotContentSpec   `json:"spec,omitempty"`
    Status SnapshotContentStatus `json:"status,omitempty"`
}
```

#### Spec

```go
// SnapshotContentSpec describes the artifact and its lifecycle policy.
// +kubebuilder:validation:XValidation:rule="self.snapshotRef == oldSelf.snapshotRef",message="snapshotRef is immutable after binding"
// +kubebuilder:validation:XValidation:rule="self.source == oldSelf.source",message="source is immutable after binding"
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.runtimeInformation) || (has(self.runtimeInformation) && self.runtimeInformation == oldSelf.runtimeInformation)",message="runtimeInformation is immutable after binding"
type SnapshotContentSpec struct {
    // SnapshotRef is the back-pointer to the bound Snapshot. May span
    // namespaces since SnapshotContent is cluster-scoped. Immutable
    // after binding.
    // +kubebuilder:validation:Required
    SnapshotRef SnapshotReference `json:"snapshotRef"`

    // Source describes where the artifact is stored. v1alpha1: PVC only.
    // Immutable after binding.
    // +kubebuilder:validation:Required
    Source SnapshotContentSource `json:"source"`

    // RuntimeInformation is a free-form map describing the runtime
    // environment that produced the artifact — typically GPU model,
    // driver version, node arch, CRIU version, etc.
    //
    // The agent performs a verification check at restore time: the
    // SnapshotContent's runtimeInformation is compared against the
    // restore node's actual runtime, and the restore is declined if
    // incompatible. The API does not enforce compatibility at
    // admission or scheduling — consumers should still steer the
    // restore pod to a compatible node via nodeSelector/affinity,
    // but the agent provides a final safety net at replay time.
    //
    // Keys are conventionally dotted (`gpu.model`, `gpu.driverVersion`,
    // `node.arch`, `criu.version`, ...) but not schema-enforced —
    // drivers may publish whatever keys are useful. Consumers should
    // tolerate missing keys.
    //
    // Populated by the producer for dynamic creation; supplied by the
    // user for pre-provisioned SnapshotContent import. Immutable after
    // binding.
    // +optional
    RuntimeInformation map[string]string `json:"runtimeInformation,omitempty"`
}

// SnapshotReference is a cross-namespace reference to a Snapshot.
type SnapshotReference struct {
    // Namespace of the referent.
    // +kubebuilder:validation:Required
    Namespace string `json:"namespace"`

    // Name of the referent.
    // +kubebuilder:validation:Required
    Name string `json:"name"`

    // UID of the referent. Populated at binding time to detect stale
    // references (e.g., if the original Snapshot is deleted and a new
    // one with the same name is created).
    // +optional
    UID types.UID `json:"uid,omitempty"`
}

// SnapshotContentSource describes the artifact backend via an opaque,
// driver-specific handle. SnapshotContent is cluster-scoped and cannot
// directly reference namespaced resources (e.g., PVCs); the handle
// encodes any cross-namespace location information as a string.
//
// v1alpha1 supports the PVC backend; future backends (object storage,
// URI, etc.) use different handle formats but the same field shape.
type SnapshotContentSource struct {
    // SnapshotHandle is an opaque, driver-specific identifier for the
    // physical artifact.
    //
    // For the v1alpha1 PVC backend, the format is:
    //   pvc://<namespace>/<claimName>/<basePath>
    //
    // The PVC's namespace is encoded in the URI because SnapshotContent
    // (cluster-scoped) cannot carry a structured reference to a
    // namespaced PVC. The basePath component is producer-assigned
    // (derived from the producing SnapshotJob's UID) to guarantee global
    // uniqueness; consumers must not interpret its structure.
    //
    // Set ONLY for pre-provisioned (imported) SnapshotContents. For
    // dynamic creation, leave nil — the controller writes the resolved
    // handle to status.snapshotHandle. The presence of this field is
    // the discriminator between import (set) and dynamic (nil). Having
    // the controller-assigned handle live in status (not spec) keeps
    // the controller's write outside the spec.source immutability CEL.
    //
    // Immutable after binding (CEL on enclosing spec.source).
    // +optional
    // +kubebuilder:validation:MinLength=1
    SnapshotHandle *string `json:"snapshotHandle,omitempty"`
}

// RuntimeInformation is a free-form key-value map. There is no `Platform`
// Go type — the field on SnapshotContent.spec is typed directly as
// `map[string]string`. The schema-less shape lets drivers publish
// arbitrary metadata (and consumers add new keys without API changes).
//
// Conventional keys for the v1alpha1 CRIU+CUDA driver:
//
//   | Key                  | Example value          |
//   |----------------------|------------------------|
//   | node.arch            | amd64                  |
//   | gpu.model            | NVIDIA H100            |
//   | gpu.driverVersion    | 535.104.05             |
//   | criu.version         | 3.19                   |
//
// Consumers should treat the map as advisory: missing keys are normal,
// unrecognized keys are normal. Strict-equality immutability after
// binding is enforced via CEL at the spec level (see XValidation on
// SnapshotContentSpec).

```

#### Status

```go
// SnapshotContentStatus is the observed state of a SnapshotContent.
type SnapshotContentStatus struct {
    // SnapshotHandle is the canonical, validated handle for the artifact.
    // Authoritative location:
    //   - Dynamic creation: controller resolves the handle post-dump and
    //     writes it here. spec.source.snapshotHandle is nil.
    //   - Pre-provisioned import: controller accepts the user-supplied
    //     spec.source.snapshotHandle, validates the artifact, and mirrors
    //     the validated value here.
    // Consumers should read this field, not spec.source.snapshotHandle.
    // +optional
    SnapshotHandle *string `json:"snapshotHandle,omitempty"`

    // CreationTime is the timestamp at which the artifact was created
    // (CRIU dump complete, manifest finalized). For pre-provisioned
    // imports, this is the time recorded in the artifact manifest.
    // +optional
    CreationTime *metav1.Time `json:"creationTime,omitempty"`

    // RestoreSize is the on-PVC artifact size in bytes. Consumers may
    // use this to size restore-target volumes / nodes.
    // +optional
    // +kubebuilder:validation:Minimum=0
    RestoreSize *int64 `json:"restoreSize,omitempty"`

    // Conditions reflect the latest observations of the SnapshotContent.
    // Standard condition types:
    //   Ready  — True when the artifact is complete and usable for restore.
    //   Failed — True if artifact provisioning or validation failed
    //            terminally; reason and message carry the detail.
    // +optional
    // +patchStrategy=merge
    // +patchMergeKey=type
    // +listType=map
    // +listMapKey=type
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}
```

---

### 2.3 SnapshotJob

**Role.** A convenience wrapper over `Snapshot` that adds pod creation. Analogous to `batch/v1.Job` in shape, but the lifecycle is: launch a pod from a PodTemplate, wait for quiesce, create a `Snapshot` against that pod, drive the dump, produce a `SnapshotContent`. The `Snapshot` produced by a SnapshotJob is indistinguishable from one created directly by the user (section 2.1); SnapshotJob exists purely so users don't have to manage the source pod themselves.

**Cross-references.**

- **Produces** a `Snapshot` + `SnapshotContent` (no ownership — deleting the SnapshotJob does not delete the artifacts).
- `Status.ContentName` names the produced `SnapshotContent`; restore paths reference the artifact by this name.

```go
// SnapshotJob is a one-shot producer of a Snapshot + SnapshotContent.
// Analogous to batch/v1.Job. Does not own the produced artifacts.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=snapjob
type SnapshotJob struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   SnapshotJobSpec   `json:"spec,omitempty"`
    Status SnapshotJobStatus `json:"status,omitempty"`
}
```

#### Spec

```go
// SnapshotJobSpec describes the desired snapshot production operation.
type SnapshotJobSpec struct {
    // PodTemplate is the pod to launch. The application running in the
    // target containers must implement the quiesce contract — see
    // QuiesceProbe.
    // +kubebuilder:validation:Required
    PodTemplate corev1.PodTemplateSpec `json:"podTemplate"`

    // QuiesceProbe defines how to determine when the pod is safe to dump.
    // When omitted, the agent uses the pod's standard Ready condition
    // (status.conditions[type=Ready]==True) as the quiesce gate.
    // +optional
    QuiesceProbe *QuiesceProbe `json:"quiesceProbe,omitempty"`

    // TargetContainers narrows which containers in PodTemplate are
    // captured. If empty, all containers in the PodTemplate are captured.
    // +optional
    TargetContainers []string `json:"targetContainers,omitempty"`

    // Storage describes where the artifact will be written.
    // v1alpha1: PVC only.
    // +kubebuilder:validation:Required
    Storage SnapshotStorage `json:"storage"`

    // ActiveDeadlineSeconds bounds the total lifetime of the dump
    // operation (pod scheduling + quiesce wait + dump execution).
    // +optional
    // +kubebuilder:default=3600
    // +kubebuilder:validation:Minimum=1
    ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

}

// SnapshotStorage describes the artifact destination. Shared between
// Snapshot.spec (direct creation) and SnapshotJob.spec (factory). v1alpha1
// supports PVC only; tagged-union shape preserved for additive future variants.
// +kubebuilder:validation:XValidation:rule="has(self.pvc)",message="exactly one storage variant must be set; v1alpha1 supports only pvc"
type SnapshotStorage struct {
    // PVC names the PersistentVolumeClaim that will store the artifact.
    // The artifact directory inside the PVC is producer-assigned (derived
    // from the producing Snapshot or SnapshotJob UID), not user-settable.
    // +optional
    PVC *PVCStorage `json:"pvc,omitempty"`
}

// PVCStorage references a PersistentVolumeClaim in the same namespace
// as the producing Snapshot or SnapshotJob.
type PVCStorage struct {
    // ClaimName of the PVC. Must support the access mode required by
    // the deployment (ReadWriteMany for production; ReadWriteOnce
    // permitted for single-node test/dev with appropriate nodeAffinity).
    // +kubebuilder:validation:Required
    ClaimName string `json:"claimName"`
}
```

#### Status

```go
// SnapshotJobStatus is the observed state of a SnapshotJob. Producer
// users read this object only — fields here are the minimum needed to
// drive restore consumption (one-pane-of-glass UX).
type SnapshotJobStatus struct {
    // Phase is the high-level lifecycle stage.
    // +optional
    // +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
    Phase SnapshotJobPhase `json:"phase,omitempty"`

    // ContentName is the name of the produced SnapshotContent. Restore
    // paths reference the artifact by this name.
    // +optional
    ContentName string `json:"contentName,omitempty"`

    // StartedAt is the time at which the SnapshotJob entered the
    // Running phase.
    // +optional
    StartedAt *metav1.Time `json:"startedAt,omitempty"`

    // CompletedAt is the time at which the SnapshotJob reached a
    // terminal phase (Succeeded or Failed).
    // +optional
    CompletedAt *metav1.Time `json:"completedAt,omitempty"`

    // Conditions reflect the latest observations of the SnapshotJob.
    // Standard condition types (modeled on batch/v1.Job):
    //   Complete — True once dump finished and Snapshot+SnapshotContent
    //              were produced.
    //   Failed   — True if dump failed (deadline, pod failure, agent error);
    //              reason and message carry the detail.
    // +optional
    // +patchStrategy=merge
    // +patchMergeKey=type
    // +listType=map
    // +listMapKey=type
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// SnapshotJobPhase is the lifecycle stage of a SnapshotJob.
// The enum validation marker lives on SnapshotJobStatus.Phase (the
// consuming field) — controller-gen reads enum markers from fields,
// not from type aliases.
type SnapshotJobPhase string

const (
    // SnapshotJobPhasePending: SnapshotJob is created but pod has not started.
    SnapshotJobPhasePending SnapshotJobPhase = "Pending"

    // SnapshotJobPhaseRunning: pod is running; may be in quiesce or dump phase.
    SnapshotJobPhaseRunning SnapshotJobPhase = "Running"

    // SnapshotJobPhaseSucceeded: dump complete; SnapshotContent + Snapshot produced.
    SnapshotJobPhaseSucceeded SnapshotJobPhase = "Succeeded"

    // SnapshotJobPhaseFailed: dump failed (deadline, pod failure, agent error).
    SnapshotJobPhaseFailed SnapshotJobPhase = "Failed"
)
```

---

### 2.4 RestoreSnapshot

**Role.** The operator-driven restore object. Targets an existing pod and drives CRIU replay against a referenced `Snapshot`. An alternative to the annotation-driven restore path (section 4) — v1alpha1 will pick one of the two (section 8 Q2). Provides a first-class CRD operators can `kubectl get`, watch, and report status against.

**Cross-references.**

- **Consumes** a `Snapshot` (namespace-local, via `Spec.Source.SnapshotName`). Never references the cluster-scoped `SnapshotContent` directly.
- **Targets** an existing pod (via `Spec.PodRef`) in the same namespace.

```go
// RestoreSnapshot drives CRIU replay of a captured artifact into an
// existing pod. Complements Mode A annotation-driven restore (section 4) by
// giving operators a discoverable, status-bearing CRD.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=restoresnap
type RestoreSnapshot struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   RestoreSnapshotSpec   `json:"spec,omitempty"`
    Status RestoreSnapshotStatus `json:"status,omitempty"`
}
```

#### Spec

```go
// RestoreSnapshotSpec describes a single restore operation.
// +kubebuilder:validation:XValidation:rule="self.source == oldSelf.source",message="source is immutable after creation"
// +kubebuilder:validation:XValidation:rule="self.podRef == oldSelf.podRef",message="podRef is immutable after creation"
type RestoreSnapshotSpec struct {
    // Source identifies the artifact to restore from (a Snapshot in
    // the same namespace as this RestoreSnapshot).
    // +kubebuilder:validation:Required
    Source RestoreSource `json:"source"`

    // PodRef identifies the existing pod to restore into. The pod must
    // be in the same namespace as the RestoreSnapshot. The pod must
    // exist and be in a state compatible with restore shaping (see
    // section 4 Restore consumption for the API contract).
    // +kubebuilder:validation:Required
    PodRef PodReference `json:"podRef"`

    // TargetContainers narrows which containers in the target pod are
    // restored. If empty, all containers covered by the referenced
    // artifact are restored.
    // +optional
    TargetContainers []string `json:"targetContainers,omitempty"`
}

// RestoreSource identifies the artifact to restore from. Kept as a
// struct (rather than inlined SnapshotName) so future variants can be
// added additively. RestoreSnapshot never references the cluster-scoped
// SnapshotContent directly — namespace boundary is the security control.
type RestoreSource struct {
    // SnapshotName references a Snapshot in the same namespace as the
    // RestoreSnapshot. The referenced Snapshot must have its Ready
    // condition set to True.
    // +kubebuilder:validation:Required
    SnapshotName string `json:"snapshotName"`
}
```

#### Status

```go
// RestoreSnapshotStatus is the observed state of a restore operation.
type RestoreSnapshotStatus struct {
    // Phase is the high-level lifecycle stage.
    // +optional
    Phase RestoreSnapshotPhase `json:"phase,omitempty"`

    // StartedAt is the time at which restore shaping began on the target pod.
    // +optional
    StartedAt *metav1.Time `json:"startedAt,omitempty"`

    // CompletedAt is the time at which the restore reached a terminal phase.
    // +optional
    CompletedAt *metav1.Time `json:"completedAt,omitempty"`

    // Conditions reflect the latest observations of the restore.
    // Standard condition types:
    //   Complete — True once CRIU replay finished and the target pod's
    //              restored containers are running.
    //   Failed   — True if restore failed; reason and message carry the detail.
    // +optional
    // +patchStrategy=merge
    // +patchMergeKey=type
    // +listType=map
    // +listMapKey=type
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// RestoreSnapshotPhase is the lifecycle stage of a RestoreSnapshot.
// +kubebuilder:validation:Enum=Pending;Restoring;Ready;Failed
type RestoreSnapshotPhase string

const (
    // RestoreSnapshotPhasePending: RestoreSnapshot exists; target pod
    // not yet shaped, or referenced artifact not yet Ready.
    RestoreSnapshotPhasePending RestoreSnapshotPhase = "Pending"

    // RestoreSnapshotPhaseRestoring: target pod has been shaped; CRIU
    // replay is in progress.
    RestoreSnapshotPhaseRestoring RestoreSnapshotPhase = "Restoring"

    // RestoreSnapshotPhaseReady: CRIU replay completed; the target
    // pod's restored containers are running.
    RestoreSnapshotPhaseReady RestoreSnapshotPhase = "Ready"

    // RestoreSnapshotPhaseFailed: restore failed and will not be retried automatically.
    RestoreSnapshotPhaseFailed RestoreSnapshotPhase = "Failed"
)
```

**API contract.**

- The target pod must exist in the same namespace as the RestoreSnapshot. If the pod does not exist at RestoreSnapshot creation, the RestoreSnapshot remains in phase `Pending`; on pod appearance the controller transitions to `Restoring`.
- The referenced Snapshot must have its `Ready` condition set to True. If not ready, the RestoreSnapshot stays in `Pending` until it becomes ready or the RestoreSnapshot is deleted (`Spec.Source` is immutable).
- On successful CRIU replay, the target pod publishes `nvsnapshot.io/restored` and the RestoreSnapshot transitions to `Ready`.
- A single `RestoreSnapshot` targets one pod (`PodRef`). To restore many pods, create one RestoreSnapshot per pod (typically driven by an outer controller).
- The same artifact (`Source`) may be restored into many target pods via many RestoreSnapshots; the SnapshotContent is not consumed by restore.
- Mode A (annotation-on-pod, section 4.1) and Mode B (this CRD) are alternative restore designs. v1alpha1 ships one of the two — see section 8 Q2.

---

## 3. Shared types

### 3.1 QuiesceProbe

**Role.** Defines how the snapshot agent determines that a target container is in a state safe to checkpoint. Modeled after `corev1.Probe` (familiar shape, reusing `corev1` action types where possible) but evaluated by the snapshot agent — not kubelet — so it can be applied without affecting the pod's serving readiness.

```go
// QuiesceProbe describes how the snapshot agent detects that the
// target containers are ready to be dumped. The same type is referenced
// from SnapshotJob.Spec and (in future versions) Snapshot.Spec.
type QuiesceProbe struct {
    // Action is the probe operation. Exactly one variant in Action must be set.
    // +kubebuilder:validation:Required
    // +kubebuilder:validation:XValidation:rule="[has(self.httpGet),has(self.exec),has(self.tcpSocket),has(self.grpc),has(self.file)].exists_one(x,x)",message="exactly one probe action must be set"
    Action QuiesceProbeAction `json:"action"`

    // PeriodSeconds is the interval between consecutive probe attempts.
    // +optional
    // +kubebuilder:default=1
    // +kubebuilder:validation:Minimum=1
    PeriodSeconds int32 `json:"periodSeconds,omitempty"`

    // TimeoutSeconds is the per-attempt timeout. A probe attempt that
    // does not complete within this many seconds counts as a failure.
    // +optional
    // +kubebuilder:default=1
    // +kubebuilder:validation:Minimum=1
    TimeoutSeconds int32 `json:"timeoutSeconds,omitempty"`

    // SuccessThreshold is the number of consecutive successes required
    // before the probe is considered satisfied and the dump begins.
    // +optional
    // +kubebuilder:default=1
    // +kubebuilder:validation:Minimum=1
    SuccessThreshold int32 `json:"successThreshold,omitempty"`

    // FailureThreshold is the number of consecutive failures after which
    // the agent abandons the probe. Bounded; SnapshotJob.ActiveDeadlineSeconds
    // is the wall-clock hard deadline.
    // +optional
    // +kubebuilder:default=7200
    // +kubebuilder:validation:Minimum=1
    FailureThreshold int32 `json:"failureThreshold,omitempty"`
}

// QuiesceProbeAction is a tagged union of probe actions.
// Exactly one variant must be set.
type QuiesceProbeAction struct {
    // HTTPGet probes an HTTP endpoint on the target container.
    // +optional
    HTTPGet *corev1.HTTPGetAction `json:"httpGet,omitempty"`

    // Exec runs a command inside the target container.
    // +optional
    Exec *corev1.ExecAction `json:"exec,omitempty"`

    // TCPSocket probes a TCP port on the target container.
    // +optional
    TCPSocket *corev1.TCPSocketAction `json:"tcpSocket,omitempty"`

    // GRPC probes via the standard gRPC health protocol.
    // +optional
    GRPC *corev1.GRPCAction `json:"grpc,omitempty"`

    // File probes for the existence of a sentinel file in the target
    // container's filesystem. NVSnapshot-specific (no corev1 equivalent).
    // This is the default mechanism if QuiesceProbe is omitted entirely.
    // +optional
    File *FileAction `json:"file,omitempty"`
}

// FileAction probes for the existence of a sentinel file inside the
// target container's filesystem.
type FileAction struct {
    // Path is the absolute path to the sentinel file inside the container.
    // +kubebuilder:validation:Required
    Path string `json:"path"`
}
```

**Default when `QuiesceProbe` is unset:**

The agent uses the pod's standard `Ready` condition (`status.conditions[type=Ready] == True`) as the quiesce gate. The workload's own readinessProbe doubles as the quiesce signal — no NVSnapshot-specific contract is required from the application beyond the existing K8s readiness probe.

The agent publishes `nvsnapshot.io/quiesce-ready` reflecting the gate it actually used:

- When `QuiesceProbe` is set: `quiesce-ready` reflects the configured probe.
- When `QuiesceProbe` is unset: `quiesce-ready` reflects the pod's `Ready` condition (transitions to True at the same time, with reason `PodReady`).

### 3.2 Pod conditions

NVSnapshot publishes state via custom pod conditions. These are **observability-only** — they do not participate in `pod.status.conditions[type=Ready]` (no readiness gate). The pod's own readinessProbe continues to govern traffic routing independently.

| Condition type | Meaning |
|---|---|
| `nvsnapshot.io/quiesce-ready` | The configured `QuiesceProbe` succeeded — or, when no QuiesceProbe is set, the pod's `Ready` condition transitioned to True. CRIU dump is about to start. The condition's `reason` field distinguishes the two paths (`ProbeSucceeded` vs `PodReady`). |
| `nvsnapshot.io/snapshotted`   | CRIU dump complete; artifact written to PVC |
| `nvsnapshot.io/restored`      | CRIU replay complete on a restore pod |

Consumers observe these via `kubectl wait --for=condition=...` and standard pod-status APIs.

### 3.3 Failure surfacing

A failed snapshot or restore action surfaces in **every related resource** so consumers can observe the failure from whichever object they were watching:

- **Pod**: a `Failed=True` condition with type matching the failed stage:
  - `nvsnapshot.io/snapshotted` with `status=False` + reason `DumpFailed` for snapshot failures.
  - `nvsnapshot.io/restored` with `status=False` + reason `RestoreFailed` for restore failures.
  - The condition's `message` carries the failure detail.
- **`Snapshot.status.conditions[Failed]`** is set to True with reason and message; the `Ready` condition stays False or Unknown.
- **`SnapshotContent.status.conditions[Failed]`** is set to True with reason and message; the `Ready` condition stays False or Unknown.
- **`SnapshotJob.status.conditions[Failed]`** is set to True; `status.phase` transitions to `Failed`.
- **`RestoreSnapshot.status.conditions[Failed]`** is set to True; `status.phase` transitions to `Failed`.

Failures clear on the next successful reconcile (e.g., a retried RestoreSnapshot pointing at a newly-ready source). The pod's `Failed`-status conditions are not auto-cleared — they reflect the historical outcome on that pod.

---

## 4. Restore consumption

Two alternative designs are presented below. v1alpha1 will ship **one** of them — the choice is an open question (see section 8 Q2). Both are documented so the tradeoffs are concrete.

### 4.1 Mode A — annotation-on-pod

A pod opts into restore via an annotation on its PodTemplate. Pods reference a `Snapshot` in their own namespace only — **never** the cluster-scoped `SnapshotContent` directly. This namespace boundary is the primary security control against cross-namespace exfiltration of captured runtime state.

```yaml
metadata:
  annotations:
    nvsnapshot.io/restore-from: my-snap                   # Snapshot in same namespace (required)
    nvsnapshot.io/restore-target-containers: main         # optional override
```

**API contract:**

- The restore annotation is **purely additive**. Pod creation is never blocked by NVSnapshot.
- The referenced Snapshot must be in the same namespace as the pod. There is no annotation for referencing a SnapshotContent directly — `SnapshotContent` is a cluster-scoped implementation-detail object that pods cannot consume.
- If the referenced Snapshot exists and has its `Ready` condition set to True at pod admission, the pod's target containers are shaped for CRIU replay; the pod publishes `nvsnapshot.io/restored` on completion.
- If the Snapshot is missing or not ready at admission, the pod admits **unchanged** — no restore shaping is applied. Callers requiring a guaranteed restore are responsible for gating pod creation on the Snapshot's readiness.

### 4.2 Mode B — RestoreSnapshot CRD

The operator creates a `RestoreSnapshot` (see section 2.4 for the type definition and full API contract — preconditions on `Spec.Source.SnapshotName`, behavior when target pod is missing, immutability, etc.) referencing an existing pod via `Spec.PodRef`. The restore engine shapes that pod's target containers for CRIU replay and reports progress on `RestoreSnapshot.Status`.

---

## 5. Checkpoint flow

End-to-end, high-level view of how a checkpoint artifact is produced.

**Direct `Snapshot` path** (low-level entry):

1. Source pod is already running and satisfies the snapshot contract (control volume, target-container labels, NRI-injected binaries).
2. User creates a `Snapshot` with `spec.source.podRef` pointing at the pod, `spec.quiesceProbe` set (or omitted to fall back on the pod's `Ready` condition), and `spec.storage.pvc.claimName` naming the destination PVC.
3. The node-local agent observes the Snapshot, waits for the quiesce gate, then dumps the target containers via CRIU into the PVC at a producer-assigned `basePath`.
4. The agent publishes `nvsnapshot.io/snapshotted` on the source pod, creates a `SnapshotContent` (cluster-scoped, with `snapshotRef` back-pointer + `runtimeInformation` populated), and updates `Snapshot.status.boundSnapshotContentName`.
5. Both objects transition their `Ready` condition to True. The artifact is now usable for restore.

**`SnapshotJob` path** (ergonomic entry):

1. User creates a `SnapshotJob` with `spec.podTemplate`, `spec.quiesceProbe`, `spec.targetContainers`, and `spec.storage`.
2. SnapshotJob controller launches the pod from the PodTemplate, shaped for the snapshot contract.
3. Controller creates a `Snapshot` against that pod (`spec.source.podRef` set, `spec.quiesceProbe`/`spec.storage` copied through). From here the flow is identical to the direct path.
4. `SnapshotJob.status` mirrors the produced Snapshot — `contentName`, `phase`, and `Conditions[Complete]` flip when the underlying Snapshot becomes Ready.

## 6. Restore flow

End-to-end, high-level view of how a snapshot is replayed into a target pod.

**Mode A — annotation-on-pod:**

1. User (or outer controller — Deployment, StatefulSet, DGD) creates a pod with `nvsnapshot.io/restore-from: <my-snap>` in its `metadata.annotations`.
2. At pod admission, NVSnapshot resolves the Snapshot reference in the same namespace. If the Snapshot's `Ready` condition is True, the target containers are shaped for CRIU replay; otherwise the pod admits unchanged (additive contract).
3. Pod is scheduled by Kubernetes to a node the consumer has chosen — the consumer should steer scheduling toward a node matching `runtimeInformation` (GPU model, driver, etc.) via nodeSelector/affinity.
4. The agent on the scheduled node fetches the bound SnapshotContent and verifies its `runtimeInformation` against the local node's actual runtime. If incompatible, the agent declines: it publishes `nvsnapshot.io/restored=False` with reason `IncompatibleNode` and a descriptive message, and the pod's restored containers are not started.
5. If compatible, the agent locates the artifact via `snapshotHandle` and performs CRIU replay into the shaped containers.
6. Agent publishes `nvsnapshot.io/restored=True` on the pod. The workload's own readinessProbe governs serving-readiness from this point.

**Mode B — `RestoreSnapshot` CRD:**

1. Operator creates a `RestoreSnapshot` with `spec.source.snapshotName` and `spec.podRef` pointing at an already-existing pod in the same namespace.
2. RestoreSnapshot controller validates that the referenced Snapshot has `Ready=True` and the target pod exists; otherwise the RestoreSnapshot remains `Pending`.
3. Controller shapes the target pod's containers for CRIU replay.
4. The agent on the target pod's node verifies the bound SnapshotContent's `runtimeInformation` against the local node's actual runtime. If incompatible, the agent declines: it publishes `nvsnapshot.io/restored=False` with reason `IncompatibleNode`, and the RestoreSnapshot transitions to `Failed` with the same reason.
5. If compatible, the agent performs CRIU replay (same mechanism as Mode A).
6. Pod publishes `nvsnapshot.io/restored=True`; `RestoreSnapshot.status.Conditions[Complete]` flips True.

---

## 7. Storage (PVC) requirements

v1alpha1 supports PVC-backed storage only.

| Requirement | Value |
|---|---|
| Access mode | `ReadWriteMany` recommended for production. Single-node test/dev MAY use `ReadWriteOnce` with appropriate nodeAffinity. |
| StorageClass | Must back onto storage that supports the chosen access mode; high sequential write throughput recommended. |
| PVC ownership | User-supplied, pre-existing. NVSnapshot does not provision PVCs in v1alpha1. |
| Reclaim policy | Artifact retention follows the backing PV's `persistentVolumeReclaimPolicy`. A snapshot persists exactly as long as the PVC/PV it lives on. |
| Sizing | `resident_memory + gpu_memory + overlay_delta`. For an 80GB H100 worker, plan ~80–120 GB per snapshot. |

### Path layout (protocol convention)

```
<basePath>/
  manifest.json      # version, file list, hashes, runtime info
  data/              # artifact payload (driver-specific contents)
```

Layout is fixed protocol. Tools reading the artifact rely on it.

---

## 8. Open questions

### Q1 — Application-facing `snapshotted` / `restored` signals

Workloads need an in-container notification that the snapshot completed, and on restore that the wake-up was a CRIU replay rather than a cold start.

- **Provisional v1alpha1**: reserved-name files at `<controlVolumeMount>/snapshotted` (on source pod post-dump) and `<controlVolumeMount>/restored` (on restore pod post-replay). Documented as alpha-stability — may be replaced or removed.
- **Resolve before v1beta1**: pick between (a) keep file convention, (b) expose the `nvsnapshot.io/snapshotted` and `nvsnapshot.io/restored` pod conditions via Downward API as files, (c) wait for upstream Kubernetes/CRIU standardization.

### Q2 — Which restore mode to ship in v1alpha1

Two alternative restore designs are documented in section 4. v1alpha1 will ship **exactly one** — both are presented so the tradeoffs are concrete, but the API surface should not carry both modes unless we have a clear reason to.

- **Mode A — annotation-on-pod (section 4.1).** Pods opt into restore via an annotation; admission shapes the target containers. No new CRD. Natural fit when an outer controller (Deployment/StatefulSet/DGD) manages the pod's PodTemplate. Cluster-side cost: admission webhook.
- **Mode B — `RestoreSnapshot` CRD (section 4.2 / 2.4).** Operator creates a first-class restore object with its own status. Discoverable via `kubectl get restoresnapshot`. No webhook. Natural fit when restore is driven by an operator that already manages pod lifecycle elsewhere.

**Resolve criteria**: which mode does the primary v1alpha1 consumer (Dynamo) prefer for its restore flow? That decides v1alpha1. The other mode can be added in v1alpha2 if a second consumer needs it.

