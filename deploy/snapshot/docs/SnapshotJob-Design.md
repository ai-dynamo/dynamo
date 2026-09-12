# SnapshotJob — Design Document

**Status:** Draft  
**Last updated:** 2026-08-09

---

## 1. Overview

SnapshotJob is a **checkpoint-only** CRD. Its single responsibility is connecting two things that today must be done separately:

1. **Pod creation** — run a workload that is ready to be checkpointed
2. **PodSnapshot creation** — request a checkpoint of that running pod

Today a user must create a pod, wait for it to be running, manually create a PodSnapshot pointing at it, and coordinate cleanup. SnapshotJob makes these one atomic operation: the user provides a PodTemplate; SnapshotJob creates the pod (via a `batch/v1 Job`) and the PodSnapshot together, waits for the node agent to complete the CRIU + CUDA dump, and produces a ready-to-use `PodSnapshot` artifact.

SnapshotJob is a CRD. The `PodSnapshot` it produces is identical to one created manually — the artifact type is unchanged, only the lifecycle management is automated.

---

## 2. Motivation & Goals

### Problem

Today the capture flow is driven entirely by `DynamoCheckpoint`, a Dynamo-specific CRD that mixes generic snapshot mechanics (pod shaping, PodSnapshot creation, agent triggering) with Dynamo-specific policy (checkpoint identity hashing, GMS wiring, deduplication). This makes the snapshot infrastructure impossible to use outside of Dynamo.

### Goals

- User creates a single object (`SnapshotJob`) to capture a GPU workload — no manual pod or PodSnapshot management
- The produced `PodSnapshot` is usable for restore via `nvidia.com/restore-from` annotation
- SnapshotJob has zero awareness of any specific consumer (Dynamo, GMS, or otherwise)
- **Dynamo only creates a SnapshotJob** — no raw Job, Pod, or PodSnapshot
- **Other consumers can use SnapshotJob** without any Dynamo component installed

### Definition of Done

- [ ] Dynamo's DGD controller creates a SnapshotJob instead of DynamoCheckpoint
- [ ] A non-Dynamo user can capture a GPU workload using only SnapshotJob + PodSnapshot (+ SnapshotClass once implemented)
- [ ] The produced PodSnapshot is usable for restore via `nvidia.com/restore-from` annotation

## 3. Non-Goals

- **Restore** — SnapshotJob is capture-only. Restore is a separate flow via pod annotations.
- **Multi-snapshot from one job** — one SnapshotJob produces exactly one PodSnapshot.
- **PVC provisioning** — users supply a pre-existing PVC.
- **Retry** — `backoffLimit: 0` in v1alpha1. No retry, not exposed in spec.
- **Cross-namespace artifact sharing** — v1alpha1 constraint, not SnapshotJob-specific.
- **arm64 support** — `cuda-checkpoint` is x86-only until CUDA 610.
- **SnapshotClass and per-job storage** — deferred to next phase. See `SnapshotClass-Design.md`.

---

## 4. API Design

### Type reference

| Type                  | Status       | Kind                                                             |
|-----------------------|--------------|------------------------------------------------------------------|
| `SnapshotJob`         | **New**      | CRD — namespaced                                                 |
| `SnapshotJobSpec`     | **New**      | Embedded struct — part of `SnapshotJob` CRD                      |
| `SnapshotJobStatus`   | **New**      | Embedded struct — part of `SnapshotJob` CRD                      |
| `PodSnapshotTemplate` | **New**      | Embedded struct — part of `SnapshotJob` CRD (see Open Questions) |
| `PodSnapshot`         | **Existing** | CRD — namespaced (defined in Snapshot API)                       |
| `PodSnapshotContent`  | **Existing** | CRD — cluster-scoped (defined in Snapshot API)                   |

### Controller reference

| Controller               | Status       | Reconciles        | Notes                                                      |
|--------------------------|--------------|-------------------|------------------------------------------------------------|
| `SnapshotJob controller` | **New**      | `SnapshotJob` CRD | Creates batch/v1 Job, Pod, PodSnapshot; manages conditions |
| `PodSnapshot controller` | **Existing** | `PodSnapshot` CRD | Unchanged — creates and binds PodSnapshotContent           |

### 4.1 Spec

```go
// NEW — embedded struct, part of SnapshotJob CRD
type SnapshotJobSpec struct {
    // PodTemplate defines the workload to run and capture.
    // The controller injects snapshot-contract requirements before creating
    // the batch/v1 Job. The caller is responsible for all other pod content:
    // GPU resources, image, sidecars, DRA claims, etc.
    PodTemplate corev1.PodTemplateSpec `json:"podTemplate"`

    // ActiveDeadlineSeconds bounds the total time allowed for pod scheduling,
    // quiesce, and dump. Defaults to 3600.
    // +optional
    // +kubebuilder:default=3600
    // +kubebuilder:validation:Minimum=1
    ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

    // PodSnapshotTemplate defines the properties of the PodSnapshot produced
    // by this job. Maps directly to PodSnapshot spec — the controller fills in
    // spec.source from the pod it creates. Separating this from PodTemplate
    // makes the boundary explicit: PodTemplate is the pod execution side;
    // PodSnapshotTemplate is the snapshot capture side.
    PodSnapshotTemplate PodSnapshotTemplate `json:"podSnapshotTemplate"`
}

// NEW — embedded struct, part of SnapshotJob CRD (see Open Questions: flat vs nested)
// PodSnapshotTemplate mirrors PodSnapshot spec fields that the user controls.
// The controller fills in spec.source (the pod reference) automatically.
type PodSnapshotTemplate struct {
    // TargetContainers names the container(s) to checkpoint with CRIU.
    // The pod may contain any number of additional containers (helpers,
    // sidecars, etc.) — this field controls only the CRIU dump target.
    //
    // v1alpha1: exactly one target is required. The plural form and nested
    // placement here is intentional: future versions will support per-container
    // config (e.g. per-container quiesceProbe) by extending this list to objects.
    //
    // Defaults to ["main"]. Each entry must be a valid DNS label
    // (1–63 chars, ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$).
    // +optional
    // +kubebuilder:default={"main"}
    // +kubebuilder:validation:MinItems=1
    // +kubebuilder:validation:MaxItems=1
    // +kubebuilder:validation:items:MinLength=1
    // +kubebuilder:validation:items:MaxLength=63
    // +kubebuilder:validation:items:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
    TargetContainers []string `json:"targetContainers,omitempty"`

    // SnapshotClassName and Storage are deferred to next phase — see SnapshotClass-Design.md.
    // QuiesceProbe is also deferred — v1alpha1 quiesce gate is pod Ready=True.
}
```

### 4.2 Status

```go
// NEW — embedded struct, part of SnapshotJob CRD
type SnapshotJobStatus struct {
    // PodSnapshotName is the name of the PodSnapshot produced by this job.
    // Set when the PodSnapshot is created. Distinguishes "never created"
    // (empty) from "created but missing" (set, not found).
    PodSnapshotName string `json:"podSnapshotName,omitempty"`

    // StartedAt is when the source pod became Ready.
    StartedAt *metav1.Time `json:"startedAt,omitempty"`

    // CompletedAt is when a terminal condition (Completed or Failed) was set.
    CompletedAt *metav1.Time `json:"completedAt,omitempty"`

    // Conditions reflect the current state of the SnapshotJob.
    // Types: Running, Captured, Completed, Failed.
    Conditions []metav1.Condition `json:"conditions,omitempty"`
}
```

**Condition types:**

| Type        | Status=True meaning                                                      | Reasons (True)                                                     | Reasons (False)                                                                                   |
|-------------|--------------------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `Running`   | Pod is running and ready (readiness probe passed)                        | `PodReady`                                                         | `PodPending` (initial / pod not yet ready)                                                        |
| `Captured`  | CRIU dump of target container complete — `PodSnapshot Ready=True`        | `DumpCompleted`                                                    | `DumpInProgress`, `DumpFailed`, `PodSnapshotFailed`                                               |
| `Completed` | All pod containers exited 0 — `batch/v1 Job Complete=True`               | `JobCompleted`                                                     | `WaitingForPodCompletion` (Captured=True; helpers still running), `JobFailed`, `DeadlineExceeded` |
| `Failed`    | Terminal failure — batch/v1 Job is **not** deleted (preserved for debug) | `DumpFailed`, `JobFailed`, `DeadlineExceeded`, `PodSnapshotFailed` | —                                                                                                 |

> **Pod scheduling failures** are not surfaced directly — they appear as `DeadlineExceeded` on the batch/v1 Job after `activeDeadlineSeconds` expires. SnapshotJob observes the Job, not the Pod, for failure status.

### 4.3 Example

```yaml
apiVersion: nvidia.com/v1alpha1
kind: SnapshotJob
metadata:
  name: warm-worker
  namespace: inference
spec:
  podTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: registry.example.com/worker:latest
          resources:
            limits:
              nvidia.com/gpu: "1"
  activeDeadlineSeconds: 3600
  podSnapshotTemplate:
    targetContainers: ["worker"]
```

### 4.4 Produced status (after success)

```yaml
status:
  podSnapshotName: warm-worker-snapshot
  startedAt: "2026-08-09T10:00:05Z"
  completedAt: "2026-08-09T10:02:30Z"
  conditions:
    - type: Running
      status: "True"
      reason: PodReady
      lastTransitionTime: "2026-08-09T10:00:05Z"
    - type: Captured
      status: "True"
      reason: DumpCompleted
      lastTransitionTime: "2026-08-09T10:02:15Z"
    - type: Completed
      status: "True"
      reason: JobCompleted
      lastTransitionTime: "2026-08-09T10:02:30Z"
    - type: Failed
      status: "False"
      lastTransitionTime: "2026-08-09T10:00:00Z"
```

---

## 5. Capture Flow

```
User creates SnapshotJob (spec.activeDeadlineSeconds: 3600)
        │                                            Running=False, Captured=False, Completed=False, Failed=False
        ▼
SnapshotJob controller creates batch/v1 Job
        │  ◄─── activeDeadlineSeconds applied to batch/v1 Job ─────────────────────────────────────────┐
        ▼        bounds: scheduling + quiesce + dump + all containers exit                              │
Kubernetes creates Pod                                                                                  │
        │                                                                                               │
        ▼  pod Create event fires                                                                       │
        │                                            Running=False, reason: PodPending                  │
SnapshotJob controller creates PodSnapshot immediately                                                  │
        │                                            Captured=False, reason: DumpInProgress             │
        ▼                                                                                               │
PodSnapshot controller creates PodSnapshotContent                                                      │
        │                                                                                               │
        ▼  pod becomes Ready=True (readiness probe passes)                                              │
        │                                            Running=True, reason: PodReady                     │
        │                                                                                               │
Node agent watches PodSnapshotContent → waits for pod Ready=True → CRIU dump                           │
        │                                                                                               │
        ├──── CRIU dump fails ──────────────────────────────────────────► Captured=False                │
        │     PodSnapshot Failed=True                   Failed=True, reason: DumpFailed                 │
        │     batch/v1 Job NOT deleted (preserved for debug)                                            │
        ▼                                                                                               │
Captured=True, reason: DumpCompleted  (PodSnapshot Ready=True)                                         │
        │                                            Completed=False, reason: WaitingForPodCompletion   │
        │  helpers still running (e.g. gms-saver writing weights)                                      │
        ├──── helper container exits non-zero ──────────────────────────► Completed=False               │
        │     batch/v1 Job Failed=True                  Failed=True, reason: JobFailed                  │
        │     batch/v1 Job NOT deleted (preserved for debug)                                            │
        ▼  all containers exit 0                                                                        │
batch/v1 Job Complete=True ◄──── deadline exceeded anywhere ────────────────────────────────────────────┘
        │                                                   Failed=True, reason: DeadlineExceeded
        │                                                   batch/v1 Job NOT deleted
        ▼
Completed=True, reason: JobCompleted → delete batch/v1 Job → Pod deleted (cascade)
```

**Why batch/v1 Job, not a bare pod:**
- Kubernetes tracks completion and failure natively — including `DeadlineExceeded` when pods fail to schedule
- `backoffLimit` field enables retry in a future version without API changes
- batch/v1 Job owns the Pod; SnapshotJob owns the batch/v1 Job

**Container exit 0 requirement:** every container in the pod (target and helpers) must exit with code 0 for the batch/v1 Job to be marked `Complete`. A non-zero exit from any container causes `Failed=True, reason: JobFailed`. Callers are responsible for ensuring helper containers exit cleanly after completing their work.

**Cleanup policy:**
- `Completed=True` → controller deletes the batch/v1 Job (cascades to Pod)
- `Failed=True` → batch/v1 Job is **NOT deleted** — preserved for user inspection and debugging
- PodSnapshot and PodSnapshotContent are never deleted by SnapshotJob — they outlive it

**Failure is terminal.** No retry in v1alpha1. The caller creates a new SnapshotJob if retry is needed.

---

## 6. Controller Lifecycle

**Two-stage completion gate:**

`Captured=True` and `Completed=True` are set independently:

- `Captured=True, reason: DumpCompleted` — set when `PodSnapshot Ready=True` (CRIU dump of target container done)
- `Completed=True, reason: JobCompleted` — set when `batch/v1 Job Complete=True` (all containers exited 0)

Between `Captured=True` and `Completed=True`, the condition `Completed=False, reason: WaitingForPodCompletion` signals that the CRIU artifact is ready but helpers are still running. SnapshotJob does not clean up until `Completed=True` — deleting the Job prematurely would kill still-running helper containers mid-work.

**Example (Dynamo+GMS):** `gms-saver` writes GPU weight artifacts concurrently with the CRIU dump. It exits after writing. If SnapshotJob cleaned up at `Captured=True`, gms-saver would be killed before the weight artifact is complete. `batch/v1 Job Complete=True` is the natural barrier ensuring all pod work is done before cleanup.

---

## 7. Relationship to Other APIs

```
SnapshotJob
    │ creates (ownerRef)
    ▼
batch/v1 Job
    │ creates (ownerRef)       injects readinessProbe:
    ▼                          cat $SNAPSHOT_CONTROL_DIR/ready-for-snapshot
Pod (target container) ──────────────────────────────────────► Node agent
    │                                                               │
    │ workload writes sentinel file                                 │
    ▼                                                               │
kubelet evaluates readinessProbe                                    │
    │ probe passes                                                  │
    ▼                                                               │
pod Ready=True ─────────────────────────────────────────────────── ┤ quiesce gate
                                                                    │ dumps target container
    │ SnapshotJob creates                                           │
    ▼                                                               ▼
PodSnapshot ◄──────────────── bound 1:1 ──────────── PodSnapshotContent
    │
    │ referenced by
    ▼
restore Pod (nvidia.com/restore-from annotation)
```

- **SnapshotJob does not own PodSnapshot or PodSnapshotContent.** Artifacts outlive the SnapshotJob.
- **SnapshotJob owns the batch/v1 Job** (which owns the Pod). Deleting SnapshotJob cascades to both.
- **PodSnapshot is the consumer-facing artifact.** Restore paths reference it by name from `status.podSnapshotName`.

---

## 8. Quiesce Gate — v1alpha1

The v1alpha1 quiesce gate is **pod `Ready=True`**. The node agent starts the CRIU dump once the source pod's Ready condition is True.

**How pod `Ready=True` is achieved:** the SnapshotJob controller injects a readiness probe on the target container (`cat $SNAPSHOT_CONTROL_DIR/ready-for-snapshot`). The workload writes the sentinel file when it is ready to be checkpointed. Kubelet evaluates the probe — when the file exists, pod becomes `Ready=True`.

This is **identical to what DynamoCheckpoint does today** via `NewCheckpointJob`. There is zero behavioral change for workloads migrating from DynamoCheckpoint to SnapshotJob v1alpha1.

**Evolution with `quiesceProbe` (next phase):** the injected readiness probe stays as Layer 1. The node agent additionally evaluates the custom `quiesceProbe` (Layer 2) before starting the dump. Layer 1 is never removed.

**`quiesceProbe` is not implemented in v1alpha1. It is planned for the next phase.**

---

## 9. Security Model

- **Namespace boundary** — SnapshotJob, its batch/v1 Job, and the produced PodSnapshot are all namespaced. A pod can reference a PodSnapshot only in its own namespace for restore.
- **ownerReference cascade** — SnapshotJob owns the batch/v1 Job (which owns the Pod). Deleting SnapshotJob cascades to both. PodSnapshot/PodSnapshotContent must be deleted independently.
- **RBAC separation** — the right to create a SnapshotJob (capture) is separate from the right to create pods with `nvidia.com/restore-from` (restore).
- **Node agent is the only privileged component** — controllers never talk to it directly.

---

## 10. Observability

| Object        | Signal                                                         | Meaning                                                                                  |
|---------------|----------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `SnapshotJob` | `conditions[Running]=False, reason: PodPending`                | Pod not yet ready (pending, image pull, init containers)                                 |
| `SnapshotJob` | `conditions[Running]=True, reason: PodReady`                   | Pod ready; CRIU dump in progress                                                         |
| `SnapshotJob` | `conditions[Captured]=False, reason: DumpInProgress`           | PodSnapshot created; CRIU dump running                                                   |
| `SnapshotJob` | `conditions[Captured]=True, reason: DumpCompleted`             | CRIU artifact ready; waiting for helper containers to finish                             |
| `SnapshotJob` | `conditions[Captured]=False, reason: DumpFailed`               | Dump failed; see message                                                                 |
| `SnapshotJob` | `conditions[Completed]=False, reason: WaitingForPodCompletion` | Captured=True; helpers still running (e.g. writing artifacts to PVC)                     |
| `SnapshotJob` | `conditions[Completed]=True, reason: JobCompleted`             | All containers exited 0; batch/v1 Job deleted; artifact fully ready                      |
| `SnapshotJob` | `conditions[Failed]=True, reason: DumpFailed`                  | CRIU dump failed; batch/v1 Job preserved for debug                                       |
| `SnapshotJob` | `conditions[Failed]=True, reason: JobFailed`                   | Helper container exited non-zero; batch/v1 Job preserved for debug                       |
| `SnapshotJob` | `conditions[Failed]=True, reason: DeadlineExceeded`            | activeDeadlineSeconds exceeded (includes pod scheduling timeout); batch/v1 Job preserved |
| `SnapshotJob` | `status.podSnapshotName`                                       | Name of the produced PodSnapshot                                                         |
| `Pod`         | `nvidia.com/quiesce-ready = True`                              | Quiesce gate passed; dump starting                                                       |
| `Pod`         | `nvidia.com/captured = True`                                   | CRIU dump complete                                                                       |
| `PodSnapshot` | `conditions[Ready] = True`                                     | Artifact verified; usable for restore                                                    |
| `PodSnapshot` | `conditions[Failed] = True`                                    | Capture failed                                                                           |
| restore `Pod` | `nvidia.com/restored = True`                                   | CRIU replay complete                                                                     |
| restore `Pod` | `nvidia.com/restored = False, reason=IncompatibleNode`         | GPU/driver/CRIU mismatch                                                                 |

**Failure propagation:**

```
SnapshotJob    conditions[Failed]=True, reason=DumpFailed, message="..."
  └─ batch/v1 Job   status.conditions[Failed]=True
       └─ Pod        nvidia.com/captured=False, reason=DumpFailed, message="..."
PodSnapshot          conditions[Failed]=True, message="..."
```

---

## 11. Error Handling

Every failure sets `Failed=True` on the SnapshotJob and deletes the batch/v1 Job. Failure is terminal — no retry in v1alpha1.

**Controller behavior on failure:**
- Sets `Failed=True` with the appropriate reason on the SnapshotJob condition
- Records `completedAt`
- Deletes the batch/v1 Job (cascades to Pod)
- PodSnapshot and PodSnapshotContent are NOT deleted — they remain for inspection

**Special case — dual completion gate:** if the CRIU dump succeeds (`PodSnapshot Ready=True`) but a helper container in the pod fails afterward, `batch/v1 Job Complete=True` never fires. The SnapshotJob sets `Failed=True, reason: JobFailed` even though the CRIU artifact exists. This is intentional — a partially-complete checkpoint (e.g. missing GMS weight artifact) is not usable for restore.

**Failure propagation across objects:**
```
SnapshotJob    conditions[Failed]=True, reason=DumpFailed, message="..."
  └─ batch/v1 Job   status.conditions[Failed]=True
       └─ Pod        nvidia.com/captured=False, reason=DumpFailed
PodSnapshot          conditions[Failed]=True, message="..."
```

Failures surface on every related object — callers can observe from whichever object they watch.

---

## 12. Future Work

| Topic                             | Notes                                                                                   |
|-----------------------------------|-----------------------------------------------------------------------------------------|
| SnapshotClass and per-job storage | Deferred — see `SnapshotClass-Design.md`                                                |
| `quiesceProbe`                    | Custom quiesce gate evaluated by the node agent. v1alpha1 uses pod `Ready=True` only.   |
| Retry (`backoffLimit`)            | v1alpha1: `backoffLimit: 0`, not exposed. Configurable retry planned for next version.  |
| Multi-target containers           | v1alpha1: exactly one CRIU target. Infrastructure supports multiple via subPath mounts. |
| PVC provisioning                  | Users supply pre-existing PVC. Dynamic provisioning deferred.                           |
| Cross-namespace artifact sharing  | PodSnapshotContent is cluster-scoped in anticipation, but not enabled in v1alpha1.      |
| Restore flow                      | Handled via pod annotations. A `RestoreJob` CRD may be added in a future version.       |
| arm64 support                     | Blocked on `cuda-checkpoint` arm64 support (CUDA 610+).                                 |
