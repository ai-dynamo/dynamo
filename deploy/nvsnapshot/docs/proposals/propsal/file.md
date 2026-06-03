
# NVSnapshot — Kubernetes API & Operator Proposal (v1alpha1)

*Container checkpoint/restore (c/r) for active GPU workloads. This proposal merges the structure of the original NVSnapshot operator proposal with the v1alpha1 API surface ("Suggested API"). Restore ships as the annotation-on-pod design (Mode A). Audience: engineering stakeholders.*

---

## 1. Overview

NVSnapshot is a Kubernetes API and operator for checkpoint/restore of **running** GPU workloads. It is modeled on `VolumeSnapshot`, but the subject of a snapshot is a live process tree (with CUDA state), not a static volume.

The user-facing API is small:

| Kind | Scope | Role |
| :--- | :--- | :--- |
| `Snapshot` | **namespaced** | Primary primitive. Identifies a captured artifact; binds to a `SnapshotContent`; consumed by restore. |
| `SnapshotContent` | **cluster-scoped** | Artifact-of-record. Where the dump lives + the runtime environment that produced it. Operator-managed. |
| `SnapshotJob` | **namespaced** | Convenience wrapper over `Snapshot` that also creates the source pod from a PodTemplate. |
| Restore | **namespaced** (annotation on a pod) | A pod opts into restore via `nvsnapshot.io/restore-from: <snapshot>`. No restore CRD in v1alpha1. |

The **data plane** is a privileged, node-local agent (DaemonSet) that drives CRIU + `cuda-checkpoint`. The **control plane** is a controller-manager that reconciles the objects and an admission webhook that shapes restore-target pods. Callers never talk to the agent directly.

> **[ FIGURE 1 — Object relationships ]**
> *Diagram: `SnapshotJob` (namespaced) → produces → `Snapshot` (namespaced) ⇄ bound ⇄ `SnapshotContent` (cluster-scoped). A pod carrying the `nvsnapshot.io/restore-from` annotation references the `Snapshot` (never the `SnapshotContent`).*

---

## 2. Workflow — how you use NVSnapshot

*(Per stakeholder request, the usage walkthrough is up front, before the type definitions.)*

### 2.1 Capture a snapshot

**Path A — `SnapshotJob` (recommended).** You hand NVSnapshot a PodTemplate, a destination PVC, and (optionally) a quiesce probe. The operator runs the pod, waits for the quiesce gate, dumps the target containers, and produces a `Snapshot` + `SnapshotContent`. You never manage the source pod.

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: SnapshotJob
metadata:
  name: llama-worker-warm
  namespace: inference
spec:
  podTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: registry.example.com/inference/worker:snapshot-ready
          resources:
            limits:
              nvidia.com/gpu: "1"
  targetContainers: ["worker"]      # omit to capture all containers
  quiesceProbe:                     # omit to fall back on the pod's Ready condition
    action:
      exec:
        command: ["cat", "/snapshot-control/ready-for-checkpoint"]
  storage:
    pvc:
      claimName: snapshot-pvc       # ReadWriteMany, user-supplied
  activeDeadlineSeconds: 3600
```

Watch it complete and learn the artifact name:

```bash
kubectl -n inference wait snapshotjob/llama-worker-warm \
  --for=condition=Complete --timeout=1h
kubectl -n inference get snapshotjob/llama-worker-warm \
  -o jsonpath='{.status.snapshotName} {.status.contentName}'
```

**Path B — direct `Snapshot` (advanced).** If you already run a pod that satisfies the snapshot contract (control volume, target-container labels, NRI-injected binaries), create a `Snapshot` against it directly. The produced `Snapshot` is identical to one a `SnapshotJob` makes.

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: Snapshot
metadata:
  name: llama-worker-warm
  namespace: inference
spec:
  source:
    podRef:
      name: llama-worker-7f9c       # existing pod, same namespace
      containers: ["worker"]
  quiesceProbe:
    action:
      exec: { command: ["cat", "/snapshot-control/ready-for-checkpoint"] }
  storage:
    pvc: { claimName: snapshot-pvc }
```

### 2.2 Restore a snapshot (Mode A — annotation on the pod)

Add one annotation to the pod template of whatever creates the pod (Deployment / StatefulSet / DGD). At admission, if the referenced `Snapshot` is `Ready`, NVSnapshot shapes the target containers for CRIU replay; the pod publishes `nvsnapshot.io/restored` when replay completes.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: llama-worker, namespace: inference }
spec:
  template:
    metadata:
      annotations:
        nvsnapshot.io/restore-from: llama-worker-warm          # Snapshot, same namespace
        nvsnapshot.io/restore-target-containers: worker
    spec:
      # Steer to a node compatible with the captured runtime (GPU model, driver, ...).
      nodeSelector:
        nvidia.com/gpu.product: H100-SXM
      containers:
        - name: worker
          image: registry.example.com/inference/worker:snapshot-ready
          resources: { limits: { nvidia.com/gpu: "1" } }
```

```bash
kubectl -n inference wait pod -l app=llama-worker \
  --for=condition=nvsnapshot.io/restored --timeout=5m
```

**Key properties you can rely on:**

- A pod references a `Snapshot` **in its own namespace only** — never the cluster-scoped `SnapshotContent`. This namespace boundary is the security control (see §4, §10).
- The annotation is **purely additive**: if the `Snapshot` is missing or not `Ready` at admission, the pod admits **unchanged** (no restore shaping). If you require a guaranteed restore, gate pod creation on the `Snapshot`'s readiness yourself.
- The agent verifies the artifact's `runtimeInformation` against the **actual** node at replay time and declines (`restored=False`, reason `IncompatibleNode`) if incompatible — a final safety net on top of your `nodeSelector`/affinity.

---

## 3. Why not just use built-in Kubernetes checkpointing?

Kubernetes has a built-in container-checkpoint mechanism (kubelet/CRI checkpoint, backed by CRIU). It is the right tool for capturing a CPU process for forensic inspection. It is **not** sufficient for live GPU inference workloads, for three reasons that NVSnapshot exists to close.

### 3.1 No quiesce contract

The built-in mechanism freezes a container at an arbitrary instant. A GPU inference worker cannot be frozen at an arbitrary instant and remain restorable — it must first reach a **checkpoint-safe state**: stop admitting work, drain in-flight requests, release device-side GPU memory (KV cache, activations), and tear down resources CRIU cannot capture (`io_uring` rings, InfiniBand QPs, NCCL P2P/NVLS buffers, external TCP connections to etcd/NATS).

There is no coordination point in the native mechanism for the application to say *"safe to freeze now."* NVSnapshot adds that contract: a `QuiesceProbe` (or, by default, the pod's `Ready` condition) plus a control-volume sentinel (`ready-for-checkpoint`). The workload signals readiness; only then does the agent dump. See §6.3 and §9.

### 3.2 No in-cluster restore

The built-in feature is capture-oriented. The produced archive is intended to be analyzed or restored **outside** Kubernetes (imported as an image and run under a low-level runtime). There is no first-class *"restore this checkpoint into a running pod"* — no native path to re-incarnate a workload in the cluster, on a fresh pod identity, as part of normal scheduling.

NVSnapshot makes restore a native operation (§2.2, §7): annotate a pod, and the node agent replays CRIU into the shaped containers, inheriting the new pod's network namespace and refreshing identity from the Downward API (§9.3). Restore is part of the cluster's normal lifecycle, not a manual out-of-band step.

### 3.3 No GPU/CUDA state

Plain CRIU checkpoints CPU process state. It does **not** capture CUDA contexts, GPU device memory, or GPU device identity, and it cannot re-bind a restored process to a GPU. A native checkpoint of a GPU process is therefore not restorable as a working GPU process.

NVSnapshot integrates `cuda-checkpoint` to lock, dump, and restore CUDA state as part of the same atomic operation, and it **remaps GPU identity**: the artifact records the *source* GPU UUIDs; at restore the agent discovers the *target* node's GPU UUIDs (via the PodResources API / DRA `ResourceClaim`) and feeds CRIU a source→target device map so DMA buffers are remapped transparently. The `SnapshotContent.runtimeInformation` (GPU model, driver, CRIU version, node arch) lets the agent refuse an incompatible target before it does damage.

> *(NVSnapshot additionally provides an artifact-of-record object — `SnapshotContent` — so restore can verify runtime compatibility and so artifacts have a managed lifecycle. The native mechanism has no equivalent.)*

---

## 4. API objects & scope

This section answers two stakeholder questions directly: **what is cluster-scoped vs namespaced**, and **whether multiple namespaces can reference the same `SnapshotContent`.**

### 4.1 The objects at a glance

| Object | Scope | Created by | Read by | References it may hold |
| :--- | :--- | :--- | :--- | :--- |
| `Snapshot` | Namespaced | User, or `SnapshotJob` controller | User, restore paths | A pod **in its own namespace**; binds to one `SnapshotContent` |
| `SnapshotJob` | Namespaced | User | User | A PVC + PodTemplate **in its own namespace** |
| `SnapshotContent` | **Cluster-scoped** | Operator (or admin, for import) | Operator / agent | Back-pointer to one `Snapshot` (may name any namespace); opaque storage handle |
| Restore annotation | Namespaced (on a pod) | User / outer controller | Admission webhook + agent | A `Snapshot` **in the pod's namespace** |

### 4.2 Namespaced vs cluster-scoped — and why

**Everything a tenant touches is namespaced.** Users create `Snapshot`, `SnapshotJob`, and annotate pods — all in their own namespace, all governed by their namespace's RBAC, and all able to reference only **same-namespace** resources (the PVC, the source pod, the `Snapshot` a pod restores from). This is the tenancy boundary.

**`SnapshotContent` is the one cluster-scoped object, and it is an operator-managed implementation detail** — tenants neither create nor consume it directly. It is cluster-scoped for four concrete reasons:

1. **It must outlive the namespaced `Snapshot`.** Under a `Retain` policy the artifact-of-record survives `Snapshot` deletion; a cluster-scoped object is the natural owner of that lifetime.
2. **The artifact isn't owned by one namespace.** The dump physically lives on storage (a PVC/PV) and is described by an opaque handle; the record of "what was produced and where" is a cluster concern.
3. **It carries node/runtime metadata.** `runtimeInformation` (GPU model, driver, CRIU version, node arch) is about cluster hardware, not a tenant's namespace.
4. **It leaves room for re-binding.** A cluster-scoped record can, in the future, be re-bound or imported into another namespace under explicit policy (see §4.3).

Because a cluster-scoped object cannot hold a structured reference to a namespaced PVC, `SnapshotContent` encodes the artifact location as an **opaque handle** (`pvc://<namespace>/<claimName>/<basePath>`) rather than a typed PVC reference (§8).

**The boundary, stated plainly:** tenant-facing API = namespaced (`Snapshot`, `SnapshotJob`, pod annotation). Artifact-of-record = cluster-scoped (`SnapshotContent`), operator-managed. A pod restores only from a `Snapshot` in its own namespace; it can never name a `SnapshotContent`.

> **[ FIGURE 2 — Scope & trust boundary ]**
> *Diagram: a box labeled "namespace: inference" containing `SnapshotJob`, `Snapshot`, and the annotated restore pod; a separate cluster-scoped band containing `SnapshotContent` and the node agent. Arrow from `Snapshot` → `SnapshotContent` labeled "bound (operator-set)". Red line on the namespace boundary labeled "pods cannot cross this to reach SnapshotContent."*

### 4.3 Can multiple namespaces reference the same `SnapshotContent`?

**Short answer: not directly in v1alpha1 — by design — but the underlying artifact can be reused, and cross-namespace reuse is possible only through an explicit, policy-gated import.** Three precise statements:

1. **Binding is 1:1 and namespace-anchored.** A `SnapshotContent` binds to exactly one `Snapshot` via `spec.snapshotRef` (immutable after binding). Restore consumers reference a `Snapshot` in their **own** namespace, never the `SnapshotContent`. So two different namespaces cannot both point a restore at the same `SnapshotContent` through the supported API path.

2. **The artifact bytes are reusable — within a namespace.** Restore does **not** consume the `SnapshotContent`. The same artifact can be replayed into many pods concurrently: any number of pods in the **same** namespace can carry `nvsnapshot.io/restore-from: <snapshot>` pointing at the same `Snapshot`/artifact. Fan-out restore is fully supported; it just happens namespace-locally.

3. **Cross-namespace sharing is a deliberate future extension, gated by policy.** Because `SnapshotContent` is cluster-scoped, the model leaves room to make an existing artifact usable from another namespace — by **importing** it: an admin pre-provisions a `SnapshotContent` (or re-binds one) and a new `Snapshot` is created in namespace B referencing that artifact's handle. v1alpha1 does **not** do this automatically; the namespace boundary on `Snapshot` is the exfiltration control. If namespace B should use an artifact produced in namespace A, it must go through that explicit, gated import — not silent sharing.

So: **one `SnapshotContent` ↔ one `Snapshot` (one namespace) at a time; the artifact can be restored many times within that namespace; cross-namespace reuse requires an explicit pre-provisioned import that v1alpha1 leaves room for but does not enable by default.**

---

## 5. Installation model

NVSnapshot installs **cluster-wide** because the data plane is node-local and privileged: the node agent must run on GPU nodes and perform host-level CRIU/CUDA work against containers scheduled there. A per-namespace install would duplicate privileged DaemonSets and muddy host ownership.

The **API objects remain namespaced** (except `SnapshotContent`). Tenants create `Snapshot`/`SnapshotJob` and annotate pods in their own namespaces; the operator uses RBAC, admission, owner references, UIDs, and the namespace boundary to decide which operations are valid. Internal labels/annotations may drive efficient watches but are never the *authority* for privileged execution.

Installed components:

- **CRDs + RBAC** defining the public API (`Snapshot`, `SnapshotContent`, `SnapshotJob`).
- **Controller-manager** — the policy and orchestration layer; reconciles the objects, drives `SnapshotJob` pods, sets bindings.
- **Admission webhook** — shapes restore-target pods that carry `nvsnapshot.io/restore-from` (required by the Mode A restore design, §7).
- **Privileged node-agent DaemonSet** — the executor; runs CRIU + `cuda-checkpoint` on GPU nodes. Callers never call it directly.
- **Helper binaries/images** — `nsrestore`, CUDA checkpoint tooling, restore-compatible image helpers; CLI/SDK for debugging.

> **[ FIGURE 3 — Components & planes ]**
> *Diagram: control plane (controller-manager + admission webhook + CRDs) over a data plane band of node agents on GPU nodes; arrow "accepted objects → internal pod metadata → agent validates against object/UID/identity."*

---

## 6. Capture — API & flow

Field-level docstrings and `+kubebuilder` validation markers are the canonical source of truth in the `apis/nvsnapshot/v1alpha1` package; trimmed shapes are shown here for orientation.

### 6.1 `Snapshot` (namespaced)

The primary primitive. Identifies *what* was captured (always an existing pod) and how it should be captured.

```go
type SnapshotSpec struct {
    Source       SnapshotSource    // pod being captured; immutable after binding
    QuiesceProbe *QuiesceProbe     // when safe to dump; nil => pod's Ready condition is the gate
    Storage      SnapshotStorage   // PVC destination (artifact dir is producer-assigned)
}
type SnapshotStatus struct {
    BoundSnapshotContentName *string             // set once dump completes + binding verified both ways
    CreationTime             *metav1.Time
    Conditions               []metav1.Condition  // Ready, Failed
}
```

`Source.PodRef` names the pod (same namespace) and optionally narrows to specific `containers`. Consumers MUST verify binding both ways (`Snapshot` ↔ `SnapshotContent`) before treating the object as restorable.

### 6.2 `SnapshotContent` (cluster-scoped)

The artifact-of-record (see §4.2 for why it's cluster-scoped).

```go
type SnapshotContentSpec struct {
    SnapshotRef        SnapshotReference  // back-pointer to bound Snapshot (may span namespaces); immutable
    Source             SnapshotContentSource // opaque handle, e.g. pvc://<ns>/<claim>/<basePath>; immutable
    RuntimeInformation map[string]string  // gpu.model, gpu.driverVersion, node.arch, criu.version, ...; immutable
}
type SnapshotContentStatus struct {
    SnapshotHandle *string        // validated handle
    CreationTime   *metav1.Time
    RestoreSize    *int64         // on-PVC bytes; size restore targets from this
    Conditions     []metav1.Condition  // Ready, Failed
}
```

`runtimeInformation` keys are conventionally dotted but not schema-enforced; consumers tolerate missing keys. The **agent** does the compatibility check at restore time — the API does not enforce it at admission/scheduling.

### 6.3 `SnapshotJob` (namespaced) & the quiesce contract

`SnapshotJob` is a convenience wrapper over `Snapshot` that adds pod creation (analogous to `batch/v1.Job`). It launches a pod from `podTemplate`, waits for quiesce, creates a `Snapshot` against that pod, drives the dump, and produces a `SnapshotContent`. It does **not** own the produced artifacts (deleting the `SnapshotJob` does not delete them); `status.snapshotName` / `status.contentName` name them.

```go
type SnapshotJobSpec struct {
    PodTemplate           corev1.PodTemplateSpec
    QuiesceProbe          *QuiesceProbe   // nil => pod's Ready condition is the gate
    TargetContainers      []string        // empty => all containers
    Storage               SnapshotStorage // PVC only in v1alpha1
    ActiveDeadlineSeconds *int64          // default 3600; wall-clock hard deadline
}
```

**`QuiesceProbe`** is modeled on `corev1.Probe` (HTTPGet / Exec / TCPSocket / GRPC actions, with `periodSeconds`, `timeoutSeconds`, `successThreshold`, `failureThreshold`) but is evaluated by the **agent**, not kubelet — so it never affects the pod's serving readiness. **When omitted, the agent uses the pod's standard `Ready` condition as the quiesce gate** — the workload's own readinessProbe doubles as the quiesce signal, requiring no NVSnapshot-specific contract beyond standard K8s readiness.

**Observability — pod conditions (no readiness gate):**

| Condition | Meaning |
| :--- | :--- |
| `nvsnapshot.io/quiesce-ready` | Configured `QuiesceProbe` succeeded — or, if unset, the pod's `Ready` condition went True (the `reason` field distinguishes `ProbeSucceeded` vs `PodReady`). |
| `nvsnapshot.io/snapshotted` | CRIU dump complete; artifact written to PVC. |
| `nvsnapshot.io/restored` | CRIU replay complete on a restore pod. |

### 6.4 Capture flow

**Direct `Snapshot`:** (1) source pod is running and satisfies the contract; (2) user creates a `Snapshot` (`podRef` + `quiesceProbe`/`Ready` fallback + `storage.pvc`); (3) the node agent waits for the quiesce gate, then dumps via CRIU into the PVC at a producer-assigned `basePath`; (4) the agent publishes `snapshotted`, creates the `SnapshotContent` (with `snapshotRef` + `runtimeInformation`), and sets `Snapshot.status.boundSnapshotContentName`; (5) both objects go `Ready=True`.

**`SnapshotJob`:** the controller launches the pod from `podTemplate`, then creates a `Snapshot` against it; from step (3) the flow is identical. `SnapshotJob.status` mirrors the produced `Snapshot` (`phase`, `contentName`, `Conditions[Complete]`).

> **[ FIGURE 4 — Capture sequence ]**
> *Sequence: user/SnapshotJob → controller (run pod) → agent (quiesce gate → CRIU+CUDA dump → atomic publish) → SnapshotContent created + Snapshot bound → Ready.*

### 6.5 `io_uring` / seccomp & failure surfacing

CRIU cannot checkpoint processes with active `io_uring` state, so capture pods run under a localhost seccomp profile that blocks `io_uring` from process start (another reason `SnapshotJob` is the default for generic workloads).

A failed action surfaces in **every related resource**: the **pod** gets a `Failed=True` condition (`snapshotted`/`restored` with reason `DumpFailed`/`RestoreFailed` and a message); `Snapshot`, `SnapshotContent`, and `SnapshotJob` set their `Failed` conditions / `Failed` phase. Failures clear on the next successful reconcile; pod conditions reflect historical outcome and are not auto-cleared. **Capture failure is `SIGKILL`, not in-place retry** — a CUDA-locked process is unrecoverable in place, so the operator must build a new job/snapshot.

---

## 7. Restore — Mode A (annotation on the pod)

v1alpha1 ships the **annotation-on-pod** restore design. A pod opts in via annotations on its template; admission shapes the target containers. There is **no restore CRD** in v1alpha1.

```
metadata:
  annotations:
    nvsnapshot.io/restore-from: my-snap                 # Snapshot in same namespace
    nvsnapshot.io/restore-target-containers: main
```

**Contract:**

- The annotation is **additive** — pod creation is never blocked by NVSnapshot.
- The referenced `Snapshot` must be **in the pod's namespace**. There is no annotation to reference a `SnapshotContent` directly — it's a cluster-scoped object pods cannot consume.
- If the `Snapshot` exists and is `Ready` at admission, the target containers are shaped for CRIU replay and the pod publishes `nvsnapshot.io/restored` on completion. If it's missing or not ready, the pod admits **unchanged**.

**Flow:** (1) an outer controller (Deployment/StatefulSet/DGD) creates the annotated pod; (2) admission resolves the `Snapshot` in-namespace and shapes the container if `Ready`; (3) K8s schedules to a node the consumer steered toward via `nodeSelector`/affinity; (4) the agent fetches the bound `SnapshotContent` and verifies `runtimeInformation` against the **actual** node — if incompatible it declines (`restored=False`, reason `IncompatibleNode`); (5) if compatible, it locates the artifact via `snapshotHandle` and performs CRIU + CUDA replay (remapping GPU UUIDs); (6) the pod publishes `restored=True` and the workload's own readinessProbe governs serving from there.

> **[ FIGURE 5 — Restore sequence (Mode A) ]**
> *Sequence: outer controller → annotated pod → admission shaping → schedule → agent runtime-compat check → CRIU+CUDA replay → restored=True.*

> **Considered alternative — Mode B (`RestoreSnapshot` CRD).** An operator-driven, status-bearing restore object (`kubectl get restoresnapshot`, no webhook) was considered. It is **deferred**, not shipped in v1alpha1. Revisit if operators need kubectl-discoverable restore objects with first-class status; the annotation model covers the outer-controller case (incl. DGD) today. See §12.

---

## 8. Storage (PVC)

v1alpha1 supports PVC-backed storage only.

| Requirement | Value |
| :--- | :--- |
| Access mode | `ReadWriteMany` (production). `ReadWriteOnce` permitted for single-node test/dev with appropriate `nodeAffinity`. |
| PVC ownership | User-supplied, pre-existing. NVSnapshot does **not** provision PVCs in v1alpha1. |
| Reclaim policy | Artifact retention follows the backing PV's `persistentVolumeReclaimPolicy`. A snapshot persists exactly as long as the PVC/PV it lives on. |

The artifact directory inside the PVC is **producer-assigned** (derived from the producing `Snapshot`/`SnapshotJob` UID), not user-settable. `SnapshotContent` records the location as an opaque handle `pvc://<namespace>/<claimName>/<basePath>`; consumers must not interpret its structure. The PVC's namespace is embedded in the handle because the cluster-scoped `SnapshotContent` cannot carry a typed reference to a namespaced PVC. The storage shape is a tagged union so object-storage / URI backends can be added additively later.

---

## 9. Workload contract

The contract a workload must obey to be snapshottable is intentionally narrow: it does **not** require running as PID 1, handling signals, or exposing an RPC. It requires some boot-time discipline and one polling loop.

### 9.1 Control volume & sentinels

The agent and workload coordinate through a per-pod `emptyDir` mounted at `$NV_SNAPSHOT_CONTROL_DIR` (default `/snapshot-control`). Three zero-byte sentinels:

| File | Writer | Reader | Meaning |
| :--- | :--- | :--- | :--- |
| `ready-for-checkpoint` | workload | quiesce probe | Model loaded, quiesced, GPU released — safe to checkpoint now. |
| `snapshot-complete` | agent | workload | Dump finished; exit cleanly. |
| `restore-complete` | agent | restored workload | You're back; resume. |

*(The product-neutral env name is `NV_SNAPSHOT_CONTROL_DIR`; Dynamo's `DYN_SNAPSHOT_CONTROL_DIR` is supported as a compatibility alias during migration.)*

### 9.2 What the workload does on boot

If the control dir is set: configure transports for c/r compatibility → load the model → quiesce (stop intake, drain, release device memory, park CUDA streams) → `touch ready-for-checkpoint` → poll for `snapshot-complete` (exit 0) or `restore-complete` (wake engine, resume, reload identity).

### 9.3 What must be released before signaling ready, and identity on restore

| Resource | Why | Mechanism |
| :--- | :--- | :--- |
| Device GPU memory (KV cache, activations) | Dump cost grows with allocation | `engine.sleep()` / `release_memory_occupation` |
| InfiniBand QPs | CRIU/cuda-checkpoint can't restore IB QP state | `NCCL_IB_DISABLE=1` before model load |
| External TCP (etcd/NATS) | CRIU dumps with `tcpClose=true`, `skipInFlight=true` | Don't connect to runtime services before snapshot prep |
| `io_uring` rings | CRIU can't checkpoint io_uring | localhost seccomp profile blocks the syscall |
| NCCL P2P/NVLS/RAS/monitoring | cuMem P2P buffers uncheckpointable; timers fire post-restore | `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_RAS_ENABLE=0`, `TORCH_NCCL_ENABLE_MONITORING=0` |

**Identity on restore:** a restored process runs in a **different pod** (new name, possibly new namespace/IP/node/GPU UUIDs). Anything variable across restores must come from a **live** source — the Downward API `/etc/podinfo` volume (and ConfigMap/Secret volumes) — not from env vars read once at boot, which CRIU bakes into the process image. After `restore-complete`, the workload re-reads `/etc/podinfo`, refreshes identity, reconnects etcd/NATS, rebuilds distributed comms, and republishes its discovery record. **Networking and peer re-discovery are the workload's responsibility**; NVSnapshot gives you a process at the right point in its execution, with weights/KV intact and the same rootfs, in a new pod.

---

## 10. API & security model

The public execution plane is **CRD/annotation-driven**; the privileged work stays inside the node agent. The controller translates accepted objects into internal pod metadata, and the agent validates that metadata against the object, repository, owner reference, UID, and pod/container identity before doing anything privileged.

This gives properties annotations-as-authority cannot:

- **RBAC** can permit capture separately from restore.
- **Admission** can prevent cross-tenant source/target/storage references; the **namespace boundary** stops a pod from reaching a `SnapshotContent` or another tenant's `Snapshot`.
- **Status and events** attach to first-class objects.
- The privileged DaemonSet remains the **only** component with host-level access; internal labels/annotations are hints, never triggers.

The stable API deliberately does **not** expose: physical artifact paths as caller inputs, Dynamo-specific podinfo fields, transport env vars, node-agent internal labels/annotations, or raw CRIU flags (only coarse runtime policy).

---

## 11. Dynamo integration

Today Dynamo's auto-checkpoint flow goes through `DynamoCheckpoint`: the `DynamoGraphDeployment` (DGD) controller prepares a checkpoint pod template with Dynamo-specific wiring, then creates a `DynamoCheckpoint` for the checkpoint operator to execute. With NVSnapshot, **that intermediate execution object is replaced by a generic `SnapshotJob`.** DGD keeps all Dynamo-specific policy; NVSnapshot owns the generic checkpoint lifecycle.

### 11.1 Ownership split

| DGD (the caller) keeps | NVSnapshot owns |
| :--- | :--- |
| Checkpoint identity, dedup, `checkpointRef` status | Running the prepared pod; waiting for the quiesce gate |
| GMS / DRA / discovery wiring, transport env | Driving CRIU + CUDA dump; producing `Snapshot` + `SnapshotContent` |
| Downward API `/etc/podinfo`, GMS sidecars | Artifact storage layout, binding, `runtimeInformation` |
| Scheduler/topology policy, NCCL/backend policy | Restore shaping (admission) + agent replay |

### 11.2 Concrete handoff

1. **DGD prepares the PodTemplate** — image, command, resources, GMS sidecars, podinfo, transport env, scheduler constraints. Everything Dynamo-specific is baked in here.
2. **DGD creates a `SnapshotJob`** (in the workload's namespace) with that prepared `podTemplate`, a `storage.pvc`, and `targetContainers`. No `DynamoCheckpoint` execution object is needed.
3. **DGD watches `SnapshotJob.status`** — when `phase: Succeeded` / `Conditions[Complete]=True`, it reads `status.snapshotName` and `status.contentName` and records them in its own `checkpointRef`.
4. **On restore**, DGD stamps the worker pod template it already manages with `nvsnapshot.io/restore-from: <snapshotName>` (Mode A) and steers scheduling toward a compatible node. The agent does the runtime-compat check and CRIU+CUDA replay.

```yaml
# Emitted by the DGD controller once the Dynamo-specific pod template is prepared.
apiVersion: nvsnapshot.io/v1alpha1
kind: SnapshotJob
metadata:
  name: dgd-llama-tp1-ckpt
  namespace: inference
  labels:
    nvidia.com/dynamo-graph-deployment-name: llama-graph
spec:
  podTemplate:
    metadata:
      labels: { nvidia.com/dynamo-component-type: worker }
    spec:
      restartPolicy: Never
      containers:
        - name: main                       # DGD's worker container, fully prepared
          image: registry.example.com/dynamo/vllm-worker:snapshot-ready
          resources: { limits: { nvidia.com/gpu: "1" } }
          # GMS sidecars, podinfo mount, transport env injected by DGD (omitted)
  targetContainers: ["main"]
  quiesceProbe:                              # omit to fall back on the pod's Ready condition
    action:
      exec: { command: ["cat", "/snapshot-control/ready-for-checkpoint"] }
  storage:
    pvc: { claimName: dynamo-snapshot-pvc }
  activeDeadlineSeconds: 3600
```

> **[ FIGURE 6 — Dynamo integration ]**
> *Diagram: DGD controller (owns prep + identity + checkpointRef) → creates `SnapshotJob` → NVSnapshot (run pod, quiesce, dump, bind) → `Snapshot`/`SnapshotContent`. Restore: DGD stamps `restore-from` on the worker pod template.*

---

## 12. Open decisions

1. **Restore mode (resolved for v1alpha1): Mode A (annotation).** Mode B (`RestoreSnapshot` CRD) is deferred; revisit if operators want kubectl-discoverable, status-bearing restore objects.
2. **Application-facing `snapshotted`/`restored` signals.** v1alpha1 uses reserved-name files in the control volume (alpha-stability). Before v1beta1, choose: keep the file convention, expose the pod conditions via Downward API as files, or wait for upstream K8s/CRIU standardization.
3. **Capturing existing pods vs Job-only.** Both are supported: the direct `Snapshot` path captures an already-running pod; `SnapshotJob` is the ergonomic default that manages the source pod for you.
4. **Cross-namespace `SnapshotContent` import.** Deferred and policy-gated (see §4.3). Not enabled by default in v1alpha1.
5. **Control-dir env name.** `NV_SNAPSHOT_CONTROL_DIR` is the stable name; `DYN_SNAPSHOT_CONTROL_DIR` is a Dynamo compatibility alias during migration.
6. **Manifest schema.** Promote the internal artifact manifest to a versioned public type before external restore tools depend on it.
