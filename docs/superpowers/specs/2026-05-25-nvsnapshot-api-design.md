# NVSnapshot API Design

**Date:** 2026-05-25
**Status:** Draft (v1alpha1 API design)
**Scope:** New open-source Kubernetes API for container checkpoint/restore (c/r). Dynamo is the first consumer. **This document describes the API surface only.** Controller behavior, reconciler logic, agent internals, and migration plans live in a separate implementation document.

---

## 1. Summary

NVSnapshot is a Kubernetes API for container checkpoint/restore, inspired by `VolumeSnapshot` but adapted to the realities of c/r: the snapshot target is an active workload, not a static volume.

Three concerns are separated into distinct CRDs:

- **The artifact-of-record** (`SnapshotContent`, cluster-scoped) — what was captured and how to find it.
- **The user-facing binding** (`Snapshot`, namespaced) — declarative reference to a SnapshotContent for restore consumption.
- **The active operation** (`SnapshotJob`, namespaced) — the workload-execution that produces a snapshot.

The API is application-agnostic. Application-specific concepts (LLM identity, framework parameters, etc.) belong in higher-level CRDs that construct NVSnapshot resources underneath.

## 2. Goals & non-goals

**Goals (v1alpha1):**

- Cleanly separate snapshot intent, artifact, and operation across CRDs.
- Support fresh-pod-launch-then-dump flow as the primary v1alpha1 use case.
- Provide a flexible quiesce protocol (HTTP/exec/TCP/gRPC/file).
- Keep producer UX simple — one CRD (`SnapshotJob`) for the full lifecycle.
- Zero application-specific concepts in the API surface.

**Non-goals (v1alpha1):**

- Live-pod snapshot of arbitrary user-launched pods (reserved for v1alpha2).
- Object-storage / URI artifact sources (PVC only).
- Multi-driver support / `SnapshotClass`.
- Restore policy enforcement (no platform validation, no nodeAffinity injection).
- Cross-cluster portability automation.

## 3. API group & versioning

- **Group:** `nvsnapshot.io`
- **Version:** `v1alpha1`
- **Storage version:** v1alpha1 is the storage version and must be marked as such from the first release so future versions can introduce a conversion webhook without breaking stored objects.

## 4. CRDs

Four CRDs total. No `SnapshotClass` in v1alpha1 — nothing varies per-class today.

### 4.1 `Snapshot` (namespaced)

User-facing binding. Equivalent to `VolumeSnapshot`. The `source` field is **descriptive** — it identifies what this snapshot is OF, not how to make it.

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: Snapshot
metadata: { namespace: default, name: my-snap }
spec:
  source:
    # Exactly one of:
    podRef:
      name: my-source-pod
      uid: <pod-uid>
      containers: [main]          # optional; default = all containers
    snapshotContentName: my-content-<uid>  # pre-provisioned import
  quiesceProbe: { ... }            # see §6
  identityKey: ""                  # optional; opaque dedup key
status:
  phase: Pending | Bound | Failed | Deleting
  boundSnapshotContentName: my-content-<uid>
  identityHash: ""
  jobRef: { name: my-snap-job }    # set when produced by a SnapshotJob
  readyAt: <ts>
  message: ""
```

**Field semantics:**

- `source` is one-of (`podRef` xor `snapshotContentName`). Immutable after binding.
- `source.podRef.containers` narrows the snapshot scope; default = all containers in the pod.
- The unit of snapshot is always container-level.

**Validation:**

- Immutability of `source` enforced via CEL `XValidation` (server-side); webhook is not the source of truth.

### 4.2 `SnapshotContent` (cluster-scoped)

The artifact-of-record. Equivalent to `VolumeSnapshotContent`. Cluster-scoped so it survives Snapshot deletion (per `deletionPolicy=Retain`) and can be re-bound across namespaces in the future.

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: SnapshotContent
metadata: { name: my-content-<uid> }
spec:
  driver: criu-cuda
  deletionPolicy: Retain | Delete
  snapshotRef:                     # back-pointer to bound Snapshot
    namespace: default
    name: my-snap
    uid: <uid>
  source:
    type: PVC                      # discriminator; PVC only in v1alpha1
    pvc:
      namespace: default
      claimName: my-snap-pvc
      basePath: /nvsnapshot/<uid>  # controller-assigned, derived from SnapshotJob UID
  platform:                        # immutable artifact metadata (see §7)
    node: { arch, kernelVersion, containerRuntime }
    gpu:  { model, driverVersion, cudaVersion, computeCapability, migProfile, count }
    criuVersion: "3.19"
    seccompProfile: nvsnapshot-agent.json
status:
  phase: Provisioning | Ready | Failed | Released
  sizeBytes: 0
  createdAt: <ts>
  message: ""
```

**Field semantics:**

- `source.type` is a discriminator. v1alpha1 supports `PVC` only; unknown values are rejected. Future variants (e.g., `S3`, `URI`) are additive.
- `source.pvc.basePath` is **controller-assigned**, not user-supplied. Derived from the producing SnapshotJob's UID to guarantee global uniqueness on the PVC. This prevents `(pvc, basePath)` collisions that would otherwise risk silent data loss on `deletionPolicy=Delete`.
- `spec.platform` is descriptive metadata; not enforced at restore in v1alpha1. Populated by the producer on dynamic creation; user-supplied on pre-provisioned import.
- `deletionPolicy` lives here, mirroring `VolumeSnapshotContent`.

**Validation:**

- `source` and `platform` immutable after binding (CEL `XValidation`).
- `source.type=PVC` requires `source.pvc` to be set; other variants of `source.*` must be empty.

### 4.3 `SnapshotJob` (namespaced)

A one-shot producer (analogous to `batch/v1.Job`). Materializes a pod from a PodTemplate, drives the dump, and **produces** a `Snapshot` + `SnapshotContent` on success. The produced objects are **not owned by** the SnapshotJob — they live independently after creation. Deleting a SnapshotJob has no effect on the artifacts it produced; their lifecycle is governed entirely by `deletionPolicy` on the SnapshotContent (and by the Snapshot ↔ SnapshotContent binding).

```yaml
apiVersion: nvsnapshot.io/v1alpha1
kind: SnapshotJob
metadata: { namespace: default, name: my-snap-job }
spec:
  podTemplate: { ... corev1.PodTemplateSpec ... }
  quiesceProbe: { ... }            # see §6
  targetContainers: [main]
  storage:
    type: PVC                      # discriminator; PVC only in v1alpha1
    pvc: { claimName: my-snap-pvc }   # basePath controller-assigned; not user-settable
  activeDeadlineSeconds: 3600
  deletionPolicy: Delete           # propagates to produced SnapshotContent
  identityKey: ""
status:
  phase: Pending | Running | Succeeded | Failed
  contentName: my-content-<uid>    # the SnapshotContent to use for restore
  ready: true | false
  identityHash: ""
```

**Field semantics:**

- **One-pane-of-glass status.** Producer users read `SnapshotJob.status` only; the produced `Snapshot` is materialized so restore paths can reference it uniformly but the producer doesn't need to inspect it.
- `storage.pvc.basePath` is **not user-settable**; assigned by the producer from the SnapshotJob UID.
- `quiesceProbe` shares the Go type defined in `Snapshot.spec.quiesceProbe` (see §6).

### 4.4 `RestoreSnapshot` (namespaced, reserved kind — not implemented in v1alpha1)

Reserved for the operator-driven restore path (Mode B in §5). Kind name is registered to prevent reuse; schema is deferred. Restore in v1alpha1 is annotation-driven (Mode A).

## 5. Restore consumption

Two modes are designed. v1alpha1 ships only Mode A.

### 5.1 Mode A — annotation-on-pod (v1alpha1)

A pod opts into restore by carrying one of the `nvsnapshot.io/restore-from*` annotations on its PodTemplate. At pod admission, NVSnapshot shapes the target container(s) for CRIU replay using the referenced SnapshotContent.

```yaml
metadata:
  annotations:
    nvsnapshot.io/restore-from: my-snap                  # Snapshot in same namespace
    # — or —
    nvsnapshot.io/restore-from-content: my-content-<uid> # SnapshotContent (cluster-scoped)
    nvsnapshot.io/restore-target-containers: main        # optional override
```

**API contract:**

- The restore annotation is **purely additive**. Pod creation is never blocked by NVSnapshot.
- If the referenced target exists and is in a ready state (`Snapshot.status.phase == Bound` or `SnapshotContent.status.phase == Ready`) at pod admission, the pod's target containers are shaped for CRIU replay; the pod publishes `nvsnapshot.io/restored` on completion (§6.2).
- If the reference is missing or not Ready at admission, the pod admits **unchanged** — no restore shaping is applied and the container starts fresh. No condition is published. Callers requiring a guaranteed restore (e.g., higher-level operators) are responsible for gating pod creation on the referenced object's readiness.

Admission-controller configuration (failure policy, selector scoping, certificate handling) is an implementation/deployment concern, not an API contract.

### 5.2 Mode B — `RestoreSnapshot` CRD (reserved, v1alpha2+)

First-class operator-driven restore. Discoverable via `kubectl get restoresnapshot`. Schema reserved (§4.4); not implemented in v1alpha1.

## 6. Quiesce protocol

The application signals "safe to dump" via a probe defined in `Snapshot.spec.quiesceProbe` (or `SnapshotJob.spec.quiesceProbe` for the SnapshotJob path).

### 6.1 `QuiesceProbe` type

A shared NVSnapshot type, defined once and embedded in both `Snapshot` and `SnapshotJob`. Not `corev1.Probe`, because evaluation is NVSnapshot-side and the type adds a `file` variant.

```yaml
quiesceProbe:
  action:                          # discriminator
    type: HTTPGet | Exec | TCPSocket | GRPC | File
    httpGet:   { path: /quiesce, port: 9090 }
    exec:      { command: ["test", "-f", "/tmp/ready"] }
    tcpSocket: { port: 9090 }
    grpc:      { port: 9091, service: nvsnapshot.Quiesce }
    file:      { path: /var/run/nvsnapshot/ready-for-checkpoint }
  periodSeconds: 1
  timeoutSeconds: 1
  successThreshold: 1
  failureThreshold: 7200           # default: 2h at periodSeconds=1; bounded, not int32-max
```

**Validation:**

- `action.type` is required; exactly one matching sub-field must be set (CEL `XValidation`).
- `failureThreshold` defaults to 7200 (≈2h at 1s period) — bounded. `activeDeadlineSeconds` on the producing SnapshotJob acts as the hard deadline.

**Default when omitted** (preserves backward-compatibility with the existing sentinel-file pattern):

```yaml
quiesceProbe:
  action:
    type: File
    file: { path: /var/run/nvsnapshot/ready-for-checkpoint }
  periodSeconds: 1
```

### 6.2 Pod conditions

NVSnapshot publishes state via custom pod conditions. **No readiness gates** — these conditions do not participate in `pod.status.conditions[type=Ready]`, which is left entirely to the workload's own probes. This is essential for live-pod snapshot semantics where the workload's Ready=False (drained, no traffic) coincides with quiesce-ready=True (dump-safe).

| Condition type | Meaning |
|---|---|
| `nvsnapshot.io/quiesce-ready` | `quiesceProbe` succeeded; CRIU dump about to start |
| `nvsnapshot.io/snapshotted`   | CRIU dump complete, artifact written to PVC |
| `nvsnapshot.io/restored`      | CRIU replay complete on a restore pod |

Consumers observe these via standard `kubectl wait --for=condition=…` and `kubectl get pod -o yaml`.

## 7. Storage (PVC) requirements

v1alpha1 supports PVC-backed storage only.

| Requirement | Value |
|---|---|
| Access mode | `ReadWriteMany` recommended for production. Single-node test/dev clusters MAY use `ReadWriteOnce` if pod scheduling is constrained via nodeAffinity — validation warns, does not reject. |
| StorageClass | Must back onto storage that supports the chosen access mode. High sequential write throughput recommended. |
| PVC ownership | User-supplied, pre-existing. NVSnapshot does not provision PVCs in v1alpha1. |
| Sharing | Multiple SnapshotJobs may share one PVC. `basePath` is producer-assigned from the SnapshotJob UID, guaranteeing uniqueness. |
| Sizing | Approx `resident_memory + gpu_memory + overlay_delta`. For an 80GB H100 LLM worker, plan ~80–120 GB per snapshot. |

### 7.1 Path layout (protocol convention)

```
<basePath>/
  manifest.json      # version, file list, hashes, platform metadata
  criu/              # CRIU image files
  cuda/              # CUDA device-memory blobs
  oci/               # OCI bundle metadata, overlay deltas
```

Layout is fixed protocol. Tools reading the artifact rely on it.

### 7.2 Lifecycle

- `deletionPolicy: Delete` → `<basePath>` is removed on SnapshotContent deletion; PVC is left alone.
- `deletionPolicy: Retain` → SnapshotContent moves to `Released` on Snapshot deletion; artifact remains.
- Manual `<basePath>` deletion while bound → SnapshotContent flips to `Failed`.
- Orphan GC is manual via `snapshotctl gc` in v1alpha1.

## 8. Application contract

What NVSnapshot requires from the workload running inside the snapshotted container(s).

### 8.1 Cooperation points

- **Quiesce probe** (§6): expose one variant (HTTP/exec/TCP/gRPC) that returns success when safe to dump, or write the default sentinel file.
- **Forks before quiesce**: late-forked subprocesses are not captured.
- **Drain external connections** before signaling quiesce (or be reconnect-tolerant).

### 8.2 Image & filesystem

- Use **digest-pinned** container images for snapshot/restore — CRIU dumps memory state, not binaries.
- Keep large mutable state on PVC mounts, not on the container overlay (overlay writes inflate the artifact).
- Use the NVSnapshot seccomp profile (`nvsnapshot-agent.json`) via `securityContext.seccompProfile.localhostProfile`.

### 8.3 GPU constraints

- Restore node's NVIDIA driver version ≥ source node's driver version.
- Restore node's GPU model **must match** source node's (`platform.gpu.model`).
- MIG profile must match exactly if MIG was in use.

### 8.4 What CRIU captures

- **Captured**: process tree, anonymous and file-backed memory, GPU device memory (via cuda-checkpoint plugin), open FDs, mount namespace, FUTEX state, container overlay writes.
- **Not captured**: mounted PVC/configmap/secret contents (reflect restore-time state), external network state.

### 8.5 Checklist for application owners

- [ ] Workload signals quiesce at a stable, repeatable point.
- [ ] All subprocesses forked before quiesce.
- [ ] External connections drained or tolerable to reconnect.
- [ ] Container image is digest-pinned.
- [ ] Pod uses the NVSnapshot seccomp profile.
- [ ] Large mutable state on PVC, not container overlay.

## 9. Cluster prerequisites

- Kubernetes ≥ 1.28.
- Container runtime: containerd ≥ 1.7 or CRI-O ≥ 1.27, with NRI enabled (NVSnapshot ships an NRI plugin to provision required binaries into target containers; mechanism details are an implementation concern, but the prerequisite is part of the API contract).
- RWX-capable StorageClass for production deployments.
- NVIDIA GPU Operator (for node labels / driver discovery).
- cert-manager (for the Mode A mutating webhook).

## 10. Out of scope for v1alpha1

- Live-pod snapshot of user-launched pods (`Snapshot.spec.source.podRef` direct entry).
- Multi-driver support / `SnapshotClass`.
- Object-storage / URI source for `SnapshotContent` (the `source.type` discriminator reserves the namespace).
- Restore Mode B (`RestoreSnapshot` CRD); kind reserved.
- Auto-enforcement of `SnapshotContent.spec.platform` (nodeAffinity injection, version gates).
- Cross-cluster portability automation.
- Auto-GC of orphan PVC paths.
- Dynamic PVC provisioning (per-snapshot PVCs from StorageClass).

## 11. Open questions

### Q1 — Application-facing `snapshotted` / `restored` signals

Workloads (especially live-pod scenarios) need an in-container notification that the snapshot completed, and on restore that the wake-up was a CRIU replay rather than a cold start.

- **Provisional v1alpha1**: reserved-name files at `<controlVolumeMount>/snapshotted` (on source post-dump) and `<controlVolumeMount>/restored` (on restore post-replay). Marked alpha-stability — may be replaced or removed.
- **Resolve before v1beta1**: pick between (a) keep file convention, (b) expose `nvsnapshot.io/snapshotted` / `restored` pod conditions via Downward API as files, (c) wait for upstream K8s/CRIU standardization.
