# Dynamo Snapshot Helm Chart

> Experimental feature. `snapshot-agent` runs as a privileged DaemonSet to
> perform CRIU checkpoint and restore operations.

This chart installs the namespace-scoped snapshot infrastructure used by Dynamo:

- `snapshot-agent` DaemonSet on eligible GPU nodes
- `snapshot-pvc`, or wiring to an existing PVC
- namespace-scoped RBAC
- the seccomp profile CRIU needs

By default, install the chart in each namespace where you want checkpoint and
restore; in that mode the chart can create or reuse that namespace's PVC.

Alternatively, install one cluster-scoped `snapshot-agent` in an infrastructure
namespace by setting `storage.accessMode=podMount` and
`rbac.namespaceRestricted=false`. In that mode the DaemonSet does not mount the
checkpoint PVC directly. Instead, the Dynamo operator mounts the namespace-local
checkpoint PVC into checkpoint/restore workload pods, and the agent reaches that
PVC through the target pod's mount namespace. The operator can create those
namespace-local workload PVCs, or it can require that they already exist.

## Prerequisites

- Kubernetes cluster with x86_64 GPU nodes
- NVIDIA driver 580.xx or newer
- **containerd** or **CRI-O** (chart defaults to containerd; see below for CRI-O / OpenShift)
- Dynamo Platform already installed with `dynamo-operator.checkpoint.enabled=true`
- a cluster where a privileged DaemonSet with `hostPID`, `hostIPC`, and `hostNetwork` is acceptable

In the default `agentMount` mode, the snapshot-agent DaemonSet mounts the
checkpoint PVC directly. On a multi-node GPU cluster that means agent pods on
multiple nodes may mount the same PVC, so the PVC generally needs
`ReadWriteMany`. The chart defaults to that mode.

`podMount` removes that agent-side RWX requirement. The snapshot-agent does not
mount the PVC directly; only the active checkpoint/restore workload pod mounts
it, and the agent reaches it through that pod's mount namespace. That allows
suitable `ReadWriteOnce` storage classes for sequential checkpoint/restore
workflows, as long as the backend can attach the volume to the node running that
workload pod.

Because `podMount` reaches storage through `/host/proc/<pid>/root`, the target
container must still be alive and visible through host proc when the agent
starts checkpoint or restore. A container restart, exited placeholder, runtime
PID lookup failure, or node security policy that blocks host proc traversal will
fail or defer that attempt until reconciliation sees a fresh container.

## CRI-O and OpenShift

For CRI-O nodes set `runtime.type=crio`. Only set `runtime.socketPath` if the CRI
socket is not the default for that type (see `values.yaml`). On OpenShift, set
`openshift.enabled=true` so the chart emits the extra RBAC and pod annotations
the agent needs. Example:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace "${NAMESPACE}" --create-namespace \
  --set storage.pvc.create=true \
  --set runtime.type=crio \
  --set openshift.enabled=true
```

## Minimal install

Create the checkpoint PVC and the agent:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

If your cluster does not use a default storage class, also set
`storage.pvc.storageClass`.

Reuse an existing PVC instead:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=false \
  --set storage.pvc.name=my-snapshot-pvc
```

## Verify

```bash
kubectl get pvc snapshot-pvc -n ${NAMESPACE}
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=snapshot -o wide
```

## Important values

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `storage.type` | Storage backend: `pvc` (shared volume) or `s3` (S3-compatible object store) | `pvc` |
| `storage.accessMode` | `agentMount` for namespace-local agent PVC mount, or `podMount` for a single cluster agent that accesses workload-mounted PVCs. `s3` requires `agentMount` | `agentMount` |
| `storage.pvc.create` | Create `snapshot-pvc` instead of using an existing PVC in `agentMount` mode | `true` |
| `storage.pvc.name` | Checkpoint PVC name. Mounted by the agent in `agentMount` mode, or by workload pods in `podMount` mode | `snapshot-pvc` |
| `storage.pvc.size` | Requested PVC size | `1Ti` |
| `storage.pvc.storageClass` | Storage class name | `""` |
| `storage.pvc.accessMode` | Access mode for the checkpoint PVC. `ReadWriteMany` is safest; `ReadWriteOnce` can be used with `podMount` for sequential checkpoint/restore on suitable storage backends | `ReadWriteMany` |
| `storage.pvc.basePath` | Mount path for checkpoint storage: inside the `snapshot-agent` pod in `agentMount`, or inside checkpoint/restore workload pods in `podMount` | `/checkpoints` |
| `storage.s3.endpoint` | S3 endpoint host[:port] (when `storage.type=s3`) | `""` |
| `storage.s3.bucket` | Existing bucket holding checkpoint artifacts | `""` |
| `storage.s3.basePath` | Agent-local staging dir (ephemeral `emptyDir`). Must match the operator's `checkpoint.storage.s3.basePath` | `/var/lib/dynamo/checkpoints` |
| `storage.s3.credentialsSecretRef` | Secret providing `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, projected via `envFrom` | `""` |
| `storage.s3.forcePathStyle` | Use path-style addressing; default is virtual-hosted (bucket DNS) style | `false` |
| `storage.s3.useSSL` / `region` / `prefix` / `concurrency` | TLS toggle, region label, optional key prefix, parallel transfers | `true` / `""` / `""` / `8` |
| `storage.s3.staging.sizeLimit` | Cap on the agent staging `emptyDir` (e.g. `200Gi`). Empty = node default | `""` |
| `storage.s3.staging.medium` | Staging `emptyDir` medium: `""` (disk) or `Memory` (tmpfs) | `""` |
| `storage.s3.staging.hostPath` | Override staging with a node host path (e.g. local NVMe) for large artifacts | `""` |
| `storage.s3.staging.pvcName` | Override staging with an existing PVC | `""` |
| `seccomp.deploy` | Deploy the CRIU seccomp profile ConfigMap and init container. Use this field name; `seccomp.enabled` is not a chart value | `true` |
| `daemonset.image.repository` | Snapshot-agent image repository | `nvcr.io/nvidia/ai-dynamo/snapshot-agent` |
| `daemonset.image.tag` | Snapshot-agent image tag | `1.0.0` |
| `daemonset.imagePullSecrets` | Image pull secrets for the agent | `[{name: ngc-secret}]` |
| `runtime.type` | CRI backend: `containerd` or `crio` | `containerd` |
| `runtime.socketPath` | CRI socket (empty = default for `runtime.type`) | `""` |
| `openshift.enabled` | OpenShift RBAC / SCC-related chart pieces | `false` |

### S3 object storage

Set `storage.type=s3` to store checkpoint artifacts in any S3-compatible object
store instead of a PVC. The agent stages each checkpoint on its own filesystem
(`storage.s3.basePath`, an `emptyDir`), uploads the artifact tree to the bucket,
and downloads it back before restore — so no shared `ReadWriteMany` PVC is
needed. This requires `storage.accessMode=agentMount`; credentials come from
`storage.s3.credentialsSecretRef` (projected via `envFrom`, never in the
ConfigMap). Configure the operator with a matching `checkpoint.storage.s3.basePath`.
See [`examples/s3`](../../../snapshot/examples/s3/) for ready-to-edit manifests.

```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

kubectl -n ${NAMESPACE} create secret generic snapshot-s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...

helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} --create-namespace \
  --set storage.type=s3 \
  --set storage.s3.endpoint=s3.example.com \
  --set storage.s3.bucket=checkpoints \
  --set storage.s3.credentialsSecretRef=snapshot-s3-credentials
```

The `oci` value remains a chart-owned placeholder for a future backend.

When using `storage.accessMode=podMount`, configure the Dynamo operator with
the same workload PVC name and mount path. The snapshot chart value is
`storage.pvc.name`; the operator config field is `pvcName` because it is a
separate API/config object. Use `create: true` if the operator
should create the namespace-local PVC when a checkpoint or restore workload is
reconciled; omit it or set it to `false` to require an existing PVC:

```yaml
dynamo-operator:
  checkpoint:
    enabled: true
    storage:
      type: pvc
      pvc:
        pvcName: snapshot-pvc
        basePath: /checkpoints
        create: true
        size: 1Ti
        storageClassName: ""
        accessMode: ReadWriteMany
```

For clusters without an affordable or available RWX storage class, `podMount`
can be paired with `accessMode: ReadWriteOnce` because the DaemonSet no longer
mounts the PVC from every GPU node. This is intended for flows where checkpoint
and restore pods use the PVC one at a time. Keep `ReadWriteMany` for concurrent
restores, multi-node simultaneous access, or backends that cannot reattach an
RWO volume to the node selected for restore.

See [values.yaml](./values.yaml) for the full configuration surface.

## Next steps

Once the chart is installed, use the snapshot guide to create a checkpoint or
exercise the lower-level `snapshotctl` flow:

- [Snapshot guide](../../../../docs/kubernetes/snapshot.md)

## Uninstall

```bash
helm uninstall snapshot -n ${NAMESPACE}
```

The chart does not delete checkpoint data automatically. Remove the PVC
yourself if you want to clear stored checkpoints:

```bash
kubectl delete pvc snapshot-pvc -n ${NAMESPACE}
```
