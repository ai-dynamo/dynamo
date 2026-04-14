# Dynamo Snapshot Helm Chart

> Experimental feature. `snapshot-agent` runs as a privileged DaemonSet to
> perform CRIU checkpoint and restore operations.

This chart installs the namespace-scoped snapshot infrastructure used by Dynamo:

- `snapshot-agent` DaemonSet on eligible GPU nodes
- `snapshot-pvc`, or wiring to an existing PVC
- namespace-scoped RBAC
- the seccomp profile CRIU needs

Install the chart in each namespace where you want checkpoint and restore.

## Prerequisites

- Kubernetes cluster with x86_64 GPU nodes
- NVIDIA driver 580.xx or newer
- containerd runtime
- Dynamo Platform already installed with `dynamo-operator.checkpoint.enabled=true`
- a cluster where a privileged DaemonSet with `hostPID`, `hostIPC`, and `hostNetwork` is acceptable

Cross-node restore requires shared `ReadWriteMany` storage. The chart defaults to
that mode.

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
| `storage.type` | Snapshot-owned storage backend | `pvc` |
| `storage.pvc.create` | Create `snapshot-pvc` instead of using an existing PVC | `true` |
| `storage.pvc.name` | PVC mounted by the snapshot-agent | `snapshot-pvc` |
| `storage.pvc.size` | Requested PVC size | `1Ti` |
| `storage.pvc.storageClass` | Storage class name | `""` |
| `storage.pvc.accessMode` | Access mode for the checkpoint PVC | `ReadWriteMany` |
| `storage.pvc.basePath` | Mount path inside the snapshot-agent pod | `/checkpoints` |
| `daemonset.image.repository` | Snapshot-agent image repository | `nvcr.io/nvidia/ai-dynamo/snapshot-agent` |
| `daemonset.image.tag` | Snapshot-agent image tag | `1.0.0` |
| `daemonset.imagePullSecrets` | Image pull secrets for the agent | `[{name: ngc-secret}]` |

Reserved `s3` and `oci` values remain chart-owned placeholders for future
snapshot backends, but only `pvc` is implemented today.

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
