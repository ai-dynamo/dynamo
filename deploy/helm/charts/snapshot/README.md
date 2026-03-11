# Dynamo Snapshot Helm Chart

> ⚠️ **Experimental Feature**: Dynamo Snapshot is currently in beta/preview. The DaemonSet runs in privileged mode to perform CRIU checkpoint and restore operations.

This chart installs the namespace-scoped checkpoint/restore infrastructure used by Dynamo:

- `snapshot-agent` DaemonSet on GPU nodes
- `snapshot-pvc` checkpoint storage, or wiring to an existing PVC
- namespace-scoped RBAC
- the seccomp profile required by CRIU

Snapshot storage is namespace-local. Install this chart in every namespace where you want checkpoint and restore.

## Prerequisites

- Kubernetes 1.21+
- x86_64 GPU nodes
- NVIDIA driver 580.xx or newer
- containerd runtime
- a cluster where a privileged DaemonSet with `hostPID`, `hostIPC`, and `hostNetwork` is acceptable
- Dynamo Platform already installed, with operator checkpointing enabled

The platform/operator configuration must point at the same checkpoint storage that this chart installs:

```yaml
dynamo-operator:
  checkpoint:
    enabled: true
    storage:
      type: pvc
      pvc:
        pvcName: snapshot-pvc
        basePath: /checkpoints
```

Cross-node restore requires a shared `ReadWriteMany` storage class. The chart defaults to `storage.pvc.accessMode=ReadWriteMany`.

For better restore times, use a fast `ReadWriteMany` StorageClass for the checkpoint PVC.

## Minimal Install

This is the smallest Helm install that creates the checkpoint PVC and the DaemonSet:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

If your cluster does not use a default storage class, also set `storage.pvc.storageClass`.

Keep `storage.pvc.accessMode=ReadWriteMany` for this chart layout. The DaemonSet mounts the same PVC on each eligible node, so a shared `ReadWriteOnce` claim only works when the agent runs on one node.

If you already have a PVC, keep the chart in "use existing PVC" mode:

Do not set `storage.pvc.create=true` when reusing an existing checkpoint PVC.

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

## Important Values

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `storage.pvc.create` | Create `snapshot-pvc` instead of using an existing PVC | `true` |
| `storage.pvc.name` | PVC name used by the agent and by the operator config | `snapshot-pvc` |
| `storage.pvc.size` | Requested PVC size | `1Ti` |
| `storage.pvc.storageClass` | Storage class name | `""` |
| `storage.pvc.accessMode` | Access mode for the checkpoint PVC | `ReadWriteMany` |
| `storage.pvc.basePath` | Checkpoint root inside the PVC | `/checkpoints` |
| `daemonset.image.repository` | Snapshot agent image repository | `nvcr.io/nvidia/ai-dynamo/snapshot-agent` |
| `daemonset.image.tag` | Snapshot agent image tag | `1.0.0` |
| `daemonset.imagePullSecrets` | Image pull secrets for the agent | `[{name: ngc-secret}]` |

See [values.yaml](./values.yaml) for the complete configuration surface.

## End To End

Once the chart is installed, use the checkpointing guide to deploy a snapshot-capable `DynamoGraphDeployment`, wait for the checkpoint to become ready, and then scale the worker to verify restore:

- [Checkpointing](../../../../docs/kubernetes/snapshot.md)

## Uninstall

```bash
helm uninstall snapshot -n ${NAMESPACE}
```

The chart does not remove checkpoint data automatically. Delete the PVC yourself if you want to remove stored checkpoints:

```bash
kubectl delete pvc snapshot-pvc -n ${NAMESPACE}
```

## Troubleshooting

If `snapshot-agent` does not schedule:

```bash
kubectl get nodes -l nvidia.com/gpu.present=true
kubectl describe daemonset snapshot-agent -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/name=snapshot --all-containers
```

If checkpoint creation never becomes ready, verify all three pieces line up:

- the operator has `dynamo-operator.checkpoint.enabled=true`
- the operator PVC name and base path match the snapshot chart values
- the workload uses a snapshot-capable worker image and command
