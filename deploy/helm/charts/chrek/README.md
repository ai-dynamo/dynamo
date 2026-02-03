# Chrek Helm Chart

> ⚠️ **Experimental Feature**: ChReK is currently in **beta/preview**. It requires privileged mode for restore operations, which may not be suitable for all production environments. See [Prerequisites](#prerequisites) for security considerations.

This Helm chart deploys the checkpoint/restore infrastructure for NVIDIA Dynamo, including:
- Persistent Volume Claim (PVC) for checkpoint storage
- DaemonSet running the CRIU checkpoint agent
- RBAC resources (ServiceAccount, Role, RoleBinding)
- Seccomp profile for blocking io_uring syscalls

**Note:**
- Each namespace gets its own isolated checkpoint infrastructure with namespace-scoped RBAC
- **Currently only supports vLLM backend** (SGLang and TensorRT-LLM support planned)

## Prerequisites

⚠️ **Security Warning**: ChReK restore operations require **privileged mode**, which grants containers elevated host access. This may violate security policies in production environments. Only deploy in environments where privileged containers are acceptable.

- Kubernetes 1.21+
- GPU nodes with NVIDIA runtime (`nvidia` runtime class)
- CRIU support in the container runtime (containerd with CRIU plugin)
- NVIDIA Dynamo operator installed (cluster-wide or namespace-scoped)
- RWX (ReadWriteMany) storage class for multi-node deployments
- **Security clearance for privileged pods** (required for restore operations)

## Installation

> **Note:** The ChReK Helm chart is not yet published to a public Helm repository. For now, you must build and deploy from source.

### Building from Source

```bash
# Set environment
export NAMESPACE=my-team  # Your target namespace
export DOCKER_SERVER=your-registry.com/  # Your container registry
export IMAGE_TAG=latest

# Build ChReK agent image
cd deploy/chrek
docker build --target agent -t $DOCKER_SERVER/chrek-agent:$IMAGE_TAG .
docker push $DOCKER_SERVER/chrek-agent:$IMAGE_TAG
cd -

# Install ChReK chart with custom image
helm install chrek ./deploy/helm/charts/chrek/ \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set daemonset.image.repository=${DOCKER_SERVER}/chrek-agent \
  --set daemonset.image.tag=${IMAGE_TAG} \
  --set daemonset.imagePullSecrets[0].name=your-registry-secret
```

## Configuration

See `values.yaml` for all configuration options.

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `storage.type` | Storage type: `pvc` (only supported), `s3` and `oci` planned | `pvc` |
| `storage.pvc.create` | Create a new PVC | `true` |
| `storage.pvc.name` | PVC name (must match operator config) | `chrek-pvc` |
| `storage.pvc.size` | PVC size | `100Gi` |
| `storage.pvc.storageClass` | Storage class name | `""` (default) |
| `storage.signalHostPath` | Host path for signal files | `/var/lib/chrek/signals` |
| `daemonset.image.repository` | DaemonSet image repository | `nvidia/chrek-agent` |
| `daemonset.nodeSelector` | Node selector for GPU nodes | `nvidia.com/gpu.present: "true"` |
| `daemonset.runtimeClassName` | Runtime class for GPU access | `nvidia` |
| `daemonset.criu.timeout` | CRIU timeout in seconds | `"21600"` (6 hours) |
| `daemonset.criu.ghostLimit` | CRIU ghost file size limit | `"512MB"` |
| `rbac.namespaceRestricted` | Use namespace-scoped RBAC | `true` |

## Usage

After installing this chart, enable checkpointing in your DynamoGraphDeployment:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-model
  namespace: my-team
spec:
  services:
    worker:
      checkpoint:
        enabled: true
        mode: auto
        identity:
          model: Qwen/Qwen3-0.6B
          backendFramework: vllm
```

## Multi-Namespace Deployment

To enable checkpointing in multiple namespaces, install this chart in each namespace:

```bash
# Namespace A
helm install chrek nvidia/chrek -n team-a

# Namespace B
helm install chrek nvidia/chrek -n team-b
```

Each namespace will have its own isolated checkpoint storage.

## Verification

```bash
# Check PVC
kubectl get pvc chrek-pvc -n my-team

# Check DaemonSet
kubectl get daemonset -n my-team

# Check DaemonSet pods are running
kubectl get pods -n my-team -l app.kubernetes.io/name=chrek
```

## Uninstallation

```bash
helm uninstall chrek -n my-team
```

**Note:** This will NOT delete the PVC by default. To delete the PVC:

```bash
kubectl delete pvc chrek-pvc -n my-team
```

## Troubleshooting

### DaemonSet pods not starting

Check if GPU nodes have the correct labels and runtime class:

```bash
kubectl get nodes -l nvidia.com/gpu.present=true
kubectl describe node <node-name> | grep -A 5 "Runtime Class"
```

If nodes don't have the `nvidia.com/gpu.present` label, you can add it:

```bash
kubectl label node <node-name> nvidia.com/gpu.present=true
```

### Checkpoint job fails

Check DaemonSet logs:

```bash
kubectl logs -n my-team -l app.kubernetes.io/name=chrek
```

### PVC not mounting

Check PVC status and events:

```bash
kubectl describe pvc chrek-pvc -n my-team
```

Ensure your storage class supports `ReadWriteMany` access mode for multi-node deployments.

## Related Documentation

- [ChReK Overview](../../../../docs/kubernetes/chrek/README.md) - ChReK architecture and use cases
- [ChReK with Dynamo Platform](../../../../docs/kubernetes/chrek/dynamo.md) - Integration guide
- [ChReK Standalone Usage](../../../../docs/kubernetes/chrek/standalone.md) - Use ChReK without Dynamo Platform

## License

Apache License 2.0

