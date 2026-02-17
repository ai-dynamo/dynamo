# RBAC Configuration for Dynamo Operator

Optional RBAC configurations for the Dynamo Kubernetes Operator.

## namespace-operator-gpu-discovery.yaml

Enables automatic GPU hardware discovery for namespace-scoped operators.

**Quick Start:**

```bash
# Apply ClusterRole
kubectl apply -f namespace-operator-gpu-discovery.yaml

# Bind to your operator (replace namespace/service-account)
kubectl create clusterrolebinding dynamo-operator-gpu-discovery \
  --clusterrole=dynamo-namespace-operator-gpu-discovery \
  --serviceaccount=dynamo-system:dynamo-operator-controller-manager
```

**Grants:** Read-only access (`get`, `list`, `watch`) to nodes for GPU label discovery.

**Alternative:** Manually configure hardware in your DGDR:

```yaml
hardware:
  numGpusPerNode: 8
  gpuModel: "H100-SXM5-80GB"
  gpuVramMib: 81920
```

**Remove:**

```bash
kubectl delete clusterrolebinding dynamo-operator-gpu-discovery
kubectl delete clusterrole dynamo-namespace-operator-gpu-discovery
```

**Docs:** [Installation Guide](../../../docs/pages/kubernetes/installation-guide.md#gpu-discovery-for-namespace-scoped-operators-optional) | [Issue #6257](https://github.com/ai-dynamo/dynamo/issues/6257)
