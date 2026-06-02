# DisaggregatedSet Operator Helm Chart

Kubernetes operator for managing disaggregated inference deployments using LeaderWorkerSets.

## Prerequisites

- Kubernetes 1.27+
- Helm 3.8+
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) installed in the cluster

## Installation

```bash
# Install LeaderWorkerSet first (if not already installed)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.8.0/manifests.yaml

# Install the DisaggregatedSet operator
helm install disaggregatedset charts/disaggregatedset
```

## Uninstallation

```bash
helm uninstall disaggregatedset
```

## Configuration

The following table lists configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `controllerManager.replicas` | Number of operator replicas | `1` |
| `controllerManager.manager.image.repository` | Operator image repository | `ghcr.io/kubernetes-sigs/disaggregatedset` |
| `controllerManager.manager.image.tag` | Operator image tag | `""` (uses appVersion) |
| `controllerManager.manager.resources.limits.cpu` | CPU limit | `500m` |
| `controllerManager.manager.resources.limits.memory` | Memory limit | `128Mi` |
| `controllerManager.manager.resources.requests.cpu` | CPU request | `10m` |
| `controllerManager.manager.resources.requests.memory` | Memory request | `64Mi` |

### Example: Custom values

```bash
helm install disaggregatedset charts/disaggregatedset \
  --set controllerManager.manager.image.tag=v0.1.0 \
  --set controllerManager.replicas=2
```

## CRDs

The chart installs the DisaggregatedSet CRD. CRDs are placed in the `crds/` directory and are automatically installed by Helm.

To upgrade CRDs manually (Helm does not upgrade CRDs automatically):

```bash
kubectl apply --server-side -f charts/disaggregatedset/crds/
```
