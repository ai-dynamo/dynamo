# Dynamo on Azure AKS

Supported Helm values for AKS deployments.

**Full guide:** [docs/kubernetes/cloud-providers/aks/aks.md](../../../docs/kubernetes/cloud-providers/aks/aks.md)

**Related guides:**

- [Storage for model caching](../../../docs/kubernetes/cloud-providers/aks/storage.md)
- [Spot VMs](../../../docs/kubernetes/cloud-providers/aks/spot-vms.md)
- [RDMA / InfiniBand](../../../docs/kubernetes/cloud-providers/aks/rdma-infiniband.md)
- [Azure Lustre CSI Driver](../../../docs/kubernetes/cloud-providers/aks/azure-lustre-csi.md)

## Contents

| Path | Description |
|------|-------------|
| `values-aks-spot.yaml` | Helm values with Spot VM tolerations for the Dynamo platform chart |

## Working Directory

Helm commands that reference `values-aks-spot.yaml` assume you are in this directory:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo/examples/deployments/AKS
```
