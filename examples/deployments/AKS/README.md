# Dynamo on Azure AKS

Supported Helm values for AKS deployments.

**Full guide:** [docs/kubernetes/cloud-providers/aks/aks.md](../../../docs/fern/kubernetes/cloud-providers/aks/aks.mdx)

**Related guides:**

- [Storage for model caching](../../../docs/fern/pages/kubernetes/installation/model-storage/aks-storage.md)
- [Spot VMs](../../../docs/fern/kubernetes/cloud-providers/aks/spot-vms.mdx)
- [RDMA / InfiniBand](../../../docs/fern/kubernetes/cloud-providers/aks/rdma-infiniband.mdx)
- [Azure Lustre CSI Driver](../../../docs/fern/kubernetes/cloud-providers/aks/azure-lustre-csi.mdx)

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
