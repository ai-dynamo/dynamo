# Dynamo on AKS

This guide covers deploying Dynamo and running LLM inference on Azure Kubernetes Service (AKS). You'll learn how to set up an AKS cluster with GPU nodes, install required components, and deploy your first model.

## Prerequisites

Before you begin, ensure you have:

- An active Azure subscription
- Sufficient Azure quota for GPU VMs
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- [Helm](https://helm.sh/docs/intro/install/) installed

## Step 1: Create AKS Cluster with GPU Nodes

If you don't have an AKS cluster yet, create one using the [Azure CLI](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-cli), [Azure PowerShell](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-powershell), or the [Azure portal](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal).

Ensure your AKS cluster has a node pool with GPU-enabled nodes. Follow the [Use GPUs for compute-intensive workloads on Azure Kubernetes Service (AKS)](https://learn.microsoft.com/en-us/azure/aks/use-nvidia-gpu?tabs=add-ubuntu-gpu-node-pool#skip-gpu-driver-installation) guide to create a GPU-enabled node pool.

**Important:** It is recommended to **skip the GPU driver installation** during node pool creation, as the NVIDIA GPU Operator will handle this in the next step.

## Step 2: Install NVIDIA GPU Operator

Once your AKS cluster is configured with a GPU-enabled node pool, install the NVIDIA GPU Operator. This operator automates the deployment and lifecycle of all NVIDIA software components required to provision GPUs in the Kubernetes cluster, including drivers, container toolkit, device plugin, and monitoring tools.

Follow the [Installing the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) guide to install the GPU Operator on your AKS cluster.

You should see output similar to the example below. Note that this is not the complete output; there should be additional pods running. The most important thing is to verify that the GPU Operator pods are in a `Running` state.

```bash
NAMESPACE     NAME                                                          READY   STATUS    RESTARTS   AGE
gpu-operator  gpu-feature-discovery-xxxxx                                   1/1     Running   0          2m
gpu-operator  gpu-operator-xxxxx                                            1/1     Running   0          2m
gpu-operator  nvidia-container-toolkit-daemonset-xxxxx                      1/1     Running   0          2m
gpu-operator  nvidia-cuda-validator-xxxxx                                   0/1     Completed 0          1m
gpu-operator  nvidia-device-plugin-daemonset-xxxxx                          1/1     Running   0          2m
gpu-operator  nvidia-driver-daemonset-xxxxx                                 1/1     Running   0          2m
```

## Step 3: Deploy Dynamo Kubernetes Operator

Follow the [Deploying Inference Graphs to Kubernetes](../../../docs/kubernetes/README.md) guide to install Dynamo on your AKS cluster.

Validate that the Dynamo pods are running:

```bash
kubectl get pods -n dynamo-system

# Expected output:
# NAME                                                              READY   STATUS    RESTARTS   AGE
# dynamo-platform-dynamo-operator-controller-manager-xxxxxxxxxx     2/2     Running   0          2m50s
# dynamo-platform-etcd-0                                            1/1     Running   0          2m50s
# dynamo-platform-nats-0                                            2/2     Running   0          2m50s
# dynamo-platform-nats-box-xxxxxxxxxx                               1/1     Running   0          2m51s
```

## Step 4: Deploy and Test a Model

Follow the [Deploy Model/Workflow](../../../docs/kubernetes/installation_guide.md#next-steps) guide to deploy and test a model on your AKS cluster.

## AKS Storage options for Model Caching and Runtime Data

For implementing tiered storage you can take advantage of the different storage options available in Azure such as:

| Storage Option | Performance | Best For |
|----------------|-------------|----------|
| Local CSI (Ephemeral Disk) | Very high | Fast model caching, warm restarts |
| [Azure Managed Lustre](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/use-csi-driver-kubernetes) | Extremely high | Large multi-node models, shared cache |
| [Azure Disk (Managed Disk)](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-disk#create-azure-disk-pvs-using-built-in-storage-classes) | High | Persistent single-writer model cache |
| [Azure Files](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-files#use-a-persistent-volume-for-storage) | Medium | Shared small/medium models |
| [Azure Blob (via Fuse or init)](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-blob#create-a-pvc-using-built-in-storage-class) | Low–Medium | Cold model storage, bootstrap downloads |

Note: Azure Managed Lustre and Local CSI (ephemeral disk) are not installed by default in AKS and require additional setup before use. Azure Disk, Azure Files, and Azure Blob CSI drivers are available out of the box. See the [AKS CSI storage options documentation](https://learn.microsoft.com/azure/aks/csi-storage-drivers) for more details.

In the cache.yaml in the different [recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes), you can set the storageClassName to a predefined storage option that are available in your AKS cluster:

```bash
kubectl get storageclass

NAME                           PROVISIONER                 RECLAIMPOLICY
azureblob-csi                  blob.csi.azure.com          Delete
azurefile                      file.csi.azure.com          Delete
azurefile-csi                  file.csi.azure.com          Delete
azurefile-csi-premium          file.csi.azure.com          Delete
azurefile-premium              file.csi.azure.com          Delete
default                        disk.csi.azure.com          Delete
managed                        disk.csi.azure.com          Delete
managed-csi                    disk.csi.azure.com          Delete
managed-csi-premium            disk.csi.azure.com          Delete
managed-premium                disk.csi.azure.com          Delete
sc.azurelustre.csi.azure.com   azurelustre.csi.azure.com   Retain

```
The recommendation for storage options for the Dynamo caches are:

- Model Cache storing raw model artifacts, configuration files, tokenizers etc.<br>
  - Persistence: Required to avoid repeated downloads and reduce cold-start latency.<br>
  - Recommended storage: Azure Managed Lustre (shared, high throughput) or Azure Disk (single-replica, persistent).

- Compilation Cache stores backend-specific compiled artifacts (e.g., TensorRT engines).<br>
  - Persistence: Optional<br>
  - Recommended storage: Local CSI (fast, node-local) or Azure Disk (persistent when GPU configuration is fixed).

- Performance Cache stores runtime tuning and profiling data.<br>
  - Persistence: Not required<br>
  - Recommended storage: Local CSI (or other ephemeral storage).

cache.yaml example:
```bash
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: "sc.azurelustre.csi.azure.com"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: compilation-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: "azurefile-csi"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: "local-ephemeral"
```

## Running on AKS Spot VMs based GPU node pools

When deploying Dynamo on AKS with GPU-enabled [Spot VM](https://azure.microsoft.com/en-us/products/virtual-machines/spot) node pools, AKS will automatically apply the following taint to those Spot nodes to prevent standard workloads from being scheduled on them by default.
```bash
kubernetes.azure.com/scalesetpriority=spot:NoSchedule
```
Because of these taints, workloads (including the Dynamo CRD controller, Platform components, and any GPU workloads) must include below tolerations in their Helm charts. Without these tolerations, Kubernetes will not schedule pods onto the Spot VM node pools, and GPU resources will remain unused.
```bash
tolerations:
  - key: kubernetes.azure.com/scalesetpriority
    operator: Equal
    value: spot
    effect: NoSchedule
```
To schedule Dynamo platform components and jobs onto these nodes, use the provided dynamo/examples/deployments/AKS/values-aks-spot.yaml, which includes all required tolerations for:
- Dynamo operator controller manager
- Webhook CA inject and cert generation jobs
- etcd
- NATS
- MPI SSH key generation job
- Other core Dynamo platform pods

Use the following commands to install or upgrade Dynamo using the AKS Spot values file:
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  --create-namespace \
  -f ./values-aks-spot.yaml
```
or
```bash
helm upgrade dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  -f ./values-aks-spot.yaml
```

## Clean Up Resources

If you want to clean up the Dynamo resources created during this guide, you can run the following commands:

```bash
# Delete all Dynamo Graph Deployments
kubectl delete dynamographdeployments.nvidia.com --all --all-namespaces

# Uninstall Dynamo Platform and CRDs
helm uninstall dynamo-platform -n dynamo-kubernetes
helm uninstall dynamo-crds -n default
```

This will spin down the Dynamo deployment and all associated resources.

If you want to delete the GPU Operator, follow the instructions in the [Uninstalling the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/uninstall.html) guide.

If you want to delete the entire AKS cluster, follow the instructions in the [Delete an AKS cluster](https://learn.microsoft.com/en-us/azure/aks/delete-cluster) guide.
