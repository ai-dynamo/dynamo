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
