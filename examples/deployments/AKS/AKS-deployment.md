# Dynamo on AKS

This document covers the process of deploying Dynamo Cloud and running inference in a vLLM distributed runtime within a Azure Kubernetes Service (AKS) environment, covering the setup process on a Azure Kubernetes Cluster, all the way from setup to testing inference.

## Infrastructure Deployment

- If you don't have an AKS Cluster yet, create one using the [Azure CLI](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-cli), [Azure PowerShell](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-powershell), or the [Azure portal](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal).

- Ensure that your AKS cluster has a node pool with GPU-enabled nodes. Follow the [Use GPUs for compute-intensive workloads on Azure Kubernetes Service (AKS)](https://learn.microsoft.com/en-us/azure/aks/use-nvidia-gpu?tabs=add-ubuntu-gpu-node-pool#skip-gpu-driver-installation) guide to create a GPU-enabled node pool. It is recommended to **skip the GPU driver installation** during node pool creation, as the NVIDIA GPU Operator will handle this in a later step.

## Install Nvidia GPU Operator

Once your AKS cluster is configured with a GPU-enabled node pool, we can proceed with setting up the NVIDIA GPU Operator. This operator automates the deployment and lifecycle of all NVIDIA software components required to provision GPUs in the Kubernetes cluster. The NVIDIA GPU operator enables the infrastructure to support GPU workloads like LLM inference and embedding generation.

Follow [Installing the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) to install the GPU Operator on your AKS cluster.

You should see output similar to the example below. Note that this is not the complete output, there should be additional pods running. The most important thing is to verify that the GPU Operator pods are in a `Running` state.

```
NAMESPACE     NAME                                                          READY   STATUS    RESTARTS   AGE   IP             NODE
gpu-operator  gpu-operator-xxxx-node-feature-discovery-gc-xxxxxxxxx         1/1     Running   0          40s   10.244.0.194   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxx-node-feature-discovery-master-xxxxxxxxx     1/1     Running   0          40s   10.244.0.200   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxx-node-feature-discovery-worker-xxxxxxxxx     1/1     Running   0          40s   10.244.0.190   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxxxxxxxxxxxx                                   1/1     Running   0          40s   10.244.0.128   aks-nodepool1-xxxx
```

## Deploy Dynamo Kubernetes Operator

Follow the [Deploying Inference Graphs to Kubernetes](../../../docs/kubernetes/README.md) guide to install Dynamo on your AKS cluster.

Validate that the Dynamo pods are running:

```bash
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-549b5d5xf7rv   2/2     Running   0          2m50s
dynamo-platform-etcd-0                                            1/1     Running   0          2m50s
dynamo-platform-nats-0                                            2/2     Running   0          2m50s
dynamo-platform-nats-box-5dbf45c748-kln82                         1/1     Running   0          2m51s
```

## Deploy and test a model

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

This will spin down the Dynamo deployment we configured and spin down all the resources that were leveraged for the deployment.

If you want to delete the GPU Operator, you can follow the instructions in the [Uninstalling the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/uninstall.html) guide.

If you want to delete the AKS cluster, you can follow the instructions in the [Delete an AKS cluster](https://learn.microsoft.com/en-us/azure/aks/delete-cluster) guide.
