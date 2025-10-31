# Dynamo on AKS


This document covers the process of deploying Dynamo Cloud and running inference in a vLLM distributed runtime within a Azure Kubernetes environment, covering the setup process on a Azure Kubernetes Cluster, all the way from setup to testing inference.


### Task 1. Infrastructure Deployment

1. Open **Azure Cloud Shell** or a ternimal on an Azure VM and install pre-reqs:
```
az login

az extension add --name aks-preview
az extension update --name aks-preview
```

generate an rsa ssh key for using with aks cluster:
```
ssh-keygen -t rsa -b 4096 -C "<email@id.com>"
```

2. Create AKS Cluster
  ```
  export REGION=<region>
  export RESOURCE_GROUP=<rg_name>
  export ZONE=<zone>
  export CLUSTER_NAME=<aks_cluster_name>
  export CPU_COUNT=1

az aks create -g  $RESOURCE_GROUP -n $CLUSTER_NAME --location $REGION --zones $ZONE --node-count $CPU_COUNT --enable-node-public-ip --ssh-key-value /home/user/.ssh/id_rsa.pub
```

3. Check if it was created correctly
``` bash
# Get Credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

kubectl config get-contexts

#You should see output like this:
CURRENT   NAME         CLUSTER      AUTHINFO                                   NAMESPACE
*         dynamo-aks   dynamo-aks   clusterUser_<rg_name>_<aks_cluster_name>
```

4. Create GPU node pool: You can use as many computes of whatever SKU you want, here we have used 4 nodes of standard_nc24ads_a100_v4, which have 1 A100 each.
```
az aks nodepool add --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --name gpupool --node-count 4 --skip-gpu-driver-install --node-vm-size standard_nc24ads_a100_v4 --node-osdisk-size 2048 --max-pods 110
```

### Task 2. Install Nvidia GPU Operator

Once your AKS cluster is configured with a GPU-enabled node pool, we can proceed with setting up the NVIDIA GPU Operator. This operator automates the deployment and lifecycle of all NVIDIA software components required to provision GPUs in the Kubernetes cluster. The NVIDIA GPU operator enables the infrastructure to support GPU workloads like LLM inference and embedding generation.

1. Add the NVIDIA Helm repository:
```
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --pass-credentials && helm repo update
```

2. Install the GPU Operator:
```
helm install --create-namespace --namespace gpu-operator nvidia/gpu-operator --wait --generate-name
```

3. Validate install (Takes about 5 mins to complete):
```
kubectl get pods -A -o wide
```

You should see output similar to the example below. Note that this is not the complete output, there should be additional pods running. The most important thing is to verify that the GPU Operator pods are in a `Running` state.

```
NAMESPACE     NAME                                                          READY   STATUS    RESTARTS   AGE   IP             NODE
gpu-operator  gpu-operator-xxxx-node-feature-discovery-gc-xxxxxxxxx         1/1     Running   0          40s   10.244.0.194   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxx-node-feature-discovery-master-xxxxxxxxx     1/1     Running   0          40s   10.244.0.200   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxx-node-feature-discovery-worker-xxxxxxxxx     1/1     Running   0          40s   10.244.0.190   aks-nodepool1-xxxx
gpu-operator  gpu-operator-xxxxxxxxxxxxxx                                   1/1     Running   0          40s   10.244.0.128   aks-nodepool1-xxxx
```

For additional guidance on setting up GPU node pools in AKS, refer to the [Microsoft Docs](https://learn.microsoft.com/en-us/azure/aks/gpu-cluster?tabs=add-ubuntu-gpu-node-pool).

### Task 3. Configure Dynamo

1. Pull Dynamo Repo
The Dynamo GitHub repository will be leveraged extensively throughout this walkthrough. Pull the repository using:
```bash
# clone Dynamo GitHub repo
git clone https://github.com/ai-dynamo/dynamo.git

# go to root of Dynamo repo, latest commit at the time of writing this document was 22e6c96f715177c776421c90e9415a7dbc4f661a
cd dynamo
```

2. Install Dynamo from Published Artifacts on NGC (see the [Dynamo Cloud guide](../../../docs/kubernetes/installation_guide.md)):
```bash
export NAMESPACE=dynamo-cloud
export RELEASE_VERSION=0.3.2

#The above linked document says to authenticate using NGC_API_KEY, not neccessary, since this is an openly available container

# Fetch the CRDs helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz

# Fetch the platform helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz

# Step 1: Install Custom Resource Definitions (CRDs)
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz \
  --namespace default \
  --wait \
  --atomic

#Step 2: Install Dynamo Platform
kubectl create namespace ${NAMESPACE}
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}

# Check pod status:
kubectl get pods -n $NAMESPACE

# output should be similar
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-549b5d5xf7rv   2/2     Running   0          2m50s
dynamo-platform-etcd-0                                            1/1     Running   0          2m50s
dynamo-platform-nats-0                                            2/2     Running   0          2m50s
dynamo-platform-nats-box-5dbf45c748-kln82                         1/1     Running   0          2m51s
```

There are other ways to install Dynamo, you can find them [here](../../../docs/kubernetes/installation_guide.md).

### Task 4. Deploy a model

We're going to be deploying MSFTs Phi-3.5-vision-instruct. You can alter this flow to deploy whatever model you need.

Refer: [dynamo/docs/examples/README.md at main · ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/blob/main/docs/examples/README.md)

```bash
# Set your dynamo root directory
cd <root-dynamo-folder>
export PROJECT_ROOT=$(pwd)

# Create a Kubernetes secret containing your sensitive values:
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=${HF_TOKEN} -n ${NAMESPACE}

# Deploying an example (Time taken depends on model, phi3v took ~5mins)
# You can edit the number os replicas of encoder/ decoder independently here to suit your deployment needs

kubectl apply -f examples/multimodal/deploy/k8s/agg-phi3v.yaml -n ${NAMESPACE}

# Get status of deployment
kubectl get dynamoGraphDeployment -n ${NAMESPACE}

# You can use any of the following commands to see logs for debugging
kubectl get pods -n  ${NAMESPACE} -o wide
kubectl logs <pod-name> -n  ${NAMESPACE}
kubectl exec -it <pod-name> -n  ${NAMESPACE} -- nvidia-smi

# Enable Port forwarding to be able to hit a curl request
kubectl get svc -n ${NAMESPACE}

#Look for one that ends in -frontend and use it for port forward.
SERVICE_NAME=$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | sed 's|.*/||' | sed 's|-frontend||' | head -n1)
kubectl port-forward svc/${SERVICE_NAME}-frontend 8000:8000 -n ${NAMESPACE} &
```

#### Task 5. Testing

```
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3.5-vision-instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "What is in this image?" },
          { "type": "image_url", "image_url": { "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" } }
        ]
      }
    ],
    "stream": false
  }'

#Output should be something like:
{"id": "a200785a-a4dd-4208-8ced-2d0ea30351a4", "object": "chat.completion", "created": 1753223375, "model": "microsoft/Phi-3.5-vision-instruct", "choices": [{"index": 0, "message": {"role": "assistant", "content": " The image features a wooden boardwalk extending into a grassy area surrounded by a wetland. There are water lilies in the water, and the sky is clear with a few clouds. The sun is shining, casting light on the scene, and there are trees visible in the background."}, "finish_reason": "stop"}]}
```

## Clean Up Resources

In order to clean up any Dynamo related resources, from the container shell you launched the deployment from, simply run the following command:

```bash
# Delete deployment
kubectl delete dynamoGraphDeployment <your-dep-name> -n ${NAMESPACE}

# Delete the AKS Cluster
az aks delete --name $CLUSTER_NAME --resource-group $RESOURCE_GROUP --yes
```

This will spin down the Dynamo deployment we configured and spin down all the resources that were leveraged for the deployment.
