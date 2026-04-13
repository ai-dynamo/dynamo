<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Steps to create an EKS cluster

This guide demonstrates the Dynamo platform on Amazon Elastic Kubernetes Service (EKS).

## Setup environment variables

We will use those environment variables throughout this guide.
If you would like to use a different region, modify the `AWS_REGION` variable

```bash
export AWS_REGION="us-east-1"
export CLUSTER_NAME="ai-dynamo"
export DYNAMO_NAMESPACE="dynamo-system"
export DYNAMO_RELEASE_VERSION="1.0.1"
```

## Install CLIs

### Install AWS CLI ([AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))

```bash
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Install Kubernetes CLI ([kubectl installation guide for EKS](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html))

```bash
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.35.2/2026-02-27/bin/darwin/amd64/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
```

### Install Eksctl CLI ([eksctl installation guide](https://eksctl.io/installation/))

```bash
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH
curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"
curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check
tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo mv /tmp/eksctl /usr/local/bin
```

### Install Helm CLI ([Helm setup for EKS](https://docs.aws.amazon.com/eks/latest/userguide/helm.html))

```bash
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 > get_helm.sh
chmod 700 get_helm.sh
./get_helm.sh
```

## Create an EKS Auto Mode cluster

Creating an EKS Auto Mode cluster using Eksctl with `eksctl.yaml`.
This will create an EKS Auto Mode cluster with the Amazon EFS CSI Driver installed as an addon, we will later use Amazon EFS to store model weights and compilation to be used by Dynamo.

```bash
# Use all availability zones in a region, exclude use1-az3 where EKS control plane is not available
export EKS_CP_AZS=$(aws ec2 describe-availability-zones \
      --region ${AWS_REGION} \
      --filters "Name=opt-in-status,Values=opt-in-not-required" \
      --query "AvailabilityZones[?ZoneId!='use1-az3'].[ZoneName]" \
      --output text | sed 's/ /, /g; s/^/  - /')

eksctl create cluster -f <(envsubst < templates/eksctl.yaml)
```
*Note: eksctl will automatically configure kubeconfig context for you, if not you can run: `aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME`*

### Create an EKS Auto Mode GPU NodePool

Creating a GPU NodePool that targets the **g5,g6,g6e,g7e,p5,p5e,p5en** instance families.

```bash
kubectl apply -f automode-np-gpu.yaml
```

## Create a default StorageClass

Create a default StorageClass to use the storage capability of EKS Auto Mode, this will make the default StorageClass to use EBS volumes for Stateful workloads needed by NATS that is used with Dynamo.

```bash
kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: auto-ebs-sc
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
allowedTopologies:
- matchLabelExpressions:
  - key: eks.amazonaws.com/compute-type
    values:
    - auto
provisioner: ebs.csi.eks.amazonaws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  encrypted: "true"
EOF
```

## Create a shared file system

We present two options for creating a shared file system that will be used for model store and compilation cache and make it available as a shared storage for Dynamo workloads.

1. Amazon Elastic File System (EFS) - Follow the [EFS setup guide](EFS.md) to create an EFS file system.
2. Amazon S3 Files - Follow the [S3 Files setup guide](S3Files.md) to create an S3 Files file system.

## Install Dynamo Kubernetes Platform

### Install Dynamo Platform
```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-"${DYNAMO_RELEASE_VERSION}".tgz
helm install dynamo-platform dynamo-platform-"${DYNAMO_RELEASE_VERSION}".tgz --namespace "${DYNAMO_NAMESPACE}" --create-namespace
```

### Setup HuggingFace TOKEN
```bash
export HF_TOKEN=<HF_TOKEN>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${DYNAMO_NAMESPACE}
```

### Verify installation

Validate that the Dynamo platform pods are running, you should see an output similar to output below.

```bash
kubectl get pods -n ${DYNAMO_NAMESPACE}
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-ff54b5dstgcq   1/1     Running   0          106s
dynamo-platform-nats-0                                            2/2     Running   0          106s
```

Validate that the Dynamo CRDs were installed

```bash
kubectl get crds | grep dynamo
dynamocheckpoints.nvidia.com                      2026-03-17T13:18:05Z
dynamocomponentdeployments.nvidia.com             2026-03-17T13:18:06Z
dynamographdeploymentrequests.nvidia.com          2026-03-17T13:18:08Z
dynamographdeployments.nvidia.com                 2026-03-17T13:18:09Z
dynamographdeploymentscalingadapters.nvidia.com   2026-03-17T13:18:10Z
dynamomodels.nvidia.com                           2026-03-17T13:18:10Z
dynamoworkermetadatas.nvidia.com                  2026-03-17T13:18:11Z
```

## Deploy a Dynamo DynamoGraphDeployment (DGD)

| Manifest | Description |
|----------|-------------|
| `manifests/vllm/disagg.yaml` | Disaggregated prefill/decode DGD using NIXL with LIBFABRIC backend over EFA. Targets `g7e.12xlarge` instances with GPUDirect RDMA support for high-throughput KV-cache transfer between prefill and decode workers. |
| `manifests/vllm/disagg-p5.yaml` | Disaggregated prefill/decode DGD using NIXL with LIBFABRIC backend over EFA. Targets `p5.48xlarge` reserved instances with 8 EFA devices (4 EFAs per 1 GPU for p5.48xlarge) and TP-2 for Qwen3-32B. Uses 2 decode and 6 prefill replicas on reserved capacity (`karpenter.sh/capacity-type: reserved`). |
| `manifests/vllm/disagg-tcp.yaml` | Alternative disaggregated prefill/decode inference graph using TCP instead of EFA. Targets `g6e.2xlarge` instances, suitable for instance types without EFA support. |
| `manifests/vllm/agg.yaml` | Aggregated (single-worker) inference graph where a single vLLM worker handles both prefill and decode phases. Simpler deployment without KV-cache transfer overhead. |


### Cache Models on EFS

Before deploying an inference graph, download the model weights onto the shared EFS file system. Each Dynamo recipe includes a `model-cache/model-download.yaml` Job manifest that downloads the model from HuggingFace.

Copy the recipe's download manifest into the local kustomize directory and apply it:

```bash
# Example: cache the Qwen3-32B model which we will be using later
cp ../../../recipes/qwen3-32b/model-cache/model-download.yaml manifests/model-download/model-download.yaml
kubectl kustomize manifests/model-download | kubectl -n ${DYNAMO_NAMESPACE} apply -f -
rm -f manifests/model-download/model-download.yaml
```

The recipe manifests don't set any memory resources on the download container. Without a memory request, the Job pod can get OOMKilled during download — especially for large models. The `kustomization.yaml` in `manifests/model-download/` patches in a memory request to prevent this. By default it adds `4Gi`.

For larger models (e.g. DeepSeek-R1, Nemotron-3-Super-120B) increase this value in `manifests/model-download/kustomization.yaml` before applying:

```yaml
patches:
  - target:
      kind: Job
      name: model-download
    patch: |
      apiVersion: batch/v1
      kind: Job
      metadata:
        name: model-download
      spec:
        template:
          spec:
            containers:
              - name: model-download
                resources:
                  requests:
                    memory: "16Gi"   # increase for larger models
```

Then apply:

```bash
kubectl kustomize manifests/model-download | kubectl -n ${DYNAMO_NAMESPACE} apply -f -
```

Monitor the download Job:

```bash
kubectl -n ${DYNAMO_NAMESPACE} get jobs model-download
kubectl -n ${DYNAMO_NAMESPACE} logs -f job/model-download
```

To re-run a download (e.g. after changing the model or fixing an OOM), delete the previous Job first:

```bash
kubectl -n ${DYNAMO_NAMESPACE} delete job model-download
```

Then copy the new recipe's manifest and apply again.

### Disaggregated Serving

This example deploys a disaggregated prefill/decode Dynamo Inference Graph that uses NIXL with the LIBFABRIC backend using [Elastic Fabric Adapter (EFA)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) for high-throughput KV-cache transfer between workers.

It targets `g7e.12xlarge` instances, which support GPUDirect RDMA, and uses the Dynamo EFA-enabled vLLM container `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0-efa-amd64` that ships with the [EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html) pre-installed.

*Note: For a full list of EFA-supported instance types, see [the AWS EC2 Docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types).*

```yaml
        nodeSelector:
          node.kubernetes.io/instance-type: g7e.12xlarge
```

KV-cache transfer between workers uses [NIXL](https://github.com/ai-dynamo/nixl) with the LIBFABRIC backend. Enable it by passing the following argument to vLLM:

`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'`

*Note: On instance types without EFA support, NIXL's libfabric backend falls back to TCP automatically. However, vLLM's `NixlConnector` defaults to `cuda` as the buffer device, so you must add `"kv_buffer_device":"cpu"` to the `kv-transfer-config` argument for disaggregated serving to work without EFA.*

Request an EFA device for each worker pod using the `vpc.amazonaws.com/efa` extended resource:
```yaml
      resources:
        requests:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "1"
        limits:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "1"
```
*Note: EKS Auto Mode includes the EFA device plugin making `vpc.amazonaws.com/efa` extended resource available.*

All workers (prefill and decode) must be co-located in the same availability zone, since EFA traffic does not cross AZ boundaries. Use a pod affinity rule to enforce this:

```yaml
        affinity:
          podAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              - topologyKey: "topology.kubernetes.io/zone"
                labelSelector:
                  matchLabels:
                    nvidia.com/dynamo-graph-deployment-name: "vllm-disagg"
```

```bash
kubectl -n ${DYNAMO_NAMESPACE} apply -f manifests/vllm/disagg.yaml
```
*Note: `manifests/vllm/disagg-tcp.yaml` provides an alternative example that uses TCP instead of EFA, targeting `g6e.2xlarge` instances.*

Verify that all pods reach `Running` status:

```bash
kubectl -n ${DYNAMO_NAMESPACE} get pods
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-ff54b5dstgcq   1/1     Running   0          39m
dynamo-platform-nats-0                                            2/2     Running   0          39m
vllm-disagg-frontend-85f8476887-wwtwk                             1/1     Running   0          2m13s
vllm-disagg-vllmdecodeworker-510a1741-7666987b-tp58w              1/1     Running   0          2m13s
vllm-disagg-vllmprefillworker-510a1741-54f76d7954-tjgn8           1/1     Running   0          2m13s
```

```bash
kubectl -n ${DYNAMO_NAMESPACE} port-forward svc/vllm-disagg-frontend 8000:8000

curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

You should see output similar to below

```bash
{"id":"chatcmpl-23a7c94b-99cb-42ca-ae56-2397aa5a560f","choices":[{"index":0,"message":{"content":"<think>\nOkay, so I need to develop a character background for someone who's an intrepid explorer in Eldoria, specifically focusing on their motivations,","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1773336002,"model":"Qwen/Qwen3-0.6B","object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":30,"total_tokens":226,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":192}},"nvext":{"worker_id":{"prefill_worker_id":4265733549773195,"prefill_dp_rank":0,"decode_worker_id":7535192362430132,"decode_dp_rank":0},"timing":{"request_received_ms":1773336002136,"prefill_wait_time_ms":0.852483,"prefill_time_ms":12.90597,"ttft_ms":13.758453000000001,"total_time_ms":110.89621500000001,"kv_hit_rate":0.0}}}
```

*Note: The initial request for each worker will occur increased latency, this is due to the NIXL backend handshake and initialization overhead, this operation is only for the very first transfer*

Watch logs
```bash
kubectl logs -n ${DYNAMO_NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=vllm-disagg --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f
```

Cleanup

```bash
kubectl -n ${DYNAMO_NAMESPACE} delete -f manifests/vllm/disagg.yaml
```

### Aggregated Serving

```bash
kubectl -n ${DYNAMO_NAMESPACE} apply -f manifests/vllm/agg.yaml
```

Your pods should be running like below output, making sure they are in status "Running".

```bash
kubectl -n ${DYNAMO_NAMESPACE} get pods
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-ff54b5dstgcq   1/1     Running   0          12m
dynamo-platform-nats-0                                            2/2     Running   0          12m
vllm-agg-frontend-ff8457bcf-tq9jh                                 1/1     Running   0          4m46s
vllm-agg-vllmdecodeworker-d0a70291-759df94478-8lc74               1/1     Running   0          4m46s
```

Watch logs
```bash
kubectl logs -n ${DYNAMO_NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=vllm-agg --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f
```

```bash
kubectl -n ${DYNAMO_NAMESPACE} port-forward svc/vllm-agg-frontend 8000:8000

curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

You should see output similar to below

```bash
{"id":"chatcmpl-093fac0e-f75e-43b5-90dc-96c8c77a2e7c","choices":[{"index":0,"message":{"content":"<think>\nOkay, I need to develop a character background for the explorer in Eldoria. Let me start by understanding the user's query. They mentioned","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1773443560,"model":"Qwen/Qwen3-0.6B","object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":30,"total_tokens":226},"nvext":{"timing":{"request_received_ms":1773443560878,"total_time_ms":99.89782}}}%
```

Cleanup

```bash
kubectl -n ${DYNAMO_NAMESPACE} delete -f manifests/vllm/agg.yaml
```

## Using On-Demand Capacity Reservations (ODCR) and Capacity Blocks (CBs) for ML

GPU instances can be difficult to acquire on-demand. AWS provides two reservation mechanisms to guarantee capacity for ML workloads:

- [On-Demand Capacity Reservations (ODCRs)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html) reserve capacity in a specific AZ for any duration. You pay for the reserved capacity whether or not you use it.
- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) reserve GPU instances for a fixed time window (hours to days). Instances are placed in EC2 UltraClusters for low-latency networking. Capacity Blocks have a defined end time, and EC2 will terminate instances before the block expires.

EKS Auto Mode uses Karpenter under the hood, which models reserved capacity as `karpenter.sh/capacity-type: reserved` and prioritizes it over on-demand and spot.

> **Note:** By default, EKS Auto Mode can launch into open ODCRs automatically, but does not prioritize them. Capacity Blocks are never used automatically. Both require explicit `capacityReservationSelectorTerms` configuration on a NodeClass to be prioritized and labeled as `reserved`.

### Create a NodeClass with Capacity Reservation

Create a NodeClass that references your ODCR or Capacity Block reservation. You can select by reservation ID or by tags.

First, extract the subnet, security group, and role configuration from the `default` NodeClass that EKS Auto Mode already created:

```bash
export NC_SUBNETS=$(kubectl get nodeclass default -o json | jq -c '.spec.subnetSelectorTerms')
export NC_SG=$(kubectl get nodeclass default -o json | jq -c '.spec.securityGroupSelectorTerms')
export NC_ROLE=$(kubectl get nodeclass default -o json | jq -r '.spec.role')
```

Replace `<CR ID>` with your actual reservation ID from the EC2 console.
```bash
export CR_ID=<CR ID>
kubectl apply -f - << EOF
apiVersion: eks.amazonaws.com/v1
kind: NodeClass
metadata:
  name: gpu-reserved
spec:
  role: ${NC_ROLE}
  subnetSelectorTerms: ${NC_SUBNETS}
  securityGroupSelectorTerms: ${NC_SG}
  capacityReservationSelectorTerms:
    # Select by reservation ID (ODCR or Capacity Block)
    - id: "${CR_ID}"
    # Or select by tags (can be combined)
    # - tags:
    #     team: "dynamo"
EOF
```

Wait until the status of the capacityReservation state is `active`.

```bash
kubectl get nodeclass gpu-reserved -o json | jq '.status.capacityReservations'
[
  {
    "availabilityZone": "us-east-2c",
    "endTime": "2026-03-18T11:30:00Z",
    "id": "cr-xxxxxxxxxxxxxx",
    "instanceMatchCriteria": "targeted",
    "instanceType": "p5.48xlarge",
    "ownerID": "xxxxxxxxxxx",
    "reservationType": "capacity-block",
    "state": "active"
  }
]
```

### Create a NodePool for Reserved Capacity

Create a NodePool that references the `gpu-reserved` NodeClass and uses the `reserved` capacity type. You can optionally include `on-demand` and `spot` as a fallback when the reservation is exhausted.

```bash
kubectl apply -f - << EOF
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-reserved
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 300s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    spec:
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: gpu-reserved
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - reserved
            # Uncomment to fallback to on-demand or spot when reservation is exhausted
            # - on-demand
            # - spot
        - key: eks.amazonaws.com/instance-family
          operator: In
          values:
            - g6e
            - g7e
            - p5
            - p5e
            - p5en
      taints:
        - effect: NoSchedule
          key: nvidia.com/gpu
          value: Exists
EOF
```

Validate that the `gpu-reserved` NodePool is ready

```bash
kubectl get nodepool gpu-reserved
NAME           NODECLASS      NODES   READY   AGE
gpu-reserved   gpu-reserved   0       True    8s
```

> **Note:** When configuring `capacityReservationSelectorTerms` on any NodeClass in the cluster, EKS Auto Mode will stop automatically using open ODCRs for all NodeClasses. Make sure all NodeClasses that should use ODCRs have explicit selector terms configured.

### Targeting Reserved Nodes from Workloads

Pods are scheduled onto reserved nodes through the existing NodePool requirements and taints. If you want to ensure a workload only runs on reserved capacity, add a node selector:

```yaml
      nodeSelector:
        karpenter.sh/capacity-type: reserved
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### Capacity Blocks Considerations

Capacity Blocks have a fixed end time. EC2 begins terminating instances 30 minutes before the block expires (60 minutes for UltraServer types). Karpenter will start draining nodes 10 minutes before EC2 termination begins, giving your workloads time to gracefully shut down.

Plan your inference workloads accordingly, and consider using `on-demand` as a fallback capacity type in the NodePool if you need continuity beyond the Capacity Block window.

## Cleanup

Delete all DynamoGraphDeployment

```bash
kubectl -n ${DYNAMO_NAMESPACE} get dgd

# If you have any, delete them
kubectl -n ${DYNAMO_NAMESPACE} delete dgd <name>
```

Uninstall Dynamo platform
```bash
helm uninstall -n ${DYNAMO_NAMESPACE} dynamo-platform
```

Clean leftover PVCs related to NATS
```bash
kubectl -n ${DYNAMO_NAMESPACE} get pvc
NAME                                             STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
dynamo-platform-nats-js-dynamo-platform-nats-0   Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   10Gi       RWO            auto-ebs-sc    <unset>                 75m

kubectl -n ${DYNAMO_NAMESPACE} delete pvc dynamo-platform-nats-js-dynamo-platform-nats-0
```

Delete the AutoMode GPU nodepool

```bash
kubectl delete nodepool gpu
```

Cleanup EFS or S3 Files related resources, follow the [EFS setup guide](EFS.md#cleanup) / [S3 Files setup guide](S3Files.md#cleanup) cleanup section

Delete the EKS Auto Mode cluster using Eksctl

```bash
eksctl delete cluster --wait -f <(envsubst < templates/eksctl.yaml)
```
