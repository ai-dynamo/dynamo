<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Steps to create an EKS cluster

This guide demonstrates Dynamo platform on Amazon Elastic Kuberentes Service (EKS) platform.

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

Creating an EKS Autoi Mode cluster in the AWS region **us-east-1**

```bash
eksctl create cluster --name ai-dynamo --region us-east-1 --enable-auto-mode
```
*Note: eksctl will automatically configure kubeconfig context for you*

### Create an EKS Auto Mode GPU NodePool

Creating a GPU NodePool that targets the **g5,g6,g6e,g7e,p5,p5e,p5en** instance families.

```bash
kubectl apply -f automode-np-gpu.yaml
```

## Create a default StorageClass

Create a default StorageClass to use the storage capability of EKS Auto Mode.

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

## Install Dynamo Kubernetes Platform

### Set environment variables

```bash
export DYNAMO_NAMESPACE=dynamo-system
export DYNAMO_RELEASE_VERSION=1.0.0
```

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

## Deploy a Dynamo Inference Graph

### Disaggregated Serving

To use EFA, you will need to use Dynamo's EFA container `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0-efa-amd64` that includes the [EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html) bundled already.

The disaggrated serving example uses g7e.12xlarge instance that is capable of GPUDirect RDMA, using Dynamo vLLM container with EFA support.

*Note: For supported instance types with EFA, visit [the AWS EC2 Docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types)*

```yaml
        nodeSelector:
          node.kubernetes.io/instance-type: g7e.12xlarge
```

Using NIXL with LIBFABRIC support for kv transfer in vLLM requires to use the following argument `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'`

*Note: If running on instance type that doesn't support EFA, NIXL with libfabric support will fallback to TCP, by default the vLLM's `NixlConnector` is set to use `cuda` as the buffer device, you will need to add `"kv_buffer_device":"cpu"` to the `kv-transfer-config` argument for the `NixlConnector` to work for disaggregated serving without EFA support.*

Requesting an EFA device using `vpc.amazonaws.com/efa` extended resource
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
*Note: EKS Auto Mode already include the EFA device plugin built in.*

We also need to make sure all workers, both prefill and decode, are on the same availability zone, as EFA traffic is limited to a single availability zone.

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
kubectl -n ${DYNAMO_NAMESPACE} apply -f vllm/disagg.yaml
```
*Note: the vllm/disagg-tcp.yaml shows an example of using disaggregated serving without EFA, fallback to TCP, on g6e.2xlarge instances*

Your pods should be running like below output, making sure they are in status "Running".

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
{"id":"chatcmpl-23a7c94b-99cb-42ca-ae56-2397aa5a560f","choices":[{"index":0,"message":{"content":"<think>\nOkay, so I need to develop a character background for someone who's an intrepid explorer in Eldoria, specifically focusing on their motivations,","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1773336002,"model":"Qwen/Qwen3-0.6B","object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":30,"total_tokens":226,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":192}},"nvext":{"worker_id":{"prefill_worker_id":4265733549773195,"prefill_dp_rank":0,"decode_worker_id":7535192362430132,"decode_dp_rank":0},"timing":{"request_received_ms":1773336002136,"prefill_wait_time_ms":0.852483,"prefill_time_ms":12.90597,"ttft_ms":13.758453000000001,"total_time_ms":110.89621500000001,"kv_hit_rate":0.0}}}
```

Watch logs
```bash
kubectl logs -n ${DYNAMO_NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=vllm-disagg --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f
```

Cleanup

```bash
kubectl -n ${DYNAMO_NAMESPACE} delete -f vllm/disagg.yaml
```

### Aggregated Serving

```bash
kubectl -n ${DYNAMO_NAMESPACE} apply -f vllm/agg.yaml
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
kubectl -n ${DYNAMO_NAMESPACE} delete -f vllm/agg.yaml
```

## Cleanup

```bash
eksctl delete cluster --region=us-east-1 --name=ai-dynamo
```