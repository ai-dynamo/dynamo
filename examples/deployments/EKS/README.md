# Dynamo 0.9.0 Deployment on EKS

This guide covers steps of creating an Amazon EKS cluster, creating a shared storage Amazon EFS and deploying Dynamo Kubernetes Platform and run inference with both TRTLLM and vLLM backends.

[Step 1. Create EKS cluster](#step-1-create-eks-cluster)

[Step 2. Install Dynamo Kubernetes Platform](#step-2-install-dynamo-kubernetes-platform)

[Step 3. Deploy Dynamo Inference Graph (TRTLLM)](#step-3-deploy-dynamo-inference-graph-trtllm)

[Step 4. Deploy Dynamo Inference Graph (vLLM)](#step-4-deploy-dynamo-inference-graph-vllm)

### Step 1. Create EKS cluster

#### a) Open a terminal and create a config file for EKS cluster

Please change `<CLUSTER_NAME>`, `<CLUSTER_REGION>`, and `<CLUSTER_AZ>`. We'll create 1 CPU node and 2 GPU nodes. The 2 GPU nodes have EFA enabled. The GPU instance as shown below is p5en.48xlarge. This has 8 x H200 with a total of 16 EFA devices. Note that we'll reinstall EFA package to enable GDR.

```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: <CLUSTER_NAME>
  region: <CLUSTER_REGION>

availabilityZones:
  - <CLUSTER_AZ>
  - <CLUSTER_AZ>
  - <CLUSTER_AZ>

iam:
  withOIDC: true

addons:
- name: eks-pod-identity-agent
- name: vpc-cni
  podIdentityAssociations:
  - namespace: kube-system
    serviceAccountName: aws-node
    permissionPolicyARNs:
      - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
- name: aws-efs-csi-driver
  podIdentityAssociations:
  - namespace: kube-system
    serviceAccountName: efs-csi-controller-sa
    permissionPolicyARNs:
      - arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy

managedNodeGroups:
  - name: cpu-ng
    instanceType: c5.2xlarge
    minSize: 1
    desiredCapacity: 1
    maxSize: 1
    volumeSize: 2048
    iam:
      withAddonPolicies:
        imageBuilder: true
        ebs: true
        efs: true
        fsx: true

  - name: p5en-ng
    instanceType: p5en.48xlarge
    minSize: 2
    desiredCapacity: 2
    maxSize: 2
    volumeSize: 2048
    efaEnabled: true
    privateNetworking: true
    iam:
      withAddonPolicies:
        imageBuilder: true
        ebs: true
        efs: true
        fsx: true
```

#### b) Create EKS cluster

```
eksctl create cluster -f <FILENAME>.yaml
```

#### c) Create EFS file system

Follow the steps to create an EFS file system: https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md. Make sure you mount subnets in the last step correctly. This will affect whether your nodes are able to access the created EFS file system.

#### d) Create a config file for StorageClass

Please change `<FILE_SYSTEM_ID>`. You can find your `fileSystemId` from AWS EFS. It starts with `fs-`.

```
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: efs-sc
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: efs.csi.aws.com
parameters:
  fileSystemId: <FILE_SYSTEM_ID>
  provisioningMode: efs-ap
  directoryPerms: "777"
  uid: "1000"
  gid: "1000"
```

#### e) Create StorageClass

```
kubectl apply -f <FILENAME>.yaml
```

### Step 2. Install Dynamo Kubernetes Platform

#### a) Create Secrets

```
# Create Image Pull Secret
export DOCKER_SERVER=<ECR_REGISTRY>
export DOCKER_USERNAME=AWS
export DOCKER_PASSWORD="$(aws ecr get-login-password --region <ECR_REGION>)"

kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=dynamo-system

# Create HuggingFace Secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<YOUR_HF_TOKEN> \
  -n dynamo-system
```

#### b) Run below to install Dynamo Kubernetes Platform
```
# Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-0.9.0.tgz
helm install dynamo-crds dynamo-crds-0.9.0.tgz --namespace default

# Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-0.9.0.tgz
helm install dynamo-platform dynamo-platform-0.9.0.tgz --namespace dynamo-system --create-namespace
```

Check pods status

```
kubectl get pods -n dynamo-system
```

Output should be similar to

```
# Example output
NAME                                                              READY   STATUS              RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-7ffdfb9g29vs   2/2     Running             0          11h
dynamo-platform-etcd-0                                            1/1     Running             0          11h
dynamo-platform-nats-0                                            2/2     Running             0          11h
```

### Step 3. Deploy Dynamo Inference Graph (TRTLLM)

#### a) Build Dynamo TRTLLM runtime image

To enable EFA, you need to build from source as shown below.

```
# Clone Dynamo Repo
git clone https://github.com/ai-dynamo/dynamo.git -b v0.9.0
cd dynamo

# Build image and this can take a few hours depending on your system
python3 container/render.py --framework trtllm --target runtime --make-efa
docker build -t dynamo:latest-trtllm-runtime -f container/<RENDERED_DOCKERFILE> .

# Create an ECR repository
aws ecr get-login-password | docker login --username AWS --password-stdin $DOCKER_SERVER/
aws ecr create-repository --repository-name <ECR_REPOSITORY_NAME>

# Push Image
docker tag dynamo:latest-trtllm-runtime $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.9.0
docker push $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.9.0
```

#### b) Create Dynamo Inference Graph (TRTLLM)

Please change `<DYNAMO_TRTLLM_IMAGE>`. For this example, we'll deploy `Qwen/Qwen3-32B-FP8` in disaggregated mode with KV router. Specificaly, we'll use the [model receipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) from Dynamo repo. This creates 8 prefill workers with TP1 and 4 decode worker with TP2 for a total of 16 x H200 GPUs (p5en.48xlarge). Note that this p5en.48xlarge has 16 EFA devices so each GPU is given 2 as shown below.

| Instance Type | EFA Device Count |
| :--- | :--- |
| `p5en.48xlarge` | 16 |
| `p5.48xlarge` | 32 |
| `G6e.48xlarge` | 4 |

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: prefill-config
data:
  prefill.yaml: |
    max_batch_size: 1
    max_num_tokens: 7800
    max_seq_len: 7800
    tensor_parallel_size: 1
    enable_attention_dp: false
    trust_remote_code: true
    backend: pytorch
    enable_chunked_prefill: false
    disable_overlap_scheduler: true

    cuda_graph_config:
      enable_padding: true
      max_batch_size: 1

    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.7
      dtype: fp8

    cache_transceiver_config:
      backend: NIXL # EFA Libfabric Required

    print_iter_log: false

---

apiVersion: v1
kind: ConfigMap
metadata:
  name: decode-config
data:
  decode.yaml: |
    max_batch_size: 128
    max_num_tokens: 7800
    max_seq_len: 7800
    tensor_parallel_size: 2
    enable_attention_dp: false
    trust_remote_code: true
    backend: pytorch
    enable_chunked_prefill: false
    disable_overlap_scheduler: false

    cuda_graph_config:
      enable_padding: true
      max_batch_size: 128

    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.7
      dtype: fp8

    cache_transceiver_config:
      backend: NIXL # EFA Libfabric Required

    print_iter_log: false

---

apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: trtllm-v1-disagg-router
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          volumeMounts:
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
    TRTLLMPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 8
      resources:
        limits:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "2" # EFA Libfabric Required
        requests:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "2" # EFA Libfabric Required
      envs:
        - name: TRTLLM_NIXL_KVCACHE_BACKEND # EFA Libfabric Required
          value: "LIBFABRIC"
        - name: FI_LOG_PROV # EFA Libfabric Required
          value: "efa"
        - name: FI_PROVIDER # EFA Libfabric Required
          value: "efa"
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/trtllm
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.trtllm \
                --model-path /data/Qwen/Qwen3-32B-FP8 \
                --served-model-name Qwen/Qwen3-32B-FP8 \
                --extra-engine-args /engine_configs/prefill.yaml \
                --disaggregation-mode prefill \
                --publish-events-and-metrics
          volumeMounts:
            - name: prefill-config
              mountPath: /engine_configs
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: prefill-config
            configMap:
              name: prefill-config
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
    TRTLLMDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 4
      resources:
        limits:
          gpu: "2"
          custom:
            vpc.amazonaws.com/efa: "4" # EFA Libfabric Required
        requests:
          gpu: "2"
          custom:
            vpc.amazonaws.com/efa: "4" # EFA Libfabric Required
      envs:
        - name: TRTLLM_NIXL_KVCACHE_BACKEND # EFA Libfabric Required
          value: "LIBFABRIC"
        - name: FI_LOG_PROV # EFA Libfabric Required
          value: "efa"
        - name: FI_PROVIDER # EFA Libfabric Required
          value: "efa"
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/trtllm
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.trtllm \
              --model-path /data/Qwen/Qwen3-32B-FP8 \
              --served-model-name Qwen/Qwen3-32B-FP8 \
              --extra-engine-args /engine_configs/decode.yaml \
              --disaggregation-mode decode \
          volumeMounts:
            - name: decode-config
              mountPath: /engine_configs
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: decode-config
            configMap:
              name: decode-config
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
```

#### c) Deploy Dynamo Inference Graph (TRTLLM)

```
kubectl apply -f <DYNAMO_INFERENCE_GRAPH>.yaml -n dynamo-system
```

Check pods status

```
kubectl get pods -n dynamo-system
```

Output should be similar to

```
# Example output
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-7ffdfb9g29vs   2/2     Running   0          11h
dynamo-platform-etcd-0                                            1/1     Running   0          11h
dynamo-platform-nats-0                                            2/2     Running   0          11h
trtllm-v1-disagg-router-frontend-544c9c46bc-bzqpk                 1/1     Running   0          6m44s
trtllm-v1-disagg-router-trtllmdecodeworker-6d77f66cd7-54hcm       1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmdecodeworker-6d77f66cd7-m5l6d       1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmdecodeworker-6d77f66cd7-tdsm5       1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmdecodeworker-6d77f66cd7-xqzjl       1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-7v82r      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-8r8sc      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-c7vnm      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-dh795      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-dr8kr      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-fknlv      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-jkb8h      1/1     Running   0          6m43s
trtllm-v1-disagg-router-trtllmprefillworker-798b566884-s5bjd      1/1     Running   0          6m43s
```

#### d) Test the Deployment

```
# Port forward
kubectl port-forward deployment/trtllm-v1-disagg-router-frontend 8080:8000 -n dynamo-system

# Send a request
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
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

Output should be similar to

```
# Example output
{"id":"chatcmpl-f44545f6-6b51-4a4f-8cc0-4d5cadf44899","choices":[{"index":0,"message":{"content":"<think>Okay, the user wants me to develop a detailed character background for an explorer seeking Aeloria. Let me start by understanding the query.","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1763753072,"model":"Qwen/Qwen3-32B-FP8","object":"chat.completion","usage":null}
```

#### e) Delete the Deployment

```
kubectl delete -f <DYNAMO_INFERENCE_GRAPH>.yaml -n dynamo-system
```

### Step 4. Deploy Dynamo Inference Graph (vLLM)

#### a) Build Dynamo vLLM runtime image

To enable EFA, you need to build from source as shown below.

```
# Clone Dynamo Repo
git clone https://github.com/ai-dynamo/dynamo.git -b v0.9.0
cd dynamo

# Build image and this can take a few hours depending on your system
python3 container/render.py --framework vllm --target runtime --make-efa
docker build -t dynamo:latest-vllm-runtime -f container/<RENDERED_DOCKERFILE> .

# Build image
./build.sh --framework vllm

# Create an ECR repository
aws ecr get-login-password | docker login --username AWS --password-stdin $DOCKER_SERVER/
aws ecr create-repository --repository-name <ECR_REPOSITORY_NAME>

# Push Image
docker tag dynamo:latest-vllm-runtime $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.9.0
docker push $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.9.0
```

#### b) Create Dynamo Inference Graph (vLLM)

Please change `<DYNAMO_VLLM_IMAGE>`. For this example, we'll deploy `Qwen/Qwen3-32B-FP8` in disaggregated mode with KV router. Specificaly, we'll use the [model receipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) from Dynamo repo. This creates 8 prefill workers with TP1 and 4 decode worker with TP2 for a total of 16 x H200 GPUs (p5en.48xlarge). Note that this p5en.48xlarge has 16 EFA devices so each GPU is given 2 as shown below.

| Instance Type | EFA Device Count |
| :--- | :--- |
| `p5en.48xlarge` | 16 |
| `p5.48xlarge` | 32 |
| `G6e.48xlarge` | 4 |

```
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-v1-disagg-router
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          volumeMounts:
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 4
      resources:
        limits:
          gpu: "2"
          custom:
            vpc.amazonaws.com/efa: "4" # EFA Libfabric Required
        requests:
          gpu: "2"
          custom:
            vpc.amazonaws.com/efa: "4" # EFA Libfabric Required
      envs:
      - name: FI_PROVIDER # EFA Libfabric Required
        value: "efa"
      - name: FI_LOG_PROV # EFA Libfabric Required
        value: "efa"
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/vllm
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.vllm \
              --model /data/Qwen/Qwen3-32B-FP8 \
              --served-model-name Qwen/Qwen3-32B-FP8 \
              --tensor-parallel-size 1 \
              --is-decode-worker \
              --connector none \ # EFA Libfabric Required
              --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' # EFA Libfabric Required
          volumeMounts:
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
    VllmPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 8
      resources:
        limits:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "2" # EFA Libfabric Required
        requests:
          gpu: "1"
          custom:
            vpc.amazonaws.com/efa: "2" # EFA Libfabric Required
      envs:
      - name: FI_PROVIDER # EFA Libfabric Required
        value: "efa"
      - name: FI_LOG_PROV # EFA Libfabric Required
        value: "efa"
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/vllm
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.vllm \
              --model /data/Qwen/Qwen3-32B-FP8 \
              --served-model-name Qwen/Qwen3-32B-FP8 \
              --tensor-parallel-size 1 \
              --is-prefill-worker \
              --connector none \ # EFA Libfabric Required
              --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' # EFA Libfabric Required
          volumeMounts:
            - name: modelcache-pvc
              mountPath: /data
        volumes:
          - name: modelcache-pvc
            persistentVolumeClaim:
              claimName: modelcache-pvc
```

#### c) Deploy Dynamo Inference Graph (vLLM)

```
kubectl apply -f <DYNAMO_INFERENCE_GRAPH>.yaml -n dynamo-system
```

Check pods status

```
kubectl get pods -n dynamo-system
```

Output should be similar to

```
# Example output
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-7ffdfb9g29vs   2/2     Running   0          12h
dynamo-platform-etcd-0                                            1/1     Running   0          12h
dynamo-platform-nats-0                                            2/2     Running   0          12h
vllm-v1-disagg-router-frontend-75bfd68ddc-2cf64                   1/1     Running   0          19m
vllm-v1-disagg-router-vllmdecodeworker-67f64569db-46d9m           1/1     Running   0          19m
vllm-v1-disagg-router-vllmdecodeworker-67f64569db-4pthw           1/1     Running   0          19m
vllm-v1-disagg-router-vllmdecodeworker-67f64569db-9xjc4           1/1     Running   0          19m
vllm-v1-disagg-router-vllmdecodeworker-67f64569db-v9xfw           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-5rx54           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-5vps8           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-5x857           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-bqdt6           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-gds2t           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-mpsrd           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-v7vbh           1/1     Running   0          19m
vllm-v1-disagg-router-vllmprefillworker-f44dd87b9-vq2hb           1/1     Running   0          19m
```

#### d) Test the Deployment

```
# Port forward
kubectl port-forward deployment/vllm-v1-disagg-router-frontend 8080:8000 -n dynamo-system

# Send a request
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
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

Output should be similar to

```
# Example output
{"id":"chatcmpl-570e1472-a18b-4ceb-8eb6-05b9067622b6","choices":[{"index":0,"message":{"content":"<think>Okay, let's see. The user wants me to develop a character background for an explorer seeking Aeloria. The key elements to cover","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1763756222,"model":"Qwen/Qwen3-32B-FP8","object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":30,"total_tokens":226}}
```

#### e) Delete the Deployment

```
kubectl delete -f <DYNAMO_INFERENCE_GRAPH>.yaml -n dynamo-system
```