# Dynamo Deployment on EKS

This guide covers steps of creating an Amazon EKS cluster, creating a shared storage Amazon EFS and deploying Dynamo Kubernetes Platform and run inference with TRTLLM backend.

[<kbd>Step 1. Create EKS cluster</kbd>](#step-1-create-eks-cluster)

[Step 2. Install Dynamo Kubernetes Platform](#step-2-install-dynamo-kubernetes-platform)

[Step 3. Deploy a model](#step-3-deploy-a-model)

[Step 4. Deploy Dynamo Inference Graph](#step-4-deploy-dynamo-inference-graph)

[Step 5. Test the Deployment](#step-5-test-the-deployment)

### Step 1. Create EKS cluster

#### a) Open a terminal and create a config file for EKS cluster

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
    minSize: 1
    desiredCapacity: 1
    maxSize: 1
    volumeSize: 2048
    efaEnabled: true
    privateNetworking: true
    iam:
      withAddonPolicies:
        imageBuilder: true
        ebs: true
        efs: true
        fsx: true
    preBootstrapCommands:
      - |
        set -ex
        curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
        tar -xf aws-efa-installer-latest.tar.gz
        cd aws-efa-installer
        sudo ./efa_installer.sh -y --enable-gdr || true
        insmod /lib/modules/$(uname -r)/extra/efa_nv_peermem.ko
```

#### b) Create EKS cluster

```
eksctl create cluster -f <FILENAME>.yaml
```

#### c) Create EFS file system

Follow the steps to create an EFS file system: https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md. Make sure you mount subnets in the last step correctly. This will affect whether your nodes are able to access the created EFS file system.

#### d) Create a config file for StorageClass

You can find your `fileSystemId` from AWS EFS. It starts with `fs-`.

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
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-0.6.0.tgz
helm install dynamo-crds dynamo-crds-0.6.0.tgz --namespace default

# Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-0.6.0.tgz
helm install dynamo-platform dynamo-platform-0.6.0.tgz --namespace dynamo-system --create-namespace
```

Check pods status

```
kubectl get pods -n dynamo-system
```

Output should be similar to

```
# Example output
NAME                                                              READY   STATUS              RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-7ffdfb9npwt8   2/2     Running             0          3h3m
dynamo-platform-etcd-0                                            1/1     Running             0          174m
dynamo-platform-nats-0                                            2/2     Running             0          174m
```

### Step 3. Deploy a model

#### a) Build Dynamo TRTLLM runtime image

This step can take a few hours depending on your system

```
git clone https://github.com/ai-dynamo/dynamo.git -b v0.6.0
cd dynamo/container

./build.sh --framework trtllm --use-default-experimental-tensorrtllm-commit --trtllm-use-nixl-kvcache-experimental
```

#### b) Push Image to Amazon ECR

```
# Create an ECR repository
aws ecr get-login-password | docker login --username AWS --password-stdin $DOCKER_SERVER/
aws ecr create-repository --repository-name <ECR_REPOSITORY_NAME>

# Push Image
docker tag dynamo:latest-trtllm $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.6.0
docker push $DOCKER_SERVER/<ECR_REPOSITORY_NAME>:0.6.0
```

#### c) Create Dynamo Inference Graph

For this example, we'll deploy `Qwen/Qwen3-32B` in disaggregated mode. We'll create 12 prefill workers with TP1 and 1 decode worker with TP4 for a total of 16 x H200 GPUs (p5en.48xlarge). Please change `<DYNAMO_TRTLLM_IMAGE>` to your built image name.

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: prefill-config
data:
  prefill.yaml: |
    build_config:
      max_batch_size: 1
      max_num_tokens: 3500
      max_seq_len: 3200
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
      free_gpu_memory_fraction: 0.8
      dtype: auto

    cache_transceiver_config:
      backend: NIXL

    print_iter_log: false

---

apiVersion: v1
kind: ConfigMap
metadata:
  name: decode-config
data:
  decode.yaml: |
    build_config:
      max_batch_size: 224
      max_num_tokens: 224
      max_seq_len: 3200
    tensor_parallel_size: 4
    enable_attention_dp: false
    trust_remote_code: true
    backend: pytorch
    enable_chunked_prefill: false
    disable_overlap_scheduler: false

    cuda_graph_config:
      enable_padding: true
      max_batch_size: 224

    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.8
      dtype: auto

    cache_transceiver_config:
      backend: NIXL

    print_iter_log: false

---

apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: trtllm-v1-disagg-router
spec:
  services:
    Frontend:
      dynamoNamespace: trtllm-v1-disagg-router
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: <DYNAMO_TRTLLM_IMAGE>
      envs:
        - name: DYN_ROUTER_MODE
          value: kv
    TRTLLMPrefillWorker:
      dynamoNamespace: trtllm-v1-disagg-router
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 12
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 64Gi
      envs:
        - name: UCX_LOG_LEVEL
          value: debug
        - name: UCX_PROTO_INFO
          value: "y"
        - name: NCCL_LAUNCH_MODE
          value: PARALLEL
        - name: NCCL_NET_SHARED_COMMS
          value: "0"
        - name: NCCL_DEBUG
          value: WARN
        - name: TRTLLM_LOG_LEVEL
          value: DEBUG
        - name: DYN_LOG
          value: DEBUG
      extraPodSpec:
        mainContainer:
          startupProbe:
            httpGet:
              path: /live
              port: system
              scheme: HTTP
            periodSeconds: 60
            timeoutSeconds: 600
            failureThreshold: 600
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/trtllm
          command:
            - /bin/sh
            - -c
          args:
            - "python3 -m dynamo.trtllm --model-path Qwen/Qwen3-32B --served-model-name Qwen/Qwen3-32B --extra-engine-args /engine_configs/prefill.yaml --disaggregation-mode prefill --disaggregation-strategy prefill_first --publish-events-and-metrics"
          resources:
            requests:
              vpc.amazonaws.com/efa: "2"
            limits:
              vpc.amazonaws.com/efa: "2"
          volumeMounts:
            - name: prefill-config
              mountPath: /engine_configs
        volumes:
          - name: prefill-config
            configMap:
              name: prefill-config
    TRTLLMDecodeWorker:
      dynamoNamespace: trtllm-v1-disagg-router
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "4"
      sharedMemory:
        size: 64Gi
      envs:
        - name: UCX_LOG_LEVEL
          value: debug
        - name: UCX_PROTO_INFO
          value: "y"
        - name: NCCL_LAUNCH_MODE
          value: PARALLEL
        - name: NCCL_NET_SHARED_COMMS
          value: "0"
        - name: NCCL_DEBUG
          value: WARN
        - name: TRTLLM_LOG_LEVEL
          value: DEBUG
        - name: DYN_LOG
          value: DEBUG
      extraPodSpec:
        mainContainer:
          startupProbe:
            httpGet:
              path: /live
              port: system
              scheme: HTTP
            periodSeconds: 60
            timeoutSeconds: 600
            failureThreshold: 600
          image: <DYNAMO_TRTLLM_IMAGE>
          workingDir: /workspace/components/backends/trtllm
          command:
            - /bin/sh
            - -c
          args:
            - "python3 -m dynamo.trtllm --model-path Qwen/Qwen3-32B --served-model-name Qwen/Qwen3-32B --extra-engine-args /engine_configs/decode.yaml --disaggregation-mode decode --disaggregation-strategy prefill_first"
          resources:
            requests:
              vpc.amazonaws.com/efa: "8"
            limits:
              vpc.amazonaws.com/efa: "8"
          volumeMounts:
            - name: decode-config
              mountPath: /engine_configs
        volumes:
          - name: decode-config
            configMap:
              name: decode-config
```

#### Step 4. Deploy Dynamo Inference Graph

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
deepseek-r1-download-784fc5f6f7-nvqdm                             1/1     Running   0          12h
dynamo-platform-dynamo-operator-controller-manager-7ffdfb9d5whh   2/2     Running   0          12h
dynamo-platform-etcd-0                                            1/1     Running   0          12h
dynamo-platform-nats-0                                            2/2     Running   0          12h
trtllm-v1-disagg-router-frontend-84bc7ccf-tkzc6                   1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmdecodeworker-7657dc5fd5-9zc52       1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-4gjq6      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-4vltd      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-62nk8      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-92rnf      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-cdprf      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-h8zvq      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-hp6z8      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-k964d      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-lrml5      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-npf52      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-nzznf      1/1     Running   0          125m
trtllm-v1-disagg-router-trtllmprefillworker-5c4f5969d8-wnmdn      1/1     Running   0          125m
```

#### Step 5. Test the Deployment

```
# Port forward
kubectl port-forward deployment/trtllm-v1-disagg-router-frontend 8080:8000 -n dynamo-system

# Send a request
curl localhost:8080/v1/chat/completions \
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

Output should be similar to

```
# Example output
{"id":"chatcmpl-f3747869-06a2-4d8e-826b-df3358040726","choices":[{"index":0,"message":{"content":"<think>Okay, let's start by developing the character's background. The user wants a detailed background, so I need to cover motivations, skills,","role":"assistant","reasoning_content":null},"finish_reason":"length"}],"created":1763501336,"model":"Qwen/Qwen3-32B","object":"chat.completion","usage":null}
```