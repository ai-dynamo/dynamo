# DGD Templates for Nebius Cluster

Example YAML templates adapted for the Nebius H200 cluster with `aflowers-exemplar` namespace defaults.

## PVC Definitions

Create shared storage for models and compilation cache:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: aflowers-exemplar
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: csi-mounted-fs-path-sc
  resources:
    requests:
      storage: 500Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: compilation-cache
  namespace: aflowers-exemplar
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: csi-mounted-fs-path-sc
  resources:
    requests:
      storage: 50Gi
```

## Model Download Job

Download a HuggingFace model to the PVC before deploying:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
  namespace: aflowers-exemplar
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: downloader
        image: python:3.10-slim
        command:
        - /bin/bash
        - -c
        - |
          pip install huggingface_hub hf_transfer
          export HF_HUB_ENABLE_HF_TRANSFER=1
          huggingface-cli download Qwen/Qwen3-32B --revision main
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: HF_TOKEN
        - name: HF_HOME
          value: /home/dynamo/.cache/huggingface
        volumeMounts:
        - name: model-cache
          mountPath: /home/dynamo/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
```

## Simple Aggregated DGD (vLLM)

Single worker with aggregated prefill/decode:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg
  namespace: aflowers-exemplar
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-agg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      dynamoNamespace: vllm-agg
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
```

## Aggregated with Tensor Parallelism

8 workers with TP=2 (uses 16 GPUs total):

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: agg-8xtp2
  namespace: aflowers-exemplar
spec:
  pvcs:
  - create: false
    name: model-cache
  - create: false
    name: compilation-cache
  services:
    Frontend:
      componentType: frontend
      dynamoNamespace: agg-8xtp2
      replicas: 1
      resources:
        requests:
          cpu: "8"
        limits:
          cpu: "8"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-reset-states
    VllmDecodeWorker:
      componentType: worker
      dynamoNamespace: agg-8xtp2
      envFromSecret: hf-token-secret
      replicas: 8
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          workingDir: /workspace
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-32B
            - --tensor-parallel-size
            - "2"
            - --disable-log-requests
            - --gpu-memory-utilization
            - "0.90"
            - --async-scheduling
            - --block-size
            - "64"
            - --max-model-len
            - "131072"
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
          useAsCompilationCache: true
```

## Disaggregated with KV Router

Separate prefill (6 workers) and decode (2 workers) with KV-aware routing:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: disagg-kv-router
  namespace: aflowers-exemplar
spec:
  pvcs:
  - create: false
    name: model-cache
  - create: false
    name: compilation-cache
  services:
    Frontend:
      componentType: frontend
      dynamoNamespace: disagg-kv-router
      replicas: 1
      resources:
        requests:
          cpu: "8"
        limits:
          cpu: "8"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          command:
            - python
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
            - --router-reset-states
          env:
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      dynamoNamespace: disagg-kv-router
      envFromSecret: hf-token-secret
      replicas: 6
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          workingDir: /workspace
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-32B
            - --is-prefill-worker
            - --tensor-parallel-size
            - "2"
            - --disable-log-requests
            - --gpu-memory-utilization
            - "0.90"
            - --async-scheduling
            - --block-size
            - "64"
            - --max-model-len
            - "131072"
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
          useAsCompilationCache: true
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      dynamoNamespace: disagg-kv-router
      envFromSecret: hf-token-secret
      replicas: 2
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
      extraPodSpec:
        mainContainer:
          image: cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
          workingDir: /workspace
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-32B
            - --tensor-parallel-size
            - "2"
            - --disable-log-requests
            - --gpu-memory-utilization
            - "0.90"
            - --no-enable-prefix-caching
            - --async-scheduling
            - --block-size
            - "64"
            - --max-model-len
            - "131072"
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
          useAsCompilationCache: true
```

## Key Configuration Notes

### Image Selection
- Use Nebius-mirrored images for faster pulls: `cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:<tag>`
- Or use NGC directly: `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:0.8.0rc2-amd64`

### GPU Resources
- `resources.limits.gpu: "2"` - Number of GPUs per worker
- `rdma/ib: "2"` - RDMA for fast inter-node communication

### Volume Mounts
- `model-cache` at `/home/dynamo/.cache/huggingface` - Pre-downloaded models
- `compilation-cache` at `/home/dynamo/.cache/vllm` with `useAsCompilationCache: true`

### Environment Variables
- `HF_HOME` - HuggingFace cache location
- `DYN_HEALTH_CHECK_ENABLED=false` - Disable for better performance
- `envFromSecret: hf-token-secret` - Inject HF token

### Disaggregated Settings
- Prefill workers: `--is-prefill-worker` flag, prefix caching enabled (default)
- Decode workers: `--no-enable-prefix-caching` flag
- Frontend: `--router-mode kv` for KV-aware routing
