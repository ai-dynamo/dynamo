---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Model Caching
subtitle: Download models once and share across all pods in a Kubernetes cluster
---

Large language models can take minutes to download. Without caching, every pod downloads the full model independently, wasting bandwidth and delaying startup. Dynamo supports two approaches to ensure models are downloaded once and shared across the cluster.

## Option 1: PVC + Download Job (Recommended)

The simplest approach: create a shared PVC, run a one-time Job to download the model, then mount the PVC in your DynamoGraphDeployment.

This is the pattern used by all Dynamo recipes today.

### Step 1: Create a shared PVC

```yaml
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
```

<Note>
`ReadWriteMany` access mode is required so multiple pods can mount the PVC simultaneously. Ensure your storage class supports RWX (e.g., NFS, CephFS, or cloud-provider shared file systems).
</Note>

### Step 2: Download the model

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: downloader
          image: python:3.12-slim
          command: ["sh", "-c"]
          args:
            - |
              pip install huggingface_hub hf_transfer
              HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
                $MODEL_NAME --revision $MODEL_REVISION
          env:
            - name: MODEL_NAME
              value: "Qwen/Qwen3-0.6B"
            - name: MODEL_REVISION
              value: "main"
            - name: HF_HOME
              value: /cache/huggingface
          envFrom:
            - secretRef:
                name: hf-token-secret
          volumeMounts:
            - name: model-cache
              mountPath: /cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
```

### Step 3: Mount in DynamoGraphDeployment

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  pvcs:
    - create: false
      name: model-cache
  services:
    VllmWorker:
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
```

All worker pods now read from the shared cache — no per-pod downloads.

### Compilation cache

For vLLM, you can also cache compiled artifacts (CUDA graphs, etc.) with a second PVC:

```yaml
spec:
  pvcs:
    - create: false
      name: model-cache
    - create: false
      name: compilation-cache
  services:
    VllmWorker:
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
```

## Option 2: Model Express (P2P Distribution)

[Model Express](https://github.com/ai-dynamo/modelexpress) is a P2P model distribution server that downloads a model once and serves it to all pods over the network. It integrates directly with vLLM's weight loading pipeline via custom load formats.

### How it works

1. A Model Express server runs in the cluster and caches model weights
2. Workers use `--load-format=mx-source` or `--load-format=mx-target` to load from the server
3. The K8s operator injects `MODEL_EXPRESS_URL` into all pods automatically

### Setup

**Install with Dynamo Platform:**

```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.modelExpressURL=http://model-express-server.model-express.svc.cluster.local:8080"
```

**Configure workers to use Model Express:**

```yaml
services:
  VllmWorker:
    envs:
      - name: VLLM_LOAD_FORMAT
        value: mx-target
```

When `MODEL_EXPRESS_URL` is configured in the operator, it is automatically injected as an environment variable into all component pods. Workers using `mx-source` or `mx-target` load formats will connect to the server for model weight distribution.

### When to use Model Express

| Scenario | Recommended Approach |
|----------|---------------------|
| Small cluster, simple setup | PVC + Download Job |
| Large cluster, many nodes | Model Express |
| Models already on shared storage (NFS) | PVC |
| Frequent model updates across fleet | Model Express |

## See Also

- [Managing Models with DynamoModel](deployment/dynamomodel-guide.md) — declarative model management CRD
- [Detailed Installation Guide](installation-guide.md) — Helm chart configuration including Model Express
- [LoRA Adapters](../features/lora/README.md) — dynamic adapter loading (separate from base model caching)
