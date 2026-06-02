---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Model Caching
subtitle: Download models once and share across all pods in a Kubernetes cluster
---

Large language models can take minutes to download. Without caching, every pod downloads the full model independently, wasting bandwidth and delaying startup. Dynamo supports a simple shared-storage path and a ModelExpress path for faster weight distribution across larger clusters.

## Option 1: PVC + Download Job (Recommended)

The simplest approach: create a shared PVC, run a one-time Job to download the model, then mount the PVC in your DynamoGraphDeployment.

This is the pattern used by all Dynamo recipes today.

### Step 1: Create a Shared PVC

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

### Find the Snapshot Path

After the Job completes, the model is stored in HuggingFace's cache layout:

```
hub/models--<org>--<model>/snapshots/<commit-hash>/
```

For example, `meta-llama/Llama-3.1-70B-Instruct` becomes:

```
hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/9d3b8e0f71f8c1e0f9b7c2a3d4e5f6a7b8c9d0e1/
```

To find the exact commit hash after the download Job completes:

```bash
kubectl run find-snapshot --rm -it --image=busybox --restart=Never \
  --overrides='{
    "spec": {
      "volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": "model-cache"}}],
      "containers": [{
        "name": "f", "image": "busybox",
        "command": ["find", "/c/hub", "-mindepth", "3", "-maxdepth", "3", "-type", "d"],
        "volumeMounts": [{"name": "c", "mountPath": "/c"}]
      }]
    }
  }'
```

Alternatively, look up the commit hash on the HuggingFace Hub model page under **Files and versions**.

You need this path for the `pvcModelPath` field in a DGDR spec (see [Model Deployment Guide — Model Caching](model-deployment-guide.md#model-caching)).

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

All `VllmWorker` pods that mount `model-cache` now read from the shared cache, avoiding per-pod worker downloads. If you also want the frontend to reuse tokenizer and config files, mount the same PVC there too.

### Compilation Cache

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

## Option 2: ModelExpress (P2P Distribution)

[ModelExpress](https://github.com/ai-dynamo/modelexpress) is a model weight distribution service that integrates with vLLM's weight loading pipeline. It can publish model weights from one worker and let later workers pull those tensors from GPU memory over NIXL/RDMA instead of repeating a full storage download.

ModelExpress can also use **ModelStreamer** as a loading strategy. ModelStreamer streams safetensors directly from object storage or a local filesystem path into GPU memory through the `runai-model-streamer` package. In that setup, the first worker can stream from storage and then publish ModelExpress metadata so later workers can use the P2P path.

Use this path when startup time or fleet-wide model rollout time matters more than the simplicity of a shared PVC.

### How It Works

1. A ModelExpress server runs in the cluster and stores metadata for available sources.
2. vLLM workers use the ModelExpress loader (`--load-format mx` on newer ModelExpress images, or `mx-source` / `mx-target` on older split-loader images).
3. If a compatible source is already serving the model, a new worker pulls model tensors from that source over NIXL/RDMA.
4. If no source is available, the worker falls back to storage. With a shared filesystem (RWX PVC, NFS, hostPath), the worker reads directly from the server's cache. Without a shared filesystem, set `MODEL_EXPRESS_NO_SHARED_STORAGE=1` so the client streams files from the server over gRPC; see [Streaming Without Shared Storage](#streaming-without-shared-storage) below. When `MX_MODEL_URI` is set, ModelStreamer can stream safetensors from S3, GCS, Azure Blob Storage, or a local path.
5. The Kubernetes operator can inject `MODEL_EXPRESS_URL` into all Dynamo pods from the platform `modelExpressURL` setting.

### What To Configure

| Layer | What to configure | Notes |
|-------|-------------------|-------|
| Runtime image | Include the `modelexpress` Python package and, for ModelStreamer, `runai-model-streamer` plus the object-storage dependencies. | Dynamo or vLLM raises an import error if the worker uses a ModelExpress load format but the package is missing. |
| ModelExpress server | Deploy the server with Redis or Kubernetes CRD metadata backend. | See the [ModelExpress deployment guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/DEPLOYMENT.md). |
| Dynamo platform | Set `dynamo-operator.modelExpressURL`. | The operator injects `MODEL_EXPRESS_URL` into pods. |
| vLLM worker | Set the ModelExpress load format and point at the server. | Newer ModelExpress images use `--load-format mx`; older Dynamo images may use `mx-source` / `mx-target`. |
| ModelStreamer | Set `MX_MODEL_URI` to the storage location. | Supported URI forms include `s3://...`, `gs://...`, `az://...`, an absolute local path, or a Hugging Face model ID resolved from the local cache. |

### Setup

**Install with Dynamo Platform:**

```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.modelExpressURL=http://model-express-server.model-express.svc.cluster.local:8080"
```

**Configure workers to use ModelExpress:**

```yaml
services:
  VllmWorker:
    extraPodSpec:
      mainContainer:
        image: <vllm-runtime-image-with-modelexpress>
        command: ["python3", "-m", "dynamo.vllm"]
        args:
          - --model
          - meta-llama/Llama-3.1-70B-Instruct
          - --load-format
          - mx
          - --model-express-url
          - http://model-express-server.model-express.svc.cluster.local:8080
        env:
          - name: VLLM_PLUGINS
            value: modelexpress
```

When `MODEL_EXPRESS_URL` is configured in the operator, it is automatically injected as an environment variable into all component pods. Passing `--model-express-url` explicitly is still useful in examples because the worker validates that a server URL is available when using the older `mx-source` / `mx-target` load formats.

<Note>
Use the load format supported by your runtime image. ModelExpress v0.3 and newer document the unified `mx` loader. Some Dynamo images still expose the older split `mx-source` and `mx-target` loader names; those require the same server URL but separate source and target roles.
</Note>

### Streaming Without Shared Storage

If the ModelExpress server's cache is on a non-shared volume (e.g. a `ReadWriteOnce` PVC, a cross-namespace deployment, or any topology where worker pods cannot mount the same filesystem as the server), the default shared-storage mode fails: the server reports the model as downloaded and returns its own local path, the worker cannot read that path from inside its own pod, and the load silently falls back to a direct HuggingFace download -- defeating the point of running ModelExpress.

Set `MODEL_EXPRESS_NO_SHARED_STORAGE=1` on every worker pod to switch the ModelExpress client into gRPC streaming mode. The server then sends model files to the client over the existing gRPC channel and the worker writes them to its own local cache.

```yaml
services:
  VllmWorker:
    extraPodSpec:
      mainContainer:
        image: <vllm-runtime-image-with-modelexpress>
        command: ["python3", "-m", "dynamo.vllm"]
        args:
          - --model
          - meta-llama/Llama-3.1-70B-Instruct
          - --load-format
          - mx
        env:
          - name: VLLM_PLUGINS
            value: modelexpress
          - name: MODEL_EXPRESS_NO_SHARED_STORAGE
            value: "1"
```

`MODEL_EXPRESS_URL` is injected automatically by the operator (`dynamo-operator.modelExpressURL`); you do not need to set it explicitly here. No volume mount for the ModelExpress cache is required on worker pods in this mode.

Use this path when:

- The server runs with an RWO PVC, or in a different namespace from the workers.
- The cluster has no RDMA / InfiniBand fabric available, so P2P over NIXL is not an option.
- You want ModelExpress to act as a centralized download-and-cache server (one HuggingFace pull, fan out over gRPC to many workers) without standing up object storage and `MX_MODEL_URI`.

Shared-filesystem mode is still faster when available, so prefer an RWX PVC mounted on both the server and the workers when the storage class supports it. See the [ModelExpress storage access modes documentation](https://github.com/ai-dynamo/modelexpress/blob/main/docs/DEPLOYMENT.md#storage-access-modes) for the full trade-off and tuning knobs (chunk size, etc.).

### ModelStreamer From Object Storage

Set `MX_MODEL_URI` when the first worker should stream safetensors directly from storage instead of reading a PVC or relying on a prior source worker.

```yaml
services:
  VllmWorker:
    extraPodSpec:
      mainContainer:
        image: <vllm-runtime-image-with-modelexpress-and-modelstreamer>
        command: ["python3", "-m", "dynamo.vllm"]
        args:
          - --model
          - meta-llama/Llama-3.1-70B-Instruct
          - --load-format
          - mx
          - --model-express-url
          - http://model-express-server.model-express.svc.cluster.local:8080
        env:
          - name: VLLM_PLUGINS
            value: modelexpress
          - name: MX_MODEL_URI
            value: s3://my-model-bucket/meta-llama/Llama-3.1-70B-Instruct
          - name: RUNAI_STREAMER_CONCURRENCY
            value: "8"
```

ModelStreamer relies on the underlying cloud SDK credentials:

| Storage backend | `MX_MODEL_URI` example | Credential options |
|-----------------|------------------------|--------------------|
| S3 or S3-compatible storage | `s3://bucket/path/to/model` | IRSA / workload identity, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, and optional `AWS_ENDPOINT_URL` |
| Google Cloud Storage | `gs://bucket/path/to/model` | GKE Workload Identity, Application Default Credentials, or `GOOGLE_APPLICATION_CREDENTIALS` |
| Azure Blob Storage | `az://container/path/to/model` | Managed Identity, service principal env vars, or `AZURE_ACCOUNT_NAME` / `AZURE_ACCOUNT_KEY` |
| Local filesystem or PVC | `/models/meta-llama/Llama-3.1-70B-Instruct` | Mount the path into the worker pod |

Credentials are consumed by the storage SDKs in the worker pod. They do not flow through the ModelExpress server.

### Relationship To Shadow Engine Failover

ModelExpress and ModelStreamer are model loading and distribution paths. They are not required for [Shadow Engine Failover](shadow-engine-failover.md), and enabling them does not create standby engines.

Use Shadow Engine Failover only when you specifically need an active/shadow recovery topology backed by GPU Memory Service (GMS), DRA, and a backend load format such as `--load-format gms`. Keep the ModelExpress / ModelStreamer configuration separate unless you have validated a combined workflow for your runtime image and cluster.

### When to Use ModelExpress

| Scenario | Recommended Approach |
|----------|---------------------|
| Small cluster, simple setup | PVC + Download Job |
| Large cluster, many nodes | ModelExpress P2P |
| Models already on shared storage (NFS) | PVC |
| Models in S3, GCS, Azure Blob Storage, or local safetensors paths | ModelExpress + ModelStreamer |
| Frequent model updates across fleet | ModelExpress P2P, optionally seeded by ModelStreamer |
| ModelExpress server with non-shared storage (RWO PVC, cross-namespace) | ModelExpress with `MODEL_EXPRESS_NO_SHARED_STORAGE=1` |

## See Also

- [Managing Models with DynamoModel](deployment/dynamomodel-guide.md) — declarative model management CRD
- [Detailed Installation Guide](installation-guide.md) — Helm chart configuration including ModelExpress
- [Shadow Engine Failover](shadow-engine-failover.md) — GMS-backed active/shadow engine recovery, separate from model distribution
- [ModelExpress deployment guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/DEPLOYMENT.md) — server, P2P, and ModelStreamer configuration
- [LoRA Adapters](../features/lora/README.md) — dynamic adapter loading (separate from base model caching)
