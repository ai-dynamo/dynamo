---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: ModelExpress
subtitle: Speed up model weight distribution across Kubernetes workers
---

ModelExpress is a model weight distribution service for faster worker startup in larger Dynamo clusters. Instead of every worker downloading the full model from storage, one worker can publish model weight availability and later workers can pull compatible tensors from that source over NIXL/RDMA. ModelExpress can also pair with ModelStreamer to stream safetensors directly from object storage into GPU memory.

Use ModelExpress when model rollout time, autoscale cold start, or fleet-wide model updates matter more than the simplicity of a shared PVC. For smaller clusters, start with [Model Caching](model-caching.md).

## When to Use It

| Scenario | Recommended path |
| --- | --- |
| Small cluster or first deployment | [Model Caching](model-caching.md) with PVC + download Job |
| Large cluster with many replicas | ModelExpress P2P distribution |
| Models already on shared storage | PVC or shared filesystem path |
| Models in S3, GCS, Azure Blob Storage, or local safetensors paths | ModelExpress + ModelStreamer |
| Frequent model updates across a fleet | ModelExpress P2P, optionally seeded by ModelStreamer |
| ModelExpress server has non-shared storage | ModelExpress with `MODEL_EXPRESS_NO_SHARED_STORAGE=1` |

## How It Works

1. A ModelExpress server runs in the cluster and stores metadata for available model sources.
2. vLLM workers use the ModelExpress loader (`--load-format mx` on newer images, or `mx-source` / `mx-target` on older split-loader images).
3. If a compatible source worker is already serving the model, a new worker pulls model tensors from that source over NIXL/RDMA.
4. If no source is available, the worker falls back to storage. With ModelStreamer, the first worker can stream safetensors from `s3://`, `gs://`, `az://`, or a local path.
5. The Kubernetes operator can inject `MODEL_EXPRESS_URL` into all Dynamo pods from the platform `modelExpressURL` setting.

## Configure the Platform

Set the ModelExpress server URL when installing the Dynamo platform:

```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.modelExpressURL=http://model-express-server.model-express.svc.cluster.local:8080"
```

If the ModelExpress server is installed separately, point `dynamo-operator.modelExpressURL` at that service. The operator injects the value into worker pods as `MODEL_EXPRESS_URL`.

## Configure vLLM Workers

Use a runtime image that includes the `modelexpress` Python package. For ModelStreamer, the image also needs `runai-model-streamer` and the relevant object-storage SDK dependencies.

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
```

> [!NOTE]
> Use the load format supported by your runtime image. ModelExpress v0.3 and
> newer document the unified `mx` loader. Some older Dynamo images expose
> `mx-source` and `mx-target` loader names instead.

## Mid-Training Weight Refit

Dynamo's vLLM backend can also receive trainer-published weights from
ModelExpress during an RL training run. This is separate from cold-start model
distribution: the worker is already running, and an external trainer publishes a
new version of model weights to the ModelExpress server.

Use a runtime image that includes:

- `modelexpress`
- NIXL with the UCX backend required by the cluster fabric
- `dynamo.vllm.mx_refit.extension.MxRefitWorkerExtension`

Enable the receiver in the DGD worker with `DYN_MX_REFIT_ENABLED=1` and use the
ModelExpress target load format supported by that image:

```yaml
services:
  VllmWorker:
    extraPodSpec:
      mainContainer:
        image: <vllm-runtime-image-with-modelexpress-refit>
        command: ["python3", "-m", "dynamo.vllm"]
        args:
          - --model
          - Qwen/Qwen3-4B-Thinking-2507
          - --load-format
          - mx-target
        env:
          - name: DYN_MX_REFIT_ENABLED
            value: "1"
          - name: MODEL_EXPRESS_URL
            value: modelexpress-server.default.svc.cluster.local:8001
          - name: UCX_TLS
            value: rc,cuda_copy
          - name: NIXL_UCX_TLS
            value: rc,cuda_copy
          - name: UCX_IB_GPU_DIRECT_RDMA
            value: "yes"
          - name: UCX_CUDA_COPY_DMABUF
            value: "yes"
          - name: MX_RDMA_NIC_PIN
            value: auto
```

When `DYN_MX_REFIT_ENABLED=1`, Dynamo registers an engine admin route named
`update_weights_via_mx`. A trainer or orchestration layer can call:

```http
POST /engine/update_weights_via_mx
Content-Type: application/json

{
  "version": 12,
  "mx_config": {
    "mx_server_url": "modelexpress-server.default.svc.cluster.local:8001",
    "timeout_seconds": 300.0,
    "same_rank_only": true,
    "tree_scale_out": true,
    "moe_expert_filter": false,
    "nic_pin": "auto"
  }
}
```

After a successful refit, call `POST /engine/flush_cache` so prefix-cache entries
created with the old weights are discarded. There is no separate prepare step;
the refit receiver reads the versioned tensor metadata published through
ModelExpress.

## Stream Without Shared Storage

If the ModelExpress server cache is on a non-shared volume, workers cannot read the server's local cache path. Set `MODEL_EXPRESS_NO_SHARED_STORAGE=1` on worker pods so the client streams model files from the server over gRPC:

```yaml
services:
  VllmWorker:
    extraPodSpec:
      mainContainer:
        env:
          - name: VLLM_PLUGINS
            value: modelexpress
          - name: MODEL_EXPRESS_NO_SHARED_STORAGE
            value: "1"
```

Use this path when the server has an RWO PVC, runs in a different namespace, or the cluster has no RDMA fabric available. Shared-filesystem mode is still faster when available.

## Stream From Object Storage

Set `MX_MODEL_URI` when the first worker should stream safetensors directly from object storage or a local mounted path:

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
        env:
          - name: VLLM_PLUGINS
            value: modelexpress
          - name: MX_MODEL_URI
            value: s3://my-model-bucket/meta-llama/Llama-3.1-70B-Instruct
          - name: RUNAI_STREAMER_CONCURRENCY
            value: "8"
```

| Storage backend | `MX_MODEL_URI` example | Credential options |
| --- | --- | --- |
| S3 or S3-compatible storage | `s3://bucket/path/to/model` | IRSA / workload identity, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, optional `AWS_ENDPOINT_URL` |
| Google Cloud Storage | `gs://bucket/path/to/model` | GKE Workload Identity, Application Default Credentials, or `GOOGLE_APPLICATION_CREDENTIALS` |
| Azure Blob Storage | `az://container/path/to/model` | Managed Identity, service principal env vars, or `AZURE_ACCOUNT_NAME` / `AZURE_ACCOUNT_KEY` |
| Local filesystem or PVC | `/models/meta-llama/Llama-3.1-70B-Instruct` | Mount the path into the worker pod |

Credentials are consumed by the storage SDKs in the worker pod. They do not flow through the ModelExpress server.

## See Also

- [Model Caching](model-caching.md) - simple PVC-based model caching and the longer ModelExpress background.
- [ModelExpress deployment guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/DEPLOYMENT.md) - server, P2P, and ModelStreamer configuration.
- [Installation Guide](installation-guide.md) - Dynamo platform install options, including `modelExpressURL`.
