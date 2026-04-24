# Kimi-K2.5 nvidia/Kimi-K2.5-NVFP4 — Aggregated Deployments on Kubernetes

> **Text only:** Current upstream TensorRT-LLM supports Kimi-K2.5 models by loading the DeepSeek-V3
> text backbone (`text_config`) only. The vision encoder is not loaded, so image inputs are not
> processed. Full multimodal support requires native upstream TRT-LLM support for Kimi K2.5.

This directory contains two aggregated deployment configurations for the `nvidia/Kimi-K2.5-NVFP4` model.

| Deployment | Manifest | Description | Hardware Requirement
|-----------|----------|-------------|----|
| **Standard Aggregated** | [`deploy.yaml`](deploy.yaml) | Basic aggregated serving with KV-aware routing | 1x8 B200 node |
| **Aggregated + EAGLE SpecDec** | [`deploy-specdec.yaml`](deploy-specdec.yaml) | Performant aggregated deployment with EAGLE speculative decoding and KV-aware routing | 8x4 GB200 nodes |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 1x8 B200 GPUs or 8x4 GB200 GPUs
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC
- `deploy-specdec.yaml` uses `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag` and works with a current top-of-tree Dynamo TRT-LLM image

---

## Standard Aggregated Deployment

Uses [`deploy.yaml`](deploy.yaml). This is the simpler configuration -- aggregated serving with KV-aware routing, no CPU-offloaded KV cache.

```bash
kubectl apply -f deploy.yaml -n ${NAMESPACE}
```

This creates:
- A **ConfigMap** (`llm-config`) with TRT-LLM engine parameters (TP=8, EP=8, FP8 KV-cache).
- A **DynamoGraphDeployment** (`kimi-k25-agg`) with a Frontend (KV-router mode) and a TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`.

---

## Aggregated Deployment with EAGLE Speculative Decoding and KV-aware routing

Uses [`deploy-specdec.yaml`](deploy-specdec.yaml). This performant configuration runs KV-aware aggregated serving with EAGLE speculative decoding on GB200.

### Speculative Decoding Prerequisites

- 8 GB200 nodes, each having 4 GPUs per node
- Update the placeholder image tag `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag` in [`deploy-specdec.yaml`](deploy-specdec.yaml) before deploying.

### Additional Model Assets

This deployment needs both the base Kimi weights and the Eagle draft model on the `model-cache` PVC.

Download the base model:

```bash
kubectl apply -f ../../../model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f ../../../model-cache/nvidia/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s
```

Download the Eagle draft model:

```bash
kubectl apply -f ../../../model-cache/nvidia/eagle-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/eagle-download -n ${NAMESPACE} --timeout=6000s
```

The worker config loads the draft model from:

```yaml
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: /opt/models/hub/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/0b0c6ac039089ad2c2418c91c039553381a302d9
```

### Speculative Decoding Deployment Topology

The manifest runs one aggregated frontend and four aggregated worker replicas. Each worker spans two nodes:

- `multinode.nodeCount: 2`
- `resources.limits.gpu: "4"` per node
- `tensor_parallel_size: 8`
- `moe_expert_parallel_size: 8`

This is an 8-node deployment in total for the workers.

### Deployment

```bash
kubectl apply -f deploy-specdec.yaml -n ${NAMESPACE}
```

This creates:
- A **ConfigMap** (`llm-config-specdec`) with the TRT-LLM speculative decoding config
- A **DynamoGraphDeployment** (`kimi-k25-agg-specdec`) with a KV-aware router frontend and four multinode TRT-LLM worker replicas serving `nvidia/Kimi-K2.5-NVFP4`
