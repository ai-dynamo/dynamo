<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-VL-32B-Instruct-FP8 Recipes

Production-ready deployments for **Qwen/Qwen3-VL-32B-Instruct-FP8**, a 32B vision-language model with FP8 quantization and multimodal (image) support.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**vllm/agg**](vllm/agg/) | 1x H100/H200 | Aggregated | Single GPU, vision + decode combined |
| [**vllm/hetero_hardware_disagg**](vllm/hetero_hardware_disagg/) | 1x Intel XPU + 1x NVIDIA GPU | Disaggregated | Encode on XPU, decode on GPU with embedding transfer via RDMA |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** matching the configuration requirements:
   - **Aggregated**: 1x NVIDIA H100/H200
   - **Disaggregated**: 1x Intel XPU (encode) + 1x NVIDIA GPU (decode) with RDMA connectivity
3. **HuggingFace token** with access to Qwen models
4. **DRA (Dynamic Resource Allocation)** configured for disaggregated mode (ResourceClaimTemplates provided)

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy (choose one configuration)
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}
# OR for disaggregated (apply resource claim templates first):
# kubectl apply -f vllm/hetero_hardware_disagg/intel_xpu_rdma_template.yaml -n ${NAMESPACE}
# kubectl apply -f vllm/hetero_hardware_disagg/nvidia_gpu_rdma_template.yaml -n ${NAMESPACE}
# kubectl apply -f vllm/hetero_hardware_disagg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/qwen3-vl-32b-fp8-vllm-agg-frontend 8000:8000 -n ${NAMESPACE}
# For disaggregated:
kubectl port-forward svc/qwen3-vl-32b-fp8-vllm-disagg-frontend 8000:8000 -n ${NAMESPACE}

# Send a text-only request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Send a multimodal (image) request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

## Run Performance Benchmarks

```bash
# Benchmark (aggregated and disaggregated share the same template)
# Example for aggregated:
./benchmark/run-benchmark.sh --config agg -n ${NAMESPACE}

# Example for disaggregated:
./benchmark/run-benchmark.sh --config disagg -n ${NAMESPACE}
```

## Model Details

- **Model**: `Qwen/Qwen3-VL-32B-Instruct-FP8`
- **Parameters**: 32B
- **Quantization**: FP8 (pre-quantized weights)
- **Modalities**: Text + Vision (image understanding)
- **KV Cache**: FP8 dtype for memory efficiency
- **Context length**: Default model context

## Architecture: Disaggregated Mode

The disaggregated configuration separates vision encoding from text decoding:

```text
Client → Frontend → EncodeWorker (Intel XPU) ──NIXL/RDMA──→ DecodeWorker (NVIDIA GPU) → Response
```

- **EncodeWorker**: Processes images and generates vision embeddings on Intel XPU
- **DecodeWorker**: Runs autoregressive text generation on NVIDIA GPU
- **KV Transfer**: NIXL connector over RDMA for low-latency embedding transfer between devices

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` to match your cluster before deploying
- Model download takes approximately 15-30 minutes depending on network speed
- The disaggregated mode requires RDMA-capable network interfaces between encode and decode nodes
- `ResourceClaimTemplate` files in `vllm/hetero_hardware_disagg/` must be applied before the hetero_hardware_disagg deployment
