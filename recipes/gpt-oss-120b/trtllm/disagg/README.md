<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPT-OSS-120B Disaggregated Prefill/Decode

Serves [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) using TensorRT-LLM with
disaggregated prefill/decode via Dynamo on GB200 nodes.

## Topology

| Role    | Nodes | GPUs/node | Total GPUs | Parallelism |
|---------|-------|-----------|------------|-------------|
| Prefill | 1     | 1         | 1          | TP1         |
| Decode  | 1     | 4         | 4          | TP4         |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../../../docs/kubernetes/README.md)
2. **Blackwell GPU nodes** (GB200 or B200)
3. **HuggingFace token** with access to the model

## Deploy

Follow the [top-level Quick Start](../../README.md) to set up the namespace, HuggingFace
token secret, and model download. Then:

```bash
kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

Monitor startup (model loading takes ~15–30 minutes depending on storage speed):

```bash
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/part-of=gpt-oss-disagg -w
```

## Test

```bash
kubectl port-forward svc/gpt-oss-disagg-frontend 8000:8000 -n ${NAMESPACE} &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## Benchmark (optional)

Edit `perf.yaml` to set your namespace and PVC, then run:

```bash
kubectl apply -f trtllm/disagg/perf.yaml -n ${NAMESPACE}
kubectl logs -f -l job-name=gpt-oss-120b-disagg-bench -n ${NAMESPACE}
```

## Key Configuration Notes

### Engine Configs

The `deploy.yaml` includes a ConfigMap with separate engine configurations for
prefill and decode workers. Key differences:

- **Prefill**: TP1, `max_batch_size=64`, `free_gpu_memory_fraction=0.8`, overlap scheduler disabled
- **Decode**: TP4, `max_batch_size=1280`, `free_gpu_memory_fraction=0.85`, overlap scheduler enabled

### KV Transfer

Uses UCX-based cache transceiver (`max_tokens_in_buffer=9216`) for KV cache
transfer between prefill and decode workers.

### Quantization

Uses `W4A8_MXFP4_MXFP8` quantization via the `OVERRIDE_QUANT_ALGO` environment variable.
