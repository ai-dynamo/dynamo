---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Examples
---

For general TensorRT-LLM features and engine configuration, see the
[Reference Guide](../trtllm-reference-guide.md).

## Recommended Path

For multinode TensorRT-LLM deployments, start from the checked-in Kubernetes
recipes under [`recipes/`](../../../../recipes/README.md). Those manifests are
the supported entrypoints for launching multi-node workers, frontend services,
and related routing components.

The main TRT-LLM recipe entrypoints are:

- [DeepSeek-R1 WideEP on GB200](../../../../recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml)
- [Qwen3-235B-A22B-FP8 aggregated](../../../../recipes/qwen3-235b-a22b-fp8/trtllm/agg/deploy.yaml)
- [Qwen3-235B-A22B-FP8 disaggregated](../../../../recipes/qwen3-235b-a22b-fp8/trtllm/disagg/deploy.yaml)
- [Qwen3-32B-FP8 aggregated](../../../../recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml)
- [Qwen3-32B-FP8 disaggregated](../../../../recipes/qwen3-32b-fp8/trtllm/disagg/deploy.yaml)
- [GPT-OSS-120B aggregated](../../../../recipes/gpt-oss-120b/trtllm/agg/deploy.yaml)
- [GPT-OSS-120B disaggregated](../../../../recipes/gpt-oss-120b/trtllm/disagg/deploy.yaml)
- [Nemotron-3-Super-FP8 disaggregated](../../../../recipes/nemotron-3-super-fp8/trtllm/disagg/deploy.yaml)

For model-level setup, prerequisites, and hardware notes, use the recipe
README files:

- [DeepSeek-R1 recipes](../../../../recipes/deepseek-r1/README.md)
- [Qwen3-235B-A22B-FP8 recipes](../../../../recipes/qwen3-235b-a22b-fp8/README.md)
- [Qwen3-32B-FP8 recipes](../../../../recipes/qwen3-32b-fp8/README.md)
- [GPT-OSS-120B recipes](../../../../recipes/gpt-oss-120b/README.md)
- [Kimi-K2.5 recipes](../../../../recipes/kimi-k2.5/README.md)

## Quick Start

At a high level, the Kubernetes workflow is:

1. Install the Dynamo platform on Kubernetes. See the
   [Kubernetes Deployment Guide](../../../kubernetes/README.md).
2. Create a namespace and any required secrets such as a Hugging Face token.
3. Apply the recipe's model cache and model download manifests when the recipe
   includes them.
4. Apply the recipe's `deploy.yaml`.
5. Port-forward the frontend service and send test requests to `/v1/models` or
   `/v1/chat/completions`.

Example flow:

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Example: deploy DeepSeek-R1 TRT-LLM WideEP on GB200.
kubectl apply -f recipes/deepseek-r1/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/deepseek-r1/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
kubectl apply -f recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml -n ${NAMESPACE}
```

After the deployment is ready, port-forward the frontend service named by the
recipe and send a test request:

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/models
```

## Notes

- The TRT-LLM engine config files used by launch and deploy flows live under
  [`examples/backends/trtllm/engine_configs/`](../../../../examples/backends/trtllm/engine_configs/README.md).
- If you need to customize model parallelism, replica counts, or routing mode,
  edit the recipe-local manifest rather than introducing a separate scheduler-specific guide.
- For the current catalog of supported recipes, see [recipes/README.md](../../../../recipes/README.md).
