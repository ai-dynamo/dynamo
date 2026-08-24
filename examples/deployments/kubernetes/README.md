<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes LLM deployments

These are runnable DynamoGraphDeployment examples for the core backend and topology combinations. Each directory is self-contained: it includes model storage, download, deployment, and, where available, AIPerf manifests.

| Backend | Topology | Model | GPU requirement | Demonstrates |
| --- | --- | --- | --- | --- |
| [vLLM](llm/vllm/aggregated-llama-3-70b/) | Aggregated | Llama 3 70B | 4 GPUs | Single-stage tensor-parallel serving |
| [vLLM](llm/vllm/disaggregated-qwen3-32b-fp8/) | Disaggregated | Qwen3-32B-FP8 | 8 GPUs | NIXL prefill/decode KV transfer |
| [TensorRT-LLM](llm/trtllm/aggregated-qwen3-32b-fp8/) | Aggregated | Qwen3-32B-FP8 | 2 GPUs | PyTorch backend with tensor parallelism |
| [TensorRT-LLM](llm/trtllm/disaggregated-qwen3-32b-fp8/) | Disaggregated | Qwen3-32B-FP8 | 8 GPUs | Independently scaled prefill and decode workers |
| [SGLang](llm/sglang/aggregated-nemotron-3-super-fp8/) | Aggregated | Nemotron-3-Super-FP8 | 4 GPUs | Tensor-parallel serving with KV-aware routing |
| [SGLang](llm/sglang/disaggregated-nemotron-3-super-fp8/) | Disaggregated | Nemotron-3-Super-FP8 | 4 GPUs | NIXL prefill/decode KV transfer |
| [vLLM](llm/vllm/kv-cache-offload-qwen3-32b/) | KV cache offload | Qwen3-32B | 1 GPU | Dynamo KVBM host-memory tier |
| [vLLM](llm/vllm/multimodal-qwen3-vl-32b-fp8/) | Multimodal aggregated | Qwen3-VL-32B-FP8 | 1 GPU | Vision-language inference and resource claims |
| [vLLM](llm/vllm/multinode-disaggregated-llama-3-70b/) | Multinode disaggregated | Llama 3 70B | 16 GPUs | TP8 workers across separate prefill and decode nodes |

## Run an example

Install the Dynamo Kubernetes platform and make a GPU cluster with a `ReadWriteMany` storage class available. Select an example directory, update `model-cache.yaml` with that storage class, and create an access token for the model:

```bash
export NAMESPACE=dynamo-demo
export EXAMPLE=deployments/kubernetes/llm/vllm/aggregated-llama-3-70b

kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  --namespace "${NAMESPACE}"

kubectl apply -f "${EXAMPLE}/model-cache.yaml" --namespace "${NAMESPACE}"
kubectl apply -f "${EXAMPLE}/model-download.yaml" --namespace "${NAMESPACE}"
kubectl wait --for=condition=complete job/model-download --timeout=3600s --namespace "${NAMESPACE}"
kubectl apply -f "${EXAMPLE}/deploy.yaml" --namespace "${NAMESPACE}"
```

The manifests deliberately pin the backend container tag and, where relevant, the Hugging Face model revision. Keep those pins together when updating a deployment: a model or backend bump is an update to the example, not an operator default.

`perf.yaml` is optional and is present only for examples with a workload matched to that topology. Run it after the service is ready, not as a readiness check.
