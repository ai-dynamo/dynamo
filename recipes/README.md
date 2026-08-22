<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Production-Ready Recipes

Production-tested Kubernetes deployment recipes for LLM inference using NVIDIA Dynamo.

No recipe for your model and hardware combination? Point an AI coding agent at this repository
and ask it to author or adapt one; the repo's agent skills guide the process.

> **Prerequisites:** This guide assumes you have already installed the Dynamo Kubernetes Platform.
> If not, follow the **[Kubernetes Deployment Guide](../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)** first.

## Available Recipes

### Aggregated & Disaggregated Recipes

These recipes demonstrate aggregated or disaggregated serving:

| Model | Framework | Mode | GPUs | Deployment | Benchmark | Notes |
|-------|-----------|------|------|------------|-----------|-------|
| **[Qwen3.8-2.4T-A95B-FP8](qwen3.8-2.4t-a95b-fp8/)** | vLLM, SGLang | Aggregated / Disaggregated | 16x GB300 / GB200 | âœ… | âŒ | Hybrid gated-delta-net + 512-expert MoE (262K ctx), FP8 weights + FP8 KV, TP16 over MNNVL, KV-aware routing + prefix caching, reasoning + tool calling | âŒ |
| **[Qwen3.5-122B-A10B-NVFP4](qwen3.5-122b/nvfp4/vllm/agg-b200-agentic/)** | vLLM | Aggregated | 2x B200 | âœ… | âœ… | Hybrid GDN+MoE, NVFP4 + FP8 KV, TP1 x `replicas: 2`, KV-aware routing; agentic profile | âŒ |
| **[Qwen3.5-122B-A10B-NVFP4](qwen3.5-122b/nvfp4/vllm/disagg-b200-agentic/)** | vLLM | Disaggregated | 3x B200 | âœ… | âœ… | Hybrid GDN+MoE, NVFP4 + FP8 KV, 1P2D over NIXL, KV-aware routing; agentic profile | âŒ |
| **[GPT-OSS-120B](gpt-oss-120b/trtllm/agg/)** | TensorRT-LLM | Aggregated | 4x GB200 | âœ… | âœ… | Blackwell only, WideEP | âŒ |
| **[GPT-OSS-120B](gpt-oss-120b/trtllm/disagg/)** | TensorRT-LLM | Disaggregated | 5x Blackwell (GB200/B200) | âœ… | âœ… | Prefill/Decode split | âŒ |
| **[GPT-OSS-120B](gpt-oss-120b/vllm/)** | vLLM | Agg + Disagg | 8x B200 / 8x H200 | âœ… | âœ… | MXFP4 MoE + FP8 KV, 8x TP1 agg / decode-heavy single-node disagg (2P6D B200, 4P4D H200), EAGLE3 spec decode, KV-aware routing, harmony reasoning + tool calling; agentic profile | âŒ |
| **[Qwen3.5-122B-A10B-FP8](qwen3.5-122b/fp8/vllm/agg-h200-agentic/)** | vLLM | Aggregated | 4x H200 | âœ… | âœ… | Hybrid GDN+MoE, TP2 x `replicas: 2`, MTP spec decode, KV-aware routing; agentic profile | âŒ |
| **[Qwen3.5-122B-A10B-FP8](qwen3.5-122b/fp8/vllm/disagg-h200-agentic/)** | vLLM | Disaggregated | 3x H200 | âœ… | âœ… | Hybrid GDN+MoE, 1P2D over NIXL, KV-aware routing, no MTP; agentic profile | âŒ |
| **[GLM-5.2](glm-5.2/)** | SGLang | Aggregated + Disaggregated | 16x/20x B200 or 24x/16x H200 | âœ… | âœ… | B200 NVFP4 or H200 FP8 with FP8 KV, KV-aware routing, EAGLE, B200 HiCache CPU offload, agentic trace profile | âŒ |
| **[DeepSeek-R1](deepseek-r1/sglang/disagg-8gpu/)** | SGLang | Disagg WideEP | 16x H200 | âœ… | âŒ | TP=8, single-node. Use `model-download-sglang.yaml` | âŒ |
| **[DeepSeek-R1](deepseek-r1/sglang/disagg-16gpu/)** | SGLang | Disagg WideEP | 32x H200 | âœ… | âŒ | TP=16, multi-node. Use `model-download-sglang.yaml` | âŒ |
| **[DeepSeek-R1](deepseek-r1/trtllm/disagg/wide_ep/gb200/)** | TensorRT-LLM | Disagg WideEP (GB200) | 36x GB200 | âœ… | âœ… | Multi-node: 8 decode + 1 prefill nodes | âŒ |
| **[DeepSeek-R1](deepseek-r1/)** | vLLM | Disagg DEP16 | 32x H200 | âœ… | âŒ | Multi-node, data-expert parallel | âŒ |
| **[DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/)** | vLLM | Agg + Disagg | 4x B200 / 4x H200 | âœ… | âœ… | Text â€” MoE 284B / 13B active, NVFP4 (B200) / public FP8 (H200) + FP8 KV, agg TP4 (B200) / DP4+TP1+EP (H200), MTP (H200), KV-aware routing, agentic trace profile, reasoning + tool calling; plus disagg 2P1D (12x B200) / 4P3D (28x H200) | âŒ |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/)** | vLLM | Agg + Disagg | 8x B200 / 8x H200 | âœ… | âœ… | Text â€” MoE 1.6T / 49B active (1M ctx; 86k on H200), NVFP4 (B200) / public FP8 (H200) + FP8 KV, TP8 + EP, MTP-2 (B200), KV-aware routing, agentic trace profile, reasoning + tool calling; plus disagg 1P1D (16x B200) / 1P3D (32x H200) | âŒ |
| **[Kimi-K2.5](kimi-k2.5/trtllm/disagg-eagle-kv-router/)** | TensorRT-LLM | Disaggregated | 24x GB200 | âœ… | âœ… | DEP4 prefill + TEP4 decode, TRTLLM-native KV host offload | âŒ |
| **[Kimi-K3](kimi-k3/vllm/)** | vLLM | Agg + Disagg | 16x GB200 / 16x GB300 | âœ… | âŒ | Multimodal MoE (1M ctx), MXFP4 experts + BF16 dense + FP8 KV, TP16 over MNNVL (GB200) / TP8 (GB300), KV-aware routing, FlashInfer MLA, reasoning + tool calling; plus disagg 1P1D (32x GB200) / 1P2D (24x GB300) | âŒ |
| **[Kimi-K2.6](kimi-k2.6/vllm/)** | vLLM | Aggregated | 4x B200 / 8x H200 | âœ… | âœ… | MoE, NVFP4+FP8 KV (B200) / INT4 (H200), TP4/TP8, EAGLE3 MLA spec decode, LMCache CPU offload; text+image, chat + agentic profiles | âŒ |
| **[Nemotron-3-Super](nemotron-3-super/vllm/)** | vLLM | Aggregated | 4x B200 / 4x H200 | âœ… | âœ… | ~120B hybrid Mamba/Attention/MoE (~12B active), NVFP4 (B200) / FP8 (H200) + FP8 KV, TP4+EP, MTP, KV-aware routing; chat + agentic profiles | âŒ |
| **[Nemotron-3-Ultra](nemotron-3-ultra/vllm/)** | vLLM | Agg + Disagg | 4x B200 / 8x H200 | âœ… | âœ… | ~550B hybrid Mamba/Attention/MoE (~55B active), NVFP4 + FP8, TP4 (B200) / TP8 (H200) + EP, MTP, KV-aware routing; chat + agentic, plus 1P1D disagg on B200 | âŒ |

**Legend:**
- **Deployment**: âœ… = Complete `deploy.yaml` manifest available
- **Benchmark**: âœ… = Includes `perf.yaml` for running AIPerf benchmarks

### Functional Recipes (Not Yet Benchmarked)

These recipes demonstrate functional deployments with Dynamo features, but have not yet been performance-tuned or paired with benchmark manifests. None are listed currently.

### Experimental Recipes

These recipes are under active development and may require additional setup steps (e.g., container patching). They are functional but not yet fully validated for production use.

| Model | Framework | Mode | GPUs | Deployment | Notes |
|-------|-----------|------|------|------------|-------|
| **[nvidia/Kimi-K2.5-NVFP4](kimi-k2.5/tokenspeed/agg/nvidia/)** | TokenSpeed | Aggregated | 4x B200 | âœ… | Text only â€” MoE model, TP4Ã—EP4, reasoning + tool calling. Requires [custom container build](kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile) (no public Dynamo+TokenSpeed image yet) and raw `Deployment`s/`Service`s instead of `DynamoGraphDeployment` (operator backend support pending). |
| **[DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/vllm/agg_b200/)** | vLLM | Aggregated | 4x B200 | âœ… | Text only â€” MoE model (284B / 13B active), DP=4 + EP, FP8 KV cache, reasoning + tool calling. Requires [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/vllm/agg_gb200/)** | vLLM | Aggregated | 4x GB200 | âœ… | Text only â€” MoE model (284B / 13B active), TP=4 + EP, `deep_gemm_mega_moe`, FP8 KV cache, reasoning + tool calling (single NVL4 tray). Requires [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/sglang/agg/)** | SGLang | Aggregated | 4x B200 | âœ… | Text only â€” MoE model (284B / 13B active), TP=4, MXFP4 MoE via FlashInfer, EAGLE MTP (3 steps / 4 draft tokens), reasoning + tool calling. Prebuilt image available; optional [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/vllm/agg/b200/)** | vLLM | Aggregated | 8x B200 | âœ… | Text only â€” MoE model (1.6T / 49B active, 1M context), TP=8 + EP, FP4+FP8 mixed checkpoint, FP8 KV cache, CSA+HCA attention, tool calling. Thinking modes unstable on Day-0 â€” run with `thinking: false`. Requires [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/vllm/agg/gb200/)** | vLLM | Aggregated | 8x GB200 (2 NVL4 trays) | âœ… | Text only â€” same model as B200 agg; TP=8 + EP cross-node via NVLink72 (MNNVL) + ComputeDomain. Requires [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200/)** | vLLM | Disaggregated | 16x GB200 (4 NVL4 trays) | âœ… | Text only â€” DP=8 + EP per worker, 1P + 1D, NVLink72 (MNNVL) + ComputeDomain. Requires [custom container build](deepseek-v4/container/). |
| **[DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/sglang/agg/)** | SGLang | Aggregated | 8x B200 | âœ… | Text only â€” MoE model (1.6T / 49B active, 1M context), TP=8, MXFP4 MoE via FlashInfer, EAGLE MTP (3 steps / 4 draft tokens), reasoning + tool calling. Prebuilt image available (shared with [DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/sglang/agg/)). |

## Recipe Structure

Each complete recipe follows this standard structure:

```
<model-name>/
â”œâ”€â”€ README.md (optional)           # Model-specific deployment notes
â”œâ”€â”€ model-cache/
â”‚   â”œâ”€â”€ model-cache.yaml          # PersistentVolumeClaim for model storage
â”‚   â””â”€â”€ model-download.yaml       # Job to download model from HuggingFace
â””â”€â”€ <framework>/                  # vllm, sglang, or trtllm
    â””â”€â”€ <deployment-mode>/        # agg, disagg, disagg-single-node, etc.
        â”œâ”€â”€ deploy.yaml           # Complete DynamoGraphDeployment manifest
        â””â”€â”€ perf.yaml (optional)  # AIPerf benchmark job
```

In addition, [`accuracy/`](accuracy/) is a shared, model-agnostic accuracy
check (deliberately outside the per-model structure above): point it at any
deployed recipe to compare the served model's benchmark score against its
model card. See [`accuracy/README.md`](accuracy/README.md).

## Quick Start

### Prerequisites

**1. Dynamo Platform Installed**

The recipes require the Dynamo Kubernetes Platform to be installed. Follow the installation guide:

- **[Kubernetes Deployment Guide](../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)** - Quickstart (~10 minutes)
- **[Detailed Installation Guide](../docs/fern/pages/kubernetes/installation/install-dynamo.md)** - Advanced options

**2. GPU Cluster Requirements**

Ensure your cluster has:
- GPU nodes matching recipe requirements (see table above)
- GPU operator installed
- Appropriate GPU drivers and container runtime

**3. HuggingFace Access**

Configure authentication to download models:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}
```

**4. Storage Configuration**

Update the `storageClassName` in `<model>/model-cache/model-cache.yaml` to match your cluster:

```bash
# Find your storage class name
kubectl get storageclass

# Edit the model-cache.yaml file and update:
# spec:
#   storageClassName: "your-actual-storage-class"
```

### Deploy a Recipe

**Step 1: Download Model**

```bash
cd recipes
# Update storageClassName in model-cache.yaml first!
kubectl apply -f <model>/model-cache/ -n ${NAMESPACE}

# Wait for download to complete (may take 10-60 minutes depending on model size)
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s

# Monitor progress
kubectl logs -f job/model-download -n ${NAMESPACE}
```

**Step 2: Deploy Service**

Update the image in `<model>/<framework>/<mode>/deploy.yaml`.

```bash
kubectl apply -f <model>/<framework>/<mode>/deploy.yaml -n ${NAMESPACE}

# Check deployment status
kubectl get dynamographdeployment -n ${NAMESPACE}

# Check pod status
kubectl get pods -n ${NAMESPACE}

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=<deployment-name> -n ${NAMESPACE} --timeout=600s
```

**Step 3: Test Deployment**

```bash
# Port forward to access the service locally
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}

# In another terminal, test the endpoint
curl http://localhost:8000/v1/models

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Step 4: Run Benchmark (Optional)**

```bash
# Only if perf.yaml exists in the recipe directory
kubectl apply -f <model>/<framework>/<mode>/perf.yaml -n ${NAMESPACE}

# Monitor benchmark progress
kubectl logs -f job/<benchmark-job-name> -n ${NAMESPACE}

# View results after completion
kubectl logs job/<benchmark-job-name> -n ${NAMESPACE} | tail -50
```


## Example Deployments

### GPT-OSS-120B with TensorRT-LLM (Aggregated)

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}

# Deploy
cd recipes
kubectl apply -f gpt-oss-120b/model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s
kubectl apply -f gpt-oss-120b/trtllm/agg/deploy.yaml -n ${NAMESPACE}

# Test
kubectl port-forward svc/gpt-oss-120b-agg-frontend 8000:8000 -n ${NAMESPACE}
```

### DeepSeek-R1 on GB200 (Multi-node)

See [deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml](deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml) for the complete multi-node WideEP configuration.

## Customization

Each `deploy.yaml` contains:
- **ConfigMap**: Engine-specific configuration (embedded in the manifest)
- **DynamoGraphDeployment**: Kubernetes resource definitions
- **Resource limits**: GPU count, memory, CPU requests/limits
- **Image references**: Container images with version tags

### Key Customization Points

**Model Configuration:**
```yaml
# In deploy.yaml under worker args:
args:
  - python3 -m dynamo.vllm --model <your-model-path> --served-model-name <name>
```

**GPU Resources:**
```yaml
resources:
  limits:
    gpu: "4"  # Adjust based on your requirements
  requests:
    gpu: "4"
```

**Scaling:**
```yaml
services:
  VllmDecodeWorker:
    replicas: 2  # Scale to multiple workers
```

**Router Mode:**
```yaml
# In Frontend args:
args:
  - python3 -m dynamo.frontend --router-mode kv --http-port 8000
# Options: round-robin, kv (KV-aware routing)
```

**Container Images:**
```yaml
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:x.y.z
# Update version tag as needed
```

## Troubleshooting

### Common Issues

**Pods stuck in Pending:**
- Check GPU availability: `kubectl describe node <node-name>`
- Verify storage class exists: `kubectl get storageclass`
- Check resource requests vs. available resources

**Model download fails:**
- Verify HuggingFace token is correct
- Check network connectivity from cluster
- Review job logs: `kubectl logs job/model-download -n ${NAMESPACE}`

**Workers fail to start:**
- Check GPU compatibility (driver version, CUDA version)
- Verify image pull secrets if using private registries
- Review pod logs: `kubectl logs <pod-name> -n ${NAMESPACE}`

**For more troubleshooting:**
- [Dynamo Operator](../docs/fern/pages/developer-guide/knowledge-base/kubernetes/kubernetes-operator/dynamo-operator.md)
- [Observability Documentation](../docs/fern/pages/kubernetes/operations/observability.mdx)

## Related Documentation

- **[Kubernetes Deployment Guide](../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)** - Platform installation and concepts
- **[API Reference](../docs/fern/pages/reference/kubernetes-api/full-api-reference.mdx)** - DynamoGraphDeployment CRD specification
- **[vLLM Backend Guide](../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/overview.md)** - vLLM-specific features
- **[SGLang Backend Guide](../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md)** - SGLang-specific features
- **[TensorRT-LLM Backend Guide](../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/overview.md)** - TensorRT-LLM features
- **[Observability](../docs/fern/pages/kubernetes/operations/observability.mdx)** - Monitoring and logging
- **[Benchmarking Guide](../docs/fern/pages/recipes/feature-benchmarks/benchmarking-guide.md)** - Performance testing

## Contributing

We welcome contributions of new recipes! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Recipe submission guidelines
- Required components checklist
- Testing and validation requirements
- Documentation standards

### Recipe Quality Standards

A production-ready recipe must include:
- âœ… Complete `deploy.yaml` with DynamoGraphDeployment
- âœ… Model cache PVC and download job
- âœ… Benchmark recipe (`perf.yaml`) for performance testing
- âœ… Verification on target hardware
- âœ… Documentation of GPU requirements
