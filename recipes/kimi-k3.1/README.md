<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi-K3.1-Pro Recipe

Recipes for `moonshotai/Kimi-K3.1-Pro` (currently private; authorized access required) on Dynamo.

## Deployment Targets

| Target | GPUs | Parallelism | Validation |
| --- | ---: | --- | --- |
| [GB300 vLLM aggregated](vllm/agg-gb300-agentic/deploy-gb300.yaml) | 8 | TP8 per worker | Text-only benchmark |
| [GB300 vLLM aggregated multimodal](vllm/agg-gb300-agentic/deploy-gb300-multimodal.yaml) | 8 | TP8 per worker | Text-only benchmark with pre-release weights; image input validated |
| [GB300 vLLM disaggregated](vllm/disagg-gb300-agentic/deploy-gb300.yaml) | 24 | TP8 per worker | Text-only benchmark |
| [GB300 vLLM disaggregated multimodal](vllm/disagg-gb300-agentic/deploy-gb300-multimodal.yaml) | 24 | TP8 per worker | Text-only benchmark with pre-release weights; image input validated |
| [GB300 SGLang aggregated](sglang/agg-gb300-agentic/deploy-gb300.yaml) | 8 | TP8 per worker | Functional only; no performance claim |
| [GB300 SGLang disaggregated](sglang/disagg-gb300-agentic/deploy-gb300.yaml) | 16 | TP8 per worker | Functional only; no performance claim |
| [B200 vLLM aggregated](vllm/agg-b200-agentx/deploy-b200.yaml) | 16 | TP8 × PP2 | Text-only benchmark |
| [VR200 vLLM aggregated](vllm/agg-vr200-agentx/deploy-vr200.yaml) | 16 | TP8 per worker | Text-only benchmark |
| [VR200 vLLM disaggregated](vllm/disagg-vr200-agentx/deploy-vr200.yaml) | 24 | TP8 per worker | Text-only benchmark |

Weights: the B200 and GB300 recipes expect the model at `/shared-model-cache/moonshotai/Kimi-K3.1-Pro`, on the `shared-model-cache` PVC. Create the PVC and download the weights with `model-cache/model-cache.yaml` and `model-cache/model-download.yaml`. The download Job needs a `hf-token-secret` Secret with your Hugging Face token. Set `storageClassName` in the PVC to a ReadWriteMany class on your cluster.

The B200 recipe also needs the draft checkpoint at its `SPECULATIVE_CONFIG` path on the PVC. VR200 uses node-local main weights at `/mnt/vr-nvme/models/moonshotai/Kimi-K3.1-Pro` and draft weights at `/mnt/vr-nvme/models/kimi-k3-dspark`; stage these on the nodes selected by the recipe. The shared download Job downloads only the main weights onto the PVC.

Deploy B200 or VR200 with the target overlay:

```bash
kubectl apply -k recipes/kimi-k3.1/vllm/agg-b200-agentx/kustomize/overlays/b200 -n ${NAMESPACE}
# Or choose one VR200 topology:
kubectl apply -k recipes/kimi-k3.1/vllm/agg-vr200-agentx/kustomize/overlays/vr200 -n ${NAMESPACE}
kubectl apply -k recipes/kimi-k3.1/vllm/disagg-vr200-agentx/kustomize/overlays/vr200 -n ${NAMESPACE}
```

Set `NAMESPACE` to your namespace and review storage, scheduling, and registry settings first. These targets use DSpark speculative decoding; their `*-synthetic-al` overlays reproduce the performance configuration with synthetic acceptance length 3.84 and seven draft tokens. GB300 and VR200 hardware overlays include ComputeDomain channel claims and require the NVIDIA DRA driver.

Deploy: render a recipe with kustomize and apply it. Each recipe has a `generic` overlay with no cluster-specific settings, for example:

    kubectl kustomize recipes/kimi-k3.1/sglang/agg-gb300-agentic/kustomize/overlays/generic | kubectl apply -f -

Use the SKU overlay (for example `gb300`) to add scheduling, registry credentials, and ComputeDomain settings for your cluster.

Modes and limits:

- Performance results are for text-only traffic.
- Both SGLang recipes are experimental and functional-only. No SGLang performance claims are made.

Docs and benchmark:

- Docs page: `docs/fern/pages/recipes/model-recipes/kimi-k3-1-pro.mdx` (deploy, smoke test, benchmarks, and target comparison).
- Benchmark: `perf/perf.yaml` runs AIPerf 0.13.0 with the AgentX workload against a deployed frontend. See the docs page for benchmark deployment commands.
