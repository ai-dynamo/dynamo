---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "GLM-5 NVFP4"
subtitle: "Serve GLM-5 NVFP4 with Dynamo and SGLang, disaggregated across five GB200 nodes."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

This recipe serves `nvidia/GLM-5-NVFP4` with disaggregated prefill/decode and EAGLE MTP speculative decoding across 5 nodes of 4x GB200 (TP4 prefill, TP16/DP16/EP16 decode), sustaining roughly 16.8K output tokens/sec on the standard UCX path (19K with the AWS EFA variant) at 512-way concurrency on a 1K ISL / 8K OSL long-output workload. Pick your KV-transfer path; every command on this page updates to match.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">KV transfer</span>
<input type="radio" id="recipe-variant-standard" name="recipe-variant" value="standard" defaultChecked />
<label htmlFor="recipe-variant-standard">Standard (UCX) <span className="dynamo-target-picker-hint">Recommended</span></label>
<input type="radio" id="recipe-variant-efa" name="recipe-variant" value="efa" />
<label htmlFor="recipe-variant-efa">AWS EFA (custom build)</label>
</div>
<div className="dynamo-target-picker-summary" data-variant="standard">
<span><b>Checkpoint</b> nvidia/GLM-5-NVFP4</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>GPUs</b> 20x GB200, 5 nodes of 4</span>
<span><b>Parallelism</b> TP4 prefill, TP16/DP16/EP16 decode</span>
<span><b>KV transfer</b> NIXL over UCX</span>
<span><b>Workload</b> 1K ISL / 8K OSL, 512 concurrency</span>
</div>
<div className="dynamo-target-picker-summary" data-variant="efa">
<span><b>Checkpoint</b> nvidia/GLM-5-NVFP4</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>GPUs</b> 20x GB200, 5x p6e-gb200.36xlarge</span>
<span><b>Parallelism</b> TP4 prefill, TP16/DP16/EP16 decode</span>
<span><b>KV transfer</b> NIXL LIBFABRIC over EFA RDMA</span>
<span><b>Workload</b> 1K ISL / 8K OSL, 512 concurrency</span>
</div>
</div>

## Prerequisites

- A Kubernetes cluster with the Dynamo Operator plus DRA / ComputeDomain support for MNNVL placement.
- A shared RWX PVC for model weights and FlashInfer JIT artifacts.
- A Hugging Face token with access to `nvidia/GLM-5-NVFP4`.

<div data-variant="standard">

- **5x 4xGB200 nodes** in an NVL36 or NVL72 domain. The published runtime image (`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13`) is used as-is.

</div>

<div data-variant="efa">

- **5x p6e-gb200.36xlarge** nodes (4x GB200 + 4x EFA NICs each) or equivalent GB200 in an MNNVL domain.
- AWS EFA driver 3.0.0g or newer on the nodes (default on modern AWS EKS AMIs).
- A container registry you can push to — this variant requires building a custom image from `Dockerfile.efa` before deploying (there is no prebuilt EFA image).

</div>

Create the namespace and token secret:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

<Warning>
The manifests use standard NVIDIA GPU Feature Discovery labels to select GB200 nodes and include common GPU/ARM tolerations. If your cluster uses different labels, taints, or storage classes, update `nodeSelector`, `tolerations`, and `storageClassName` before deploying.
</Warning>

## Deploy

Create the model-cache PVC and download the weights (shared by both targets):

```bash
# Edit model-cache.yaml first and set storageClassName to a RWX storage class.
kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-cache.yaml -n ${NAMESPACE}

kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

If your cluster already provides a shared RWX cache PVC, skip `model-cache.yaml` and update `claimName: model-cache` in the download, deploy, and perf manifests, keeping the mount path as `/model-store`.

<div data-variant="standard">

Edit `sglang/disagg/deploy.yaml` and replace the `<your-namespace>` placeholder, then:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=glm5-sglang \
  -n ${NAMESPACE} --timeout=7200s
```

First cold starts can take up to about an hour while the runtime loads weights and JIT-compiles FlashInfer/DeepGEMM kernels:

```bash
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/part-of=glm5-sglang -w
```

</div>

<div data-variant="efa">

Build the custom container image first — it bakes a patched libfabric ([ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/pull/12019)) into the runtime so `fi_mr_reg` on CUDA VRAM succeeds on GB200's 64K-page arm64 kernel:

```bash
docker buildx build \
  --platform linux/arm64 \
  --build-arg ARCH=arm64 \
  -t <your-registry>/sglang-dynamo-glm5-efa:latest \
  -f recipes/glm-5-nvfp4/sglang/disagg/efa/Dockerfile.efa \
  --push .
```

Then edit `sglang/disagg/efa/deploy.yaml` to replace `<your-namespace>` and the image placeholder, and:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/efa/deploy.yaml
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/part-of=glm5-sglang-efa -w
```

After startup, verify the LIBFABRIC backend is actually carrying KV traffic (not silent TCP fallback) — the [EFA README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/README.md) includes three checks (NIXL backend log line, executable `libplugin_LIBFABRIC.so` mapping, and `nixl_num_failed_transfers_total` staying at 0).

</div>

## Smoke Test

Send a test request to verify the deployment serves traffic:

<div data-variant="standard">

```bash
kubectl port-forward svc/glm5-sglang-frontend 8000:8000 -n ${NAMESPACE} &
```

</div>

<div data-variant="efa">

```bash
kubectl port-forward svc/glm5-sglang-efa-frontend 8000:8000 -n ${NAMESPACE} &
```

</div>

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/GLM-5-NVFP4","messages":[{"role":"user","content":"Write a one-sentence readiness check."}],"max_tokens":128}'
```

## Benchmark

Both targets ship a `perf.yaml` Kubernetes Job with the same workload shape: ISL=1000, OSL=8192, concurrency=512 (32 per decode GPU), 1,536 requests.

<div data-variant="standard">

The Job wraps this AIPerf run:

```bash
aiperf profile \
  --model nvidia/GLM-5-NVFP4 --tokenizer nvidia/GLM-5-NVFP4 \
  --endpoint-type chat --endpoint /v1/chat/completions --streaming \
  --url http://glm5-sglang-frontend:8000 \
  --synthetic-input-tokens-mean 1000 --output-tokens-mean 8192 \
  --extra-inputs max_tokens:8192 --extra-inputs min_tokens:8192 --extra-inputs ignore_eos:true \
  --concurrency 512 --request-count 1536
```

Edit `sglang/disagg/perf.yaml` to replace the namespace placeholder, then:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/perf.yaml
kubectl logs -f -l job-name=glm5-disagg-bench -n ${NAMESPACE}
```

</div>

<div data-variant="efa">

The Job wraps this AIPerf run:

```bash
aiperf profile \
  --model nvidia/GLM-5-NVFP4 --tokenizer nvidia/GLM-5-NVFP4 \
  --endpoint-type chat --endpoint /v1/chat/completions --streaming \
  --url http://glm5-sglang-efa-frontend:8000 \
  --synthetic-input-tokens-mean 1000 --output-tokens-mean 8192 \
  --extra-inputs max_tokens:8192 --extra-inputs min_tokens:8192 --extra-inputs ignore_eos:true \
  --concurrency 512 --request-count 1536
```

Edit `sglang/disagg/efa/perf.yaml` to replace the namespace placeholder, then:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/efa/perf.yaml
kubectl logs -f -l job-name=glm5-disagg-efa-bench -n ${NAMESPACE}
```

</div>

## Expected Performance

<div data-variant="standard">

Reference AIPerf run for the standard target (ISL=1k, OSL=8k, concurrency=512): 1,536 requests, 0 errors, 747.87s benchmark duration. This is a concurrency-burst benchmark, so TTFT includes queueing under 512 concurrent users.

| Metric | Value |
| --- | ---: |
| Output throughput | 16,824 tokens/sec |
| Request throughput | 2.05 requests/sec |
| TTFT p50 | 15,423 ms |
| ITL avg | 23.31 ms/token |
| Tokens/user/sec avg | 43.39 |
| Request errors | 0 |

</div>

<div data-variant="efa">

Reference AIPerf run for the EFA variant at the same workload shape (ISL=1k, OSL=8k, concurrency=512; measured on 5x p6e-gb200.36xlarge, EFA driver 3.0.0g):

| Metric | Value |
| --- | ---: |
| Output throughput | 19,131 tokens/sec |
| TTFT p50 | 621 ms |
| ITL avg | 24.5 ms/token |
| Output tokens/user/sec | 41.0 |

At long context (ISL=20k, OSL=2k, concurrency=64), the LIBFABRIC backend delivers 39% higher throughput and 56% lower TTFT p50 than the UCX default — full tables in the [EFA README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/README.md).

</div>

## Compare All Targets

Both targets run the same disaggregated topology — 1 prefill node (TP4) plus 4 decode nodes (TP16/DP16/EP16) with EAGLE MTP speculative decoding and FP8 KV cache — and differ only in infrastructure and the KV-transfer backend:

| | Standard (UCX) | AWS EFA (custom build) |
|---|---|---|
| **Hardware** | 20x GB200 (5x 4-GPU nodes, NVL36/NVL72) | 20x GB200 (5x p6e-gb200.36xlarge) |
| **KV transfer** | NIXL over UCX | NIXL LIBFABRIC over EFA RDMA |
| **Container image** | Published runtime image | Custom build from `Dockerfile.efa` |
| **Extra requirements** | — | EFA driver 3.0.0g+, privileged containers |
| **Output throughput** | 16,824 tok/s | 19,131 tok/s |

## Notes

- **EAGLE MTP speculative decoding** (~85-95% accept rate) is enabled by two env vars: `SGLANG_ENABLE_SPEC_V2=1` (EAGLEWorkerV2 with overlap scheduler) and `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1` (quantizes the BF16 MTP layer to FP8 at load time, matching the base model's compute path).
- **FP8 KV cache**: uses `--kv-cache-dtype fp8_e4m3` (the NSA backend auto-selects this on SM100/GB200), saving roughly 50% KV memory vs BF16.
- **FlashInfer JIT cache**: the runtime image has no prebuilt `flashinfer-jit-cache` wheel, so the recipe sets `FLASHINFER_WORKSPACE_BASE=/model-store` to persist first-run JIT artifacts on the shared PVC for later pod starts.
- Worker containers run as root because FlashInfer's bundled cubin package creates TRTLLM MoE symlinks inside its installed package directory during startup. The benchmark pod runs as a non-root user and pins Transformers v5 because `nvidia/GLM-5-NVFP4` declares `tokenizer_class=TokenizersBackend`.
- The standard target sets `UCX_TLS=cuda_copy,cuda_ipc,tcp` for NIXL/UCX KV transfer; the EFA variant instead sets `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` and runs containers privileged so `fi_mr_reg` can pin VRAM for RDMA. Without the env var, SGLang silently falls back to TCP on kernel 6.8+.
- **Recovery caveat**: the decode side is one TP16 rank group spread across four nodes. Treat single decode-pod replacement as disruptive and validate full-group recovery before relying on individual decode pod restarts; in validation, deleting one decode worker left the graph NotReady through repeated rank-group reinitialization attempts.

## Source

- Source README: [recipes/glm-5-nvfp4/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/README.md)
- SGLang disaggregated prefill/decode: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml) and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/perf.yaml)
- EFA variant: [README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/README.md), [Dockerfile.efa](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/Dockerfile.efa), [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/deploy.yaml), and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/sglang/disagg/efa/perf.yaml)
- Setup assets: [recipes/glm-5-nvfp4/model-cache/model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/model-cache/model-cache.yaml) and [recipes/glm-5-nvfp4/model-cache/model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5-nvfp4/model-cache/model-download.yaml)
