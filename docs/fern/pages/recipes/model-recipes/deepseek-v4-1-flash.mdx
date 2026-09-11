---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "DeepSeek-V4.1-Flash"
subtitle: "Serve DeepSeek-V4.1-Flash with Dynamo and SGLang on GB200, aggregated or disaggregated."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Both targets are Dynamo and SGLang deployments of DeepSeek-V4.1-Flash. Each uses 8x GB200 GPUs with TP4 workers. The checkpoint is FP8 dense with FP4 MoE experts, and the KV cache is FP8. Both targets serve up to 1,048,576 tokens of context. Pick your serving topology.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated <span className="dynamo-target-picker-hint">Recommended</span></label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-variant="agg">
<span><b>Checkpoint</b> deepseek-ai/DeepSeek-V4.1-Flash</span>
<span><b>Precision</b> FP8 dense, FP4 MoE experts, FP8 KV cache</span>
<span><b>GPUs</b> 8x GB200, two workers</span>
<span><b>Parallelism</b> TP4 and EP4 per worker</span>
<span><b>Speculation</b> DSpark, block size 5</span>
<span><b>Routing</b> KV-aware</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
<div className="dynamo-target-picker-summary" data-variant="disagg">
<span><b>Checkpoint</b> deepseek-ai/DeepSeek-V4.1-Flash</span>
<span><b>Precision</b> FP8 dense, FP4 MoE experts, FP8 KV cache</span>
<span><b>GPUs</b> 4x GB200 prefill + 4x GB200 decode</span>
<span><b>Parallelism</b> TP4 and EP4 per role</span>
<span><b>Speculation</b> None. SGLang refuses it under disaggregation.</span>
<span><b>KV transfer</b> Mooncake over TCP, or RDMA where the cluster provides it</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
</div>

<Warning>
Day-0 recipe. Both targets pass a functional probe. Neither is benchmarked, and neither carries a performance claim.
</Warning>

## Overview

DeepSeek-V4.1-Flash is a Mixture-of-Experts model. SGLang picks its attention, MoE, and GEMM backends from the checkpoint. Do not set those backend flags. A hand-set backend flag selects a slower fallback.

The checkpoint also accepts images. Both targets serve text only, and neither enables multimodal input.

The aggregated target caps the decode CUDA graph batch at 64. DSpark verify graphs run out of memory at batch 256.

## Prerequisites

<div data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- 8x GB200 GPUs on two ARM64 nodes. Each worker needs 4 GPUs on one node.
- A ReadWriteMany PVC named `shared-model-cache`, with at least 1000Gi.
- Access to `deepseek-ai/DeepSeek-V4.1-Flash`, which is 510 GB over 48 shards.

</div>

<div data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- 8x GB200 GPUs on two ARM64 nodes. Prefill and decode each need 4 GPUs.
- The NVIDIA DRA driver with ComputeDomain support. The manifest creates the ComputeDomain and both workers claim its channel.
- A ReadWriteMany PVC named `shared-model-cache`, with at least 1000Gi.
- Access to `deepseek-ai/DeepSeek-V4.1-Flash`, which is 510 GB over 48 shards.

</div>

Create the namespace and the token secret:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

<Warning>
Edit the namespace, node selectors, tolerations, storage class, and image tag to match your cluster.
</Warning>

## Deploy

Both targets read the checkpoint from the `shared-model-cache` PVC with `HF_HUB_OFFLINE=1`. First create the PVC and download the weights:

```bash
kubectl apply -f recipes/deepseek-v4.1-flash/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/deepseek-v4.1-flash/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

Then apply the manifest for your target.

<div data-variant="agg">

```bash
kubectl apply -f recipes/deepseek-v4.1-flash/sglang/agg-gb200/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-variant="disagg">

```bash
kubectl apply -f recipes/deepseek-v4.1-flash/sglang/disagg-gb200/deploy-generic.yaml -n ${NAMESPACE}
```

On Google Kubernetes Engine, apply the RDMA variant instead. It claims the four GKE RDMA networks that GB200 node pools advertise:

```bash
kubectl apply -f recipes/deepseek-v4.1-flash/sglang/disagg-gb200/deploy-gke-rdma.yaml -n ${NAMESPACE}
```

</div>

The first worker start loads 510 GB of weights and captures CUDA graphs. This step can take tens of minutes.

## Smoke Test

Forward the frontend port for your target:

<div data-variant="agg">

```bash
kubectl port-forward svc/deepseek-v41-flash-sglang-gb200-agg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-variant="disagg">

```bash
kubectl port-forward svc/deepseek-v41-flash-sglang-gb200-disagg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":64}'
```

Read the response body, not the status code. On the disaggregated target a KV transfer failure still returns HTTP 200, with `content: null` and zero completion tokens. Readiness does not exercise the KV path. The Notes explain what to read in the worker log.

The model reasons before it answers, and the reasoning text arrives in `reasoning_content`. Both targets support streaming chat, tool calls, and schema-constrained output. The `deepseek_v41` reasoning and tool-call parsers build the response.

## Compare All Targets

| | Aggregated | Disaggregated |
|---|---|---|
| **Checkpoint** | deepseek-ai/DeepSeek-V4.1-Flash | deepseek-ai/DeepSeek-V4.1-Flash |
| **Precision** | FP8 dense, FP4 MoE experts | FP8 dense, FP4 MoE experts |
| **GPUs** | 8x GB200, two workers | 4x GB200 prefill + 4x GB200 decode (1P1D) |
| **Parallelism** | TP4, EP4 | TP4, EP4 per role |
| **Attention backend** | dsv4 | dsv4 |
| **MoE backend** | flashinfer_mxfp4 | flashinfer_mxfp4 |
| **KV cache dtype** | fp8_e4m3 | fp8_e4m3 |
| **Grammar backend** | xgrammar | xgrammar |
| **Page size** | 256 | 256 |
| **GPU memory utilization** | 0.80 | 0.80 |
| **Maximum running requests** | 256 | 256 |
| **Maximum prefill tokens** | 16,384 | 16,384 |
| **Decode CUDA graph batch** | 64 | 512 |
| **Speculative decoding** | DSpark, block size 5 | None |
| **KV transfer** | — | Mooncake over TCP, or RDMA |
| **Routing** | KV-aware | KV-aware |
| **Context length** | 1,048,576 | 1,048,576 |

## Related Feature Benchmarks

No feature benchmark is linked to this recipe. Browse the [Feature Benchmarks](../feature-benchmarks/browse-all-benchmarks.mdx) for comparable KV-routing and disaggregation studies.

## Notes

- The disaggregated target has no speculative decoding. SGLang refuses DSpark under prefill and decode disaggregation for this model. Use the aggregated target for speculative decoding.
- Each worker needs TP4 and a GB200 node holds 4 GPUs. So prefill and decode always sit on different nodes, and KV crosses nodes. The generic target pins Mooncake to TCP with `MC_FORCE_TCP=1`. The GKE variant drops that pin and uses RDMA, because it attaches RDMA devices. Mooncake does not default to TCP on its own: with no RDMA device it reports `Found 0 HCAs` and takes the NVLink fabric, which is why the pin is explicit.
- The NVLink fabric is the faster KV path, and this recipe does not take it. To opt in, remove `MC_FORCE_TCP` and set `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` and `MC_FORCE_MNNVL=true` on both workers. Measured on GB200: TCP across cliques 0.707 GB/s, NVLink inside one clique 1.7 to 2.2 GB/s, so the fabric is about 3 times faster on KV. Read the warning that follows first.

<Warning>
The NVLink fabric KV path needs both workers in one NVLink clique. A fabric memory handle cannot be imported across cliques, and a split gives no error: the deployment reports Ready, and chat completions return HTTP 200 with `content: null` and zero completion tokens. The workers log `cuMemImportFromShareableHandle failed: 400`. Readiness does not exercise the KV path. Pin both workers to one clique with a `nvidia.com/gpu.clique` nodeSelector, read from `kubectl get nodes -L nvidia.com/gpu.clique`, and send one request to confirm the answer is not empty.
</Warning>

- If completions come back empty on the disaggregated target, read the worker log. `cuMemImportFromShareableHandle failed: 400` means prefill and decode landed in different NVLink cliques, and only the fabric KV path fails this way. `Failed to register memory: Bad address [14]` means GPU memory did not register for RDMA, because the nodes load no `nvidia_peermem` module. The GKE target sets `WITH_NVIDIA_PEERMEM=0` for that reason, which selects the DMA-BUF path instead.
- Both targets set the router to KV-aware. On the disaggregated target this is a no-op today, because one decode worker gives the router nothing to choose between. To make it count, raise the decode replicas and add `--kv-events-config` to the decode worker.
- Both targets set `SGLANG_DEFAULT_THINKING=true`. Thinking is off by default for this model, and then `reasoning_content` is empty.
- Both targets keep `--enable-metrics` on and set `DYN_FORWARDPASS_METRIC_PORT` to an empty value. The forward-pass publisher crashes this model on a null `seq_lens_cpu`.
- Both targets pair the native `reasoning-parser` with `--dyn-reasoning-parser`. The native parser gates guided decoding, and Dynamo builds the response. Do not pair the native and Dynamo tool-call parsers. Dynamo rejects that pair.
- The disaggregated target is authored as a Kustomize matrix. Edit `kustomize/base/`, then regenerate with `scripts/kustomize-matrix.py`. Never edit `deploy-generic.yaml` or `deploy-gke-rdma.yaml` by hand.

## Source

- Source README: [recipes/deepseek-v4.1-flash/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4.1-flash/README.md)
- Aggregated GB200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4.1-flash/sglang/agg-gb200/deploy.yaml)
- Disaggregated GB200: [deploy-generic.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4.1-flash/sglang/disagg-gb200/deploy-generic.yaml)
- Disaggregated GB200 on GKE: [deploy-gke-rdma.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4.1-flash/sglang/disagg-gb200/deploy-gke-rdma.yaml)
