<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash aggregated serving on B200 (vLLM)

Serves `deepseek-ai/DeepSeek-V4.1-Flash` through the Dynamo KV-aware router on
8 B200 GPUs: two TP4 workers, each with expert parallelism and DSpark
speculative decoding, up to 1,048,576 tokens of context.

| Setting | Value |
| --- | --- |
| Workers | 2 x TP4 |
| GPUs | 8x B200 |
| Precision | MXFP4 MoE experts, MXFP4 sparse-indexer KV |
| MoE backend | `deep_gemm_mega_moe` with expert parallelism and EPLB |
| Speculation | DSpark, 3 draft tokens, adaptive verification off |
| Routing | KV-aware, with decode-load balancing |
| Context | Up to 1,048,576 tokens |

This is the same serving configuration as the [GB200 aggregated
recipe](../agg-gb200/): only the architecture label, the GPU product label, and
the deployment name differ. The GB200 recipe measures higher on identical
settings, and the cause is KV capacity rather than compute — a B200 TP4 worker
holds roughly 19 percent fewer KV tokens, and prefix-cache hit rate follows.

Aggregated serving wins decisively on time to first token. The
[disaggregated recipe](../disagg-b200/) wins on throughput per GPU and on the
inter-token latency tail, but needs 12 GPUs and pays a large TTFT penalty. The
two are not interchangeable; pick by the TTFT your workload accepts.

## Deploy

The namespace needs the `shared-model-cache` PVC populated with the pinned
model snapshot — see [`../../model-cache/`](../../model-cache/). Eight B200
GPUs in two groups of four must be schedulable, with 512 GiB of host memory per
worker. Set `CONTEXT` and `NAMESPACE`, then run from this directory:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f deploy-generic.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv41-flash-vllm-b200-agg \
  --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  service/dsv41-flash-vllm-b200-agg-frontend 8000:8000
```

> [!IMPORTANT]
> First boot takes roughly an hour. The checkpoint is 510 GB over 48 shards,
> and a large silent host-memory load runs with the GPUs at 0 percent, so every
> operator signal during that window looks like a hang. The startup probe
> allows 60 minutes for exactly this reason.

Call `/v1/models` and `/v1/chat/completions` through the forwarded port with
model `deepseek-ai/DeepSeek-V4.1-Flash`. Confirm both workers receive requests
before benchmarking.

## Benchmark

See [`../../perf/README.md`](../../perf/README.md). The validated operating
point for this variant is concurrency 88.

## Edit and render

Edit `kustomize/base/deploy.yaml`; `deploy-generic.yaml` is generated. From the
repository root, regenerate it with:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/deepseek-v4.1-flash/vllm/agg-b200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/deepseek-v4.1-flash/vllm/agg-b200/.kustomize-matrix.yaml
```

The same configuration can be applied through the generated overlay:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -k kustomize/overlays/generic
```

To compose extra Components without checking in an overlay, use
`scripts/kustomize-matrix.py compose`.
