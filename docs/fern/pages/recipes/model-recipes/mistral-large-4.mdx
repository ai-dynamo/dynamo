---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Mistral Large 4"
subtitle: "Serve Mistral Large 4 (1051B total, 54B active, NVFP4) with Dynamo and vLLM on GB300 or Vera Rubin."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Each target below is a Dynamo + vLLM deployment of Mistral Large 4, a 1,051B-parameter mixture-of-experts model (54B active) with DSpark speculative decoding and a 1M-token context, served from an NVFP4 checkpoint. Pick your GPU and serving topology; the details below update with the selection.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-gb300" name="recipe-sku" value="gb300" defaultChecked />
<label htmlFor="recipe-sku-gb300">GB300</label>
<input type="radio" id="recipe-sku-vr200" name="recipe-sku" value="vr200" />
<label htmlFor="recipe-sku-vr200">Vera Rubin</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-sku="gb300" data-variant="agg">
<span><b>Precision</b> NVFP4 weights, FP8 KV</span>
<span><b>GPUs</b> 8x GB300</span>
<span><b>Parallelism</b> 2 x TP4 workers</span>
<span><b>Routing</b> KV-aware, hard session affinity</span>
<span><b>Speculative decoding</b> DSpark, 3 draft tokens</span>
<span><b>KV cache</b> Native CPU KV offload (200 GiB per replica), hybrid KV manager off</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="gb300" data-variant="disagg">
<span><b>Precision</b> NVFP4 weights, FP8 KV</span>
<span><b>GPUs</b> 16x GB300 (2P2D)</span>
<span><b>Parallelism</b> TP4 per role, decode expert-parallel</span>
<span><b>Routing</b> KV-aware, hard session affinity</span>
<span><b>Speculative decoding</b> DSpark, 3 draft tokens, both roles</span>
<span><b>KV cache</b> MultiConnector: NIXL + native CPU KV offload, hybrid KV manager off</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="vr200" data-variant="agg">
<span><b>Precision</b> NVFP4 weights, NVFP4 KV</span>
<span><b>GPUs</b> 4x VR200 (1 worker)</span>
<span><b>Parallelism</b> TP4</span>
<span><b>Routing</b> Default router (single worker)</span>
<span><b>Speculative decoding</b> DSpark, 3 draft tokens</span>
<span><b>KV cache</b> Native CPU KV offload, hybrid KV manager off</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="vr200" data-variant="disagg">
<span><b>Precision</b> NVFP4 weights, NVFP4 KV</span>
<span><b>GPUs</b> 8x VR200 (1P1D)</span>
<span><b>Parallelism</b> TP4 per role</span>
<span><b>Routing</b> KV-aware, hard session affinity</span>
<span><b>Speculative decoding</b> DSpark, 3 draft tokens, both roles</span>
<span><b>KV cache</b> MultiConnector: NIXL + CPU KV offload on both roles, hybrid KV manager off</span>
</div>
</div>

## Prerequisites

- A Kubernetes cluster with the Dynamo platform installed (`nvidia.com/v1beta1` operator) and the GPUs for the selected target. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- The NVIDIA DRA driver with ComputeDomain support (multi-node NVLink) on GB300 and Vera Rubin racks.
- A `shared-model-cache` PersistentVolumeClaim that holds the NVFP4 checkpoint (`params.json`, `tekken.json` and all weight shards in one directory). `recipes/mistral-large-4/model-cache/` holds the PVC and a Job that downloads the checkpoint from Hugging Face into `/shared-model-cache/mistral-large-4-nvfp4`; the Job pulls `mistralai/Mistral-Large-4-1T-A52B-NVFP4`, so create an `hf-token-secret` first.

## Deploy

Apply the manifest for the selected target:

<div data-sku="gb300" data-variant="agg">

```bash
kubectl apply -f recipes/mistral-large-4/vllm/agg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="disagg">

```bash
kubectl apply -f recipes/mistral-large-4/vllm/disagg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="vr200" data-variant="agg">

```bash
kubectl apply -f recipes/mistral-large-4/vllm/agg-vr-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="vr200" data-variant="disagg">

```bash
kubectl apply -f recipes/mistral-large-4/vllm/disagg-vr-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

The first launch loads the weights and, on the Vera Rubin targets, compiles the MoE kernels at first use; give it tens of minutes. Keep the model-cache and JIT-cache mounts persistent across restarts.

The recipes run with the vLLM hybrid KV cache manager off.

The published `deploy.yaml` recipes are generic base recipes agnostic to cluster-specific settings. Adapt to your cluster scheduling, tolerations, and networking by using [kustomizations](https://github.com/ai-dynamo/dynamo/tree/main/recipes/templates/kustomize).

## Smoke Test

Forward the frontend and send a request. All recipes serve the model as `mistral-large-4`:

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral-large-4","messages":[{"role":"user","content":"Write a one-sentence readiness check."}],"max_tokens":64}'
```

Tool calling and reasoning use the Mistral parsers on the workers and the Dynamo chat processor on the frontend.

## Runtime image

All targets use `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0-mistral-large-4-dev.1`, with `runtimeVersionOverride: "1.6.0"`. It carries the released `mistral-common` 1.13.0 that the checkpoint's `tekken.json` reasoning-effort levels require and, for Vera Rubin, the sm_107 FlashInfer MoE and sparse-indexer kernels.

<div data-sku="vr200">

## Vera Rubin settings

- **KV cache.** `nvfp4_ds_mla` stores the latent KV in NVFP4; it changes numerics, so re-run an accuracy check after any model or image change. Both roles of a disaggregated deployment must use the same dtype.
- **Host KV tier.** Aggregated uses `--kv-offloading-size`. Disaggregated wraps the `OffloadingConnector` in a `MultiConnector` next to NIXL on both roles; setting only `--kv-offloading-size` replaces NIXL and silently disables the handoff.
- **Disaggregated transfer.** `UCX_CUDA_IPC_ENABLE_MNNVL=y` with `UCX_TLS=cuda_copy,cuda_ipc,tcp` moves KV over NVLink; place both workers in one NVL72 rack. `NIXL_TELEMETRY_ENABLE=y` is required because the operator injects `n`, which makes some vLLM versions fail every transfer.
- **Stability.** `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `DYN_HEALTH_CHECK_ENABLED=false` (the health canary can time out under very long prefills) and `disable_eagle_block_drop` (DSpark counts as eagle and would otherwise drop the trailing prefix block).
- **Tool calls.** Workers run `--dyn-tool-call-parser mistral --dyn-reasoning-parser mistral` and the frontend sets `DYN_PARSER_VERSION=2`; keep the Dynamo chat processor on the disaggregated frontend, which also uses `fastokens`.

</div>

<div data-sku="gb300">

## GB300 settings

- **KV cache dtype.** FP8 latent KV, which is the checkpoint default (`dsa.fp8_kv_cache` in `params.json`); the GB300 recipes do not set `--kv-cache-dtype`.
- **Hybrid KV cache manager off.** Both topologies run `--disable-hybrid-kv-cache-manager`. The DSpark draft layers use sliding-window MLA attention; with the manager on, the target and draft layers form two KV cache groups and a prefix hit needs both. With it off they share one group, so a continuation turn can reuse the cached prefix.
- **Host KV tier.** Aggregated workers use native CPU offload (`--kv-offloading-size 200`, per replica). Disaggregated workers wrap the `OffloadingConnector` in a `MultiConnector` next to NIXL on both roles; setting only `--kv-offloading-size` would replace NIXL and silently disable the handoff.
- **Disaggregated transfer.** Prefill is the NIXL producer and decode the consumer, over UCX with `UCX_CUDA_IPC_ENABLE_MNNVL=yes`, so KV moves over NVLink inside one NVL72 rack. The deployment creates one `ComputeDomain` (`numNodes: 0`) whose channel claim every worker requests. Prefill and decode must use the same block size (64) and speculative configuration.
- **Roles.** Prefill runs without CUDA graphs and with synchronous scheduling; decode adds expert parallelism and chunked prefill with a small batched-token budget.
- **Resources.** Aggregated workers request 4 GPUs, 64 CPUs and 704 GiB of memory with 264 GiB of shared memory. Disaggregated workers request 4 GPUs, 64 CPUs and 840 GiB with 640 GiB of shared memory. Shared memory and offload both draw on the worker's host memory budget.
- **Routing and parsing.** The frontend uses KV routing with hard session affinity, the Dynamo chat processor with `DYN_PARSER_VERSION=2` and `--tokenizer fastokens`. Workers run `--dyn-tool-call-parser mistral --dyn-reasoning-parser mistral`.

</div>

## Notes

- **Session affinity.** Clients send `x-dynamo-session-id` so the KV router can pin a conversation to a worker. Keep the Dynamo chat processor; the vLLM processor drops this header.
- **Speculative decoding.** The recipes use real DSpark acceptance. A synthetic-acceptance alternative is commented next to the flag for performance benchmarking only; never use it for accuracy evaluation. On disaggregated deployments, set it on both roles or neither.
- **Disaggregated KV transfer.** Prefill and decode must share block size and speculative configuration.
- **Concurrent deployments.** Never run two `DynamoGraphDeployment`s in one namespace where one name is a prefix of the other; the frontend cross-routes requests between them.

## Benchmark

`recipes/mistral-large-4/perf/` holds an AIPerf Job that replays the AgentX agentic trace (`inferencex-agentx-mvp`) against the deployment with a 60-minute measured window after the warmup phase (allowed up to 30 minutes). Set `ENDPOINT`, `CONCURRENCY` and the `podAffinity` DGD name for the deployed recipe, then apply it:

```bash
kubectl apply -f recipes/mistral-large-4/perf/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/mistral-large-4-agentx-aiperf -n ${NAMESPACE}
```

Run several concurrency values; the shipped gate is decode tok/s/user P50 above 50 and TTFT P50 below 2 seconds. See the [perf README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/mistral-large-4/perf/README.md) for artifacts and options.


## Source

- Model recipe: [recipes/mistral-large-4](https://github.com/ai-dynamo/dynamo/tree/main/recipes/mistral-large-4)

<div data-sku="gb300" data-variant="agg">

- Selected recipe: [GB300 agg deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/mistral-large-4/vllm/agg-gb300-agentic/deploy.yaml)

</div>

<div data-sku="gb300" data-variant="disagg">

- Selected recipe: [GB300 disagg deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/mistral-large-4/vllm/disagg-gb300-agentic/deploy.yaml)

</div>

<div data-sku="vr200" data-variant="agg">

- Selected recipe: [Vera Rubin (VR200) agg deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/mistral-large-4/vllm/agg-vr-agentic/deploy.yaml)

</div>

<div data-sku="vr200" data-variant="disagg">

- Selected recipe: [Vera Rubin (VR200) disagg deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/mistral-large-4/vllm/disagg-vr-agentic/deploy.yaml)

</div>
