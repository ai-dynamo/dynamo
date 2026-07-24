---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Disaggregated Serving
subtitle: Write a DynamoGraphDeployment that splits prefill and decode into independent worker pools
---

Disaggregated serving separates the two phases of LLM inference into their own
worker pools so you can scale, size, and place them independently:

| Phase | What it does | Scaling pressure |
|---|---|---|
| Prefill | Processes the prompt and produces the initial KV cache. | Input length, prompt reuse, context size |
| Decode | Generates output tokens using the KV cache. | Concurrency, output length, active KV memory |

In an **aggregated** deployment, one worker does both phases. In a
**disaggregated** deployment, prefill and decode are separate pools on separate
GPUs: Dynamo routes each request through prefill first, transfers the KV cache to
the decode worker, then streams the response from decode.

Structurally, that difference is one worker versus two. The rest of the
`DynamoGraphDeployment` (DGD) is the same:

<Tabs>
<Tab title="Aggregated (1 worker)">

```yaml
  services:
    Frontend:
      componentType: frontend
    VLLMWorker:
      componentType: worker
      # one worker does both prefill and decode
```

</Tab>
<Tab title="Disaggregated (prefill + decode)">

```yaml
  services:
    Frontend:
      componentType: frontend
    VLLMPrefillWorker:
      componentType: worker
      subComponentType: prefill   # prompt processing only
    VLLMDecodeWorker:
      componentType: worker
      subComponentType: decode    # token generation only
```

</Tab>
</Tabs>

This guide walks through writing the disaggregated spec end to end. If you have
not authored a DGD before, read the [Deploy with DGD](../../kubernetes/dgd-guide.md)
guide first — this page assumes you know the overall shape and focuses on what
disaggregation adds.

## Should you disaggregate?

Disaggregation helps most when prefill and decode need different resource shapes:

- long prompts or retrieval-heavy traffic make prefill expensive
- long generations or high concurrency make decode the bottleneck
- you want to scale prefill and decode replicas independently
- large models need different parallelism for prompt processing and generation

It is not automatically better. For small models, short prompts, low concurrency,
or clusters without a fast KV-transfer fabric, an aggregated deployment is
simpler and often faster. Use [Sizing with AIConfigurator](aiconfigurator.md) to
compare aggregated vs. disaggregated layouts before committing.

This guide uses a **single-node, multi-GPU** example — prefill and decode run on
separate GPUs in the same node (for instance, a node with 8×H100 serving
Qwen3-32B, one GPU per worker). On a single node the KV cache moves GPU-to-GPU
over NVLink, so there is nothing extra to configure for the transport. Scaling
prefill and decode across **multiple nodes** adds RDMA networking — see
[Enable multi-node transfer](#enable-multi-node-transfer) at the end.

## Write the deployment

<Steps toc={true} tocDepth={2}>

<Step title="Check prerequisites">

For the single-node example in this guide you need:

1. **A node with multiple GPUs** — one per prefill and decode worker (this
   example uses 8×H100 with `gpu: "1"` per worker).
2. **ETCD and NATS** deployed for Dynamo coordination.
3. **A HuggingFace token secret** (`hf-token-secret`) if the model is gated.

Model caching (the PVC and `HF_HOME` wiring the workers mount) is configured
separately — see [Model caching](../../kubernetes/dgd-kv-offloading.md). This
guide omits those fields for clarity; without them the model downloads to the
container's default cache.

Scaling across multiple nodes additionally requires an RDMA-capable network and
device plugin — covered in [Enable multi-node transfer](#enable-multi-node-transfer).

</Step>

<Step title="Start the spec: the frontend">

Begin with the DGD skeleton — just a frontend. This is identical to an
aggregated deployment; nothing about disaggregation changes it.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-disagg
  namespace: your-namespace
spec:
  backendFramework: vllm
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          imagePullPolicy: IfNotPresent
```

</Step>

<Step title="Add the prefill worker">

The prefill worker runs the prompt phase. Two things make it a prefill worker
rather than a plain worker:

- `subComponentType: prefill` tags the role for the operator and router.
- `--disaggregation-mode prefill` tells the engine to stop after producing the
  KV cache instead of generating tokens.

```yaml
    VLLMPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 3
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 16Gi
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          imagePullPolicy: IfNotPresent
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - Qwen/Qwen3-32B
            - --disaggregation-mode
            - prefill
```

Qwen3-32B fits on a single H100, so each worker takes `gpu: "1"` with no tensor
parallelism. `replicas: 3` gives three prefill workers; size the count for your
prompt load independently of decode. (`sharedMemory` is explained in the KV
transfer step.)

</Step>

<Step title="Add the decode worker">

The decode worker continues generation after prefill hands off the KV cache. It
mirrors the prefill worker with `subComponentType: decode` and
`--disaggregation-mode decode`, and its replica count is sized for concurrency
rather than prompt load.

```yaml
    VLLMDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 4
      resources:
        limits:
          gpu: "1"
      sharedMemory:
        size: 16Gi
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          imagePullPolicy: IfNotPresent
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - Qwen/Qwen3-32B
            - --disaggregation-mode
            - decode
```

<Warning>
Prefill and decode workers must pass the **same `--model`** (and, if you set
them, the same dtype, block size, and KV layout). A mismatch means the KV cache
prefill produces cannot be consumed by decode — you get transfer errors or
silently corrupt output.
</Warning>

</Step>

<Step title="Enable KV transfer">

Because prefill and decode run on **different GPUs**, the KV cache prefill
produces has to be copied into the decode worker's GPU memory before decode can
generate. Dynamo's transfer layer, **NIXL**, does this automatically. On a
single node it selects **CUDA IPC** and moves the cache GPU-to-GPU over NVLink —
no network, no RDMA, nothing to configure for the transport itself.

The one field you do set is shared memory, already in the worker specs above:

| Setting | Why |
|---|---|
| `sharedMemory.size: 16Gi` | NIXL stages transfer metadata through the pod's `/dev/shm`. The pod default (64 MB) is too small and NIXL fails to initialize. `16Gi` is comfortable headroom, not a tightly-computed per-request budget — it does not consume that memory unless needed. |

That is the entire KV-transfer configuration for a single node: two workers
tagged prefill/decode, matching `--model`, and enough shared memory for NIXL.

</Step>

<Step title="Apply and verify the transfer path">

Apply the assembled spec:

```bash
kubectl apply -f dynamo-disagg.yaml -n your-namespace
```

Then confirm NIXL initialized its transfer backend — the key check for a
disaggregated deployment. Grep a prefill worker's logs:

```bash
kubectl logs <prefill-worker-pod> | grep -i "NIXL"
```

You want to see NIXL bring up a backend, for example:

```text
NIXL INFO Backend UCX was instantiated
```

If NIXL fails to initialize, the usual cause is `sharedMemory.size` left at the
default. Symptoms of a broken transfer path include high TTFT despite free
prefill capacity, decode workers sitting idle while prefill is busy, or
disaggregated throughput falling below your aggregated baseline.

</Step>

<Step title="Enable multi-node transfer">

Everything above keeps prefill and decode on **one node**, where the KV cache
moves over NVLink. When a worker's parallelism exceeds the GPUs on a node, or you
want prefill and decode pools on separate machines, the KV cache travels over the
**network** instead — and that path needs **RDMA** (InfiniBand or RoCE). Without
it, transfers fall back to TCP and KV movement can dominate TTFT and throughput.

Multi-node adds RDMA fields to each worker (`rdma/ib` resource requests, the
`IPC_LOCK` capability, and `UCX_*` transport env vars) plus an RDMA device plugin
on the cluster. That setup is out of scope here — see the
[Disaggregated Communication Guide](../../kubernetes/disagg-communication-guide.md)
for the transport configuration and [Multinode Deployments](../../kubernetes/deployment/multinode-deployment.md)
for spanning workers across machines.

</Step>

</Steps>

## Faster starting points

Typing the full spec from scratch is rarely the fastest path. Copy a validated
template and adapt it:

| Starting point | Use when |
|---|---|
| [Dynamo Recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) | A recipe matches your model, backend, and hardware. Best for validated baselines and `perf.yaml` benchmarks. |
| Backend `disagg.yaml` templates | You want a working DGD to adapt. Each backend ships `disagg.yaml`, `disagg_router.yaml`, and `disagg_planner.yaml` under its `deploy/` folder. |
| [DGDR](../../kubernetes/dgdr-reference.mdx) | You want Dynamo to generate a DGD from model, backend, hardware, and SLA intent. |
| [Sizing with AIConfigurator](aiconfigurator.md) | You want to compare aggregated vs. disaggregated layouts and estimate prefill/decode sizing first. |

Good recipe starting points:

- [Qwen3-32B vLLM disagg + KV router](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b)
- [DeepSeek V3.2 TensorRT-LLM disagg + KV router](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v32-fp4)
- [Llama 3 70B vLLM disaggregated recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes/llama-3-70b)

Backend deployment examples with concrete worker flags:

| Backend | Examples |
|---|---|
| vLLM | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy) — `disagg.yaml`, `disagg_router.yaml`, `disagg_planner.yaml` |
| TensorRT-LLM | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy) — disaggregated, router, and planner variants |
| SGLang | [Deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy) — NIXL-based disaggregated serving |

## How disaggregation relates to KV-aware routing

These are separate features that pair well. Disaggregation splits prefill and
decode. KV-aware routing chooses workers based on cache locality. Many
production deployments use both, but you can reason about them independently.

For router behavior, see [Router: Disaggregated Serving](../../components/router/router-disaggregated-serving.md)
and [KV Cache Aware Routing](../../components/router/router-guide.md).

## Before production

- Confirm NIXL initialized on each worker (the log check above).
- Confirm prefill and decode workers agree on model, dtype, block size, and KV layout.
- Confirm pods have the required GPU and shared memory.
- Confirm frontend/router flags match your routing strategy.
- Run benchmarks inside the cluster, not through local port-forwarding.
- If you scale across nodes, validate the RDMA transfer path — see
  [Enable multi-node transfer](#enable-multi-node-transfer).

Use [Dynamo Benchmarking](../../benchmarks/benchmarking.md) to compare
aggregated and disaggregated configurations under the same workload.

## Next steps

1. Start from a matching [Dynamo Recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes) when one exists.
2. Read the backend-specific `disagg.yaml` for your engine.
3. Use [Sizing with AIConfigurator](aiconfigurator.md) or DGDR to choose prefill/decode sizing.
4. Validate with [Dynamo Benchmarking](../../benchmarks/benchmarking.md).
