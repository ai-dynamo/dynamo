---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy with DGD
subtitle: Author a DynamoGraphDeployment spec to deploy a model on Kubernetes, from a minimal aggregated graph to multinode, cached, and disaggregated serving.
---

A **DynamoGraphDeployment (DGD)** is the Kubernetes Custom Resource (CRD) that describes how to deploy a model with Dynamo. You write the spec, `kubectl apply` it, and the Dynamo operator reconciles it into the running pods, services, and scheduling resources that serve your model. One DGD describes one inference graph: a Frontend plus one or more workers.

This guide walks through authoring that spec, starting from the simplest aggregated deployment and layering on parallelism, disaggregation, multinode, and model caching as you need them. Each step builds on the previous one.

> [!NOTE]
> **TODO (author):** Confirm the API version this guide should teach. Examples in the repo use both `nvidia.com/v1alpha1` (the storage version, used by most current examples and the Quickstart) and `nvidia.com/v1beta1` (the newer list-based API: `spec.components` instead of `spec.services`, `podTemplate` instead of `extraPodSpec`). This scaffold uses **v1alpha1** to match the Quickstart. Decide whether to teach v1beta1 as primary and either migrate the examples or add a migration note.

## Prerequisites

Before authoring a DGD, make sure you have:

- A Kubernetes cluster with the **Dynamo Platform installed**. See the [Installation Guide](installation-guide.md).
- `kubectl` access to that cluster and a target namespace.
- A **HuggingFace token secret** in that namespace for gated or rate-limited models (referenced below as `hf-token-secret`).
- New to Dynamo on Kubernetes? Run one model end to end with the [Kubernetes Quickstart](README.md) first, then come back here to author your own spec.

For the concepts behind the CRDs and the operator, see the [Deployment Overview](model-deployment-guide.md) and the [API Reference](api-reference.md).

## How a DGD is structured

Every DGD has the same top-level shape. The spec is mostly a map of **services**, where each entry is one Dynamo component:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    Frontend:          # the OpenAI-compatible API gateway
      componentType: frontend
      # ...
    MyWorker:          # one or more inference workers
      componentType: worker
      # ...
```

- `componentType: frontend` is the HTTP entry point — see [Frontend](../components/frontend/README.md).
- `componentType: worker` runs the inference engine (vLLM, SGLang, or TensorRT-LLM).
- Per-service fields you will use most: `replicas`, `resources` (CPU/memory/GPU), `envFromSecret`, `extraPodSpec.mainContainer` (image and the `command`/`args` that launch the engine), `multinode`, and `volumeMounts`.

The steps below fill in these fields for progressively more capable deployments.

<Steps toc={true} tocDepth={2}>

<Step title="Author a minimal aggregated deployment">

Start with the simplest possible graph: one **Frontend** and one **worker** that handles both prefill and decode (this is *aggregated* serving). Use a small model so it fits on a single GPU with `--tensor-parallel-size 1`.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<release-version>
    VllmWorker:
      componentType: worker
      replicas: 1
      envFromSecret: hf-token-secret
      resources:
        requests:
          gpu: "1"
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<release-version>
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

Apply it and watch the pods come up:

```bash
kubectl apply -f vllm-agg.yaml -n <namespace>
kubectl get pods -n <namespace> -w
```

Once the Frontend and worker are `Running`, port-forward and send a request:

```bash
kubectl port-forward svc/vllm-agg-frontend 8000:8000 -n <namespace>
```

> [!NOTE]
> **TODO (author):** Add the `curl` request against `/v1/chat/completions` and the expected response shape. Pin `<release-version>` to a real release tag and link to the release notes, matching the version convention in the [Installation Guide](installation-guide.md). Confirm the Frontend service name pattern (`<name>-frontend`).

> [!TIP]
> `envFromSecret` injects your HuggingFace token into the worker so it can pull gated or rate-limited weights. See [Step 6](#cache-model-weights) for avoiding per-pod downloads entirely.

</Step>

<Step title="Size your workers: parallelism and GPU resources">

Your worker's parallelism determines how many GPUs it needs and how it splits the model. Dynamo passes these as **engine CLI arguments** in the worker `command`/`args` — they are not dedicated CRD fields — and the worker's GPU `limits` must match the product of the parallel sizes.

| Argument | What it does | Rule of thumb |
|---|---|---|
| `--tensor-parallel-size` (TP) | Splits each layer's tensors across GPUs | Keep **within a single node** (uses NVLink); size up to fit weights + KV cache in GPU memory |
| `--pipeline-parallel-size` (PP) | Splits layers into stages across GPUs | Use **across nodes** when one node can't hold the model |
| `--data-parallel-size` (DP) | Replicates the model for more throughput | Scale out once a single replica meets latency |

The GPU count must equal **TP × PP per node**. For a TP-2 worker:

```yaml
    VllmWorker:
      componentType: worker
      resources:
        limits:
          gpu: "2"          # must equal TP × PP
      extraPodSpec:
        mainContainer:
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.vllm \
                --model Qwen/Qwen3-32B \
                --tensor-parallel-size 2
```

**Do you need the profiler?** No — for a first deployment, pick TP to fit the model in GPU memory and move on. When you need to hit a specific latency SLA (TTFT/ITL) at minimum cost, Dynamo can pick TP/PP/DP and replica counts for you:

- [Sizing with AIConfigurator](../features/disaggregated-serving/aiconfigurator.md) — a standalone CLI that sweeps layouts and emits ready-to-use DGD YAML. Run it *before* authoring the spec.
- [Profiler Guide](../components/profiler/profiler-guide.md) — an in-cluster job that profiles your model on your hardware and produces a deployable DGD.

> [!NOTE]
> **TODO (author):** Confirm the exact arg names per backend (vLLM/SGLang use CLI flags; TensorRT-LLM sets parallelism in an engine config). Add a short note on KV-cache headroom vs. TP, and link [Performance Tuning](../performance/tuning.md) for the GPU-count tradeoff table.

</Step>

<Step title="Choose aggregated or disaggregated serving">

So far the worker does both prefill and decode (**aggregated**). **Disaggregated** serving splits these into separate prefill and decode workers, each sized and scaled independently, with the KV cache transferred between them over the network.

Use aggregated when you want the simplest deployment or have uniform traffic. Move to disaggregated when prefill and decode have different bottlenecks — long prompts saturating prefill while decode sits idle, or vice versa. See [Disaggregated Serving](../features/disaggregated-serving/README.md) for the full tradeoff and [the design doc](../design-docs/disagg-serving.md) for depth.

A disaggregated graph adds a second worker and tags each with its role:

```yaml
  services:
    Frontend:
      componentType: frontend
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      # ... resources + command with prefill-mode flags
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      # ... resources + command with decode-mode flags
```

> [!IMPORTANT]
> Disaggregated serving moves KV cache between workers over the network. For acceptable performance, the cluster needs RDMA — see the [Disaggregated Communication Guide](disagg-communication-guide.md). Without it, transfers fall back to TCP with severe latency penalties.

> [!NOTE]
> **TODO (author):** Fill in the full prefill/decode worker `command` blocks (the disaggregation-mode flags per backend) and note that disaggregated graphs require Grove or LWS (see the next step). Reference a complete example under `examples/backends/vllm/deploy/`.

</Step>

<Step title="Optional: Deploy across multiple nodes">

When a worker needs more GPUs than a single node provides — a large model that can't fit with TP alone — set `multinode.nodeCount`. The operator spreads the worker's leader and worker pods across that many nodes using Grove or LeaderWorkerSet; you do **not** author the leader/worker pods yourself.

```yaml
    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      multinode:
        nodeCount: 2          # total GPUs = nodeCount × per-node gpu limit
      resources:
        limits:
          gpu: "8"            # per node
      extraPodSpec:
        mainContainer:
          command:
            - /bin/bash
            - -c
            - |
              exec python3 -m dynamo.vllm \
                --model <large-model> \
                --tensor-parallel-size 8 \
                --pipeline-parallel-size 2
```

Multinode deployments require a gang scheduler. Install and choose an orchestrator first:

- [Multinode Orchestration](multinode-installation.md) — install-time prerequisites (Grove + KAI, or LWS + Volcano).
- [Multinode Deployments](deployment/multinode-deployment.md) — how to author and operate them.
- [Grove](grove.md) (default) and [LWS](lws.md) — the two orchestration backends.

> [!NOTE]
> **TODO (author):** Confirm how TP/PP map onto `nodeCount` (e.g. TP within node, PP across the `nodeCount` nodes) and add a worked example tying `gpu` limit, `nodeCount`, and the parallel sizes together.

</Step>

<Step title="Optional: Cache model weights">

By default each worker pod downloads the model from HuggingFace on startup. For large models (>70B) this is slow per pod, and many replicas will hit HuggingFace rate limits. Mount a shared volume so the weights download once and every pod reads from cache.

The pattern: declare a `pvc`, run a one-time download job, then mount it into your workers with `volumeMounts`.

```yaml
spec:
  pvcs:
    - name: model-cache
      create: true
      size: 100Gi
  services:
    VllmWorker:
      # ...
      volumeMounts:
        - name: model-cache
          mountPoint: /models
```

See [Model Caching](model-caching.md) for the full walkthrough (download job, PVC sizing, access modes), [ModelExpress](modelexpress.md) for faster RDMA-based weight distribution at scale, and [Model Caching with Fluid](model-caching-with-fluid.md) for the Fluid-based alternative.

> [!NOTE]
> **TODO (author):** Add the model-download job manifest (or link the exact section in [Model Caching](model-caching.md)), and show the worker `--model` path pointing at the mounted cache instead of the HuggingFace ID.

</Step>

<Step title="Optional: Add routing, offloading, and autoscaling">

These are optional capabilities you opt into based on your workload. None are required for a working deployment.

- **KV-aware routing** — route requests to the worker most likely to have the prompt's prefix already cached, improving TTFT and throughput. Enable it on the Frontend/router; see the [Router Guide](../components/router/router-guide.md) and [Routing Concepts](../components/router/router-concepts.md). For disaggregated graphs, see [Router with Disaggregated Serving](../components/router/router-disaggregated-serving.md).
- **KV cache offloading** — spill KV blocks to host memory or disk to serve longer contexts and reuse cache across requests. See the [KVBM Guide](../components/kvbm/kvbm-guide.md), plus the [LMCache](../integrations/lmcache-integration.md) and [FlexKV](../integrations/flexkv-integration.md) integrations.

> [!IMPORTANT]
> **The Planner is not part of the DGD spec.** Autoscaling with the [Planner](../components/planner/README.md) is configured through a **DynamoGraphDeploymentRequest (DGDR)**, not a DGD — see the [Planner Guide](../components/planner/planner-guide.md) and the [DGDR Guide](dgdr-guide.md). If you need SLA-driven scaling, author a DGDR instead of (or in addition to) the DGD here. For simpler scaling, see [Autoscaling](autoscaling.md).

> [!NOTE]
> **TODO (author):** Decide how much of routing/offloading belongs in this authoring guide vs. a cross-link. Show the one or two `command`/spec lines that turn KV routing on, rather than duplicating the Router Guide.

</Step>

</Steps>

## Validate your spec

Before applying, check that the spec is well-formed and the operator accepts it:

```bash
kubectl apply -f my-deployment.yaml -n <namespace> --dry-run=server
```

> [!NOTE]
> **TODO (author):** Add validation tips — common operator rejection reasons (multinode without Grove/LWS, GPU limit not matching TP×PP, missing secret), and how to read `kubectl describe dynamographdeployment` / operator events.

## Next steps

| Goal | Guide |
|---|---|
| SLA-driven autoscaling | [Planner Guide](../components/planner/planner-guide.md), [DGDR Guide](dgdr-guide.md) |
| Auto-pick parallelism for a latency target | [Sizing with AIConfigurator](../features/disaggregated-serving/aiconfigurator.md), [Profiler Guide](../components/profiler/profiler-guide.md) |
| Full field reference | [API Reference](api-reference.md) |
| Operate the deployment | [Autoscaling](autoscaling.md), [Rolling Update](rolling-update.md), [Observability Metrics](observability/metrics.md) |
