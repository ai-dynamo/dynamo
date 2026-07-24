---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Auto Deploy with DGDR
subtitle: Deploy a model by intent — describe the model, workload, and SLA targets, and let Dynamo profile your hardware and generate the DynamoGraphDeployment for you.
---

A **DynamoGraphDeploymentRequest (DGDR)** is Dynamo's deploy-by-intent path. Instead of hand-authoring a [DynamoGraphDeployment (DGD)](dgd-guide.md) with explicit parallelism, replica counts, and resource limits, you describe *what* you want to run — model, backend, workload, and optional latency targets — and Dynamo's profiler analyzes your cluster's GPUs, selects a configuration, and generates the DGD that serves traffic.

This guide walks through authoring that request, starting from the smallest possible DGDR and layering on workload targets, search strategy, hardware sizing, model caching, runtime autoscaling, and review-before-deploy as you need them. Each step builds on the previous one. For the full field table and lifecycle reference, see the [DGDR Reference](dgdr-reference.mdx); for ready-to-copy manifests, see [DGDR Examples](dgdr-examples.md).

> [!NOTE]
> **TODO (author):** Confirm the image-tag convention this guide should teach. The DGDR examples in the repo pin a concrete tag (`nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1`, with `dynamo-frontend` for Dynamo < 1.1.0). The sibling [DGD Guide](dgd-guide.md) uses a `<release-version>` placeholder instead. Pick one convention across both guides and link the release notes.

## When to use DGDR

Reach for a DGDR when you want Dynamo to size the deployment for you. Use a direct [DGD](dgd-guide.md) or a [recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes) when you already know the topology.

| | DGDR (this guide) | DGD ([DGD Guide](dgd-guide.md)) |
|---|---|---|
| **You provide** | Model, backend, workload, hardware, optional SLA targets | Full spec: services, parallelism, replicas, resource limits |
| **What happens** | The profiler generates a DGD; with `autoApply: true` the operator deploys it | The operator reconciles your spec into pods directly |
| **Best for** | New model/hardware combinations, SLA-driven sizing, generated YAML | Known-good configs, tuned recipes, full manual control |
| **Outcome** | Reaches a terminal state after generation/deploy | Persists and serves traffic |

For the full mental model — DGD, DCD, DGDR, recipes, and strategy selection — see the [Deployment Overview](model-deployment-guide.md).

## Prerequisites

Before authoring a DGDR, make sure you have:

- A Kubernetes cluster with the **Dynamo Platform installed**, including the operator and profiler. See the [Installation Guide](installation-guide.md).
- `kubectl` access to that cluster and a target namespace.
- GPU nodes the operator can discover (via DCGM or node labels). The operator auto-detects SKU, VRAM, and GPU count; you can override any of these (see [Step 4](#set-hardware-and-handle-large-or-multinode-models)).
- A **HuggingFace token secret** named `hf-token-secret` in the namespace for gated or rate-limited models — both the profiling job and the deployed pods use it.
- New to Dynamo on Kubernetes? Run one model end to end with the [Kubernetes Quickstart](README.md) first.

For SLA-driven autoscaling, also install [Prometheus](installation-guide.md#kube-prometheus-stack) before creating the DGDR (see [Step 6](#enable-the-planner-for-runtime-autoscaling)).

## How a DGDR is structured

A DGDR is a small intent document. The only required field is `model`; everything else has a default or is auto-detected. The profiler fills in the parts you leave out.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B          # HuggingFace model ID (required)
  backend: auto                   # auto | vllm | sglang | trtllm
  searchStrategy: rapid           # rapid (~30s, simulated) | thorough (2–4h, real GPU)
  autoApply: true                 # deploy the generated DGD automatically
  # workload: { ... }             # expected traffic shape (Step 2)
  # sla: { ... }                  # latency targets (Step 2)
  # hardware: { ... }             # override auto-detected GPUs (Step 4)
  # modelCache: { ... }           # mount cached weights (Step 5)
  # features: { planner: { ... } }# runtime autoscaling (Step 6)
  # overrides: { ... }            # customize the generated DGD (Step 7)
```

The steps below fill in these fields for progressively more demanding deployments. For the complete field reference and defaults, see the [DGDR Reference — Field Reference](dgdr-reference.mdx#spec-reference).

<Steps toc={true} tocDepth={2}>

<Step title="Submit a minimal DGDR">

Start with the smallest request: a model and an image. With `searchStrategy` and `autoApply` at their defaults, the profiler uses rapid simulation (~30 seconds, no GPUs consumed during profiling) and the operator deploys the result.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-small
spec:
  model: Qwen/Qwen3-0.6B
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Apply it and watch the request progress through its phases:

```bash
kubectl apply -f qwen-small.yaml -n <namespace>
kubectl get dgdr qwen-small -n <namespace> -w
```

The DGDR moves through `Pending` → `Profiling` → `Ready` → `Deploying` → `Deployed`. Once it reaches `Deployed`, the generated DGD is running. See [Monitor profiling and deployment](#monitor-profiling-and-deployment) for the full phase list and log commands.

> [!TIP]
> The generated DGD does **not** get an owner reference back to the DGDR. Deleting the DGDR leaves the DGD serving traffic. To tear everything down, delete both (see [Clean up](#clean-up)).

</Step>

<Step title="Describe your workload and SLA targets">

The profiler sizes the deployment against the traffic you expect and the latency you need. Provide a `workload` shape, and optionally `sla` targets, so the profiler optimizes for your case instead of the defaults.

```yaml
spec:
  model: meta-llama/Llama-3.1-8B-Instruct
  backend: vllm
  workload:
    isl: 4000          # average input sequence length (tokens)
    osl: 1000          # average output sequence length (tokens)
    requestRate: 10    # target requests per second
  sla:
    ttft: 500          # target Time To First Token (ms)
    itl: 50            # target Inter-Token Latency (ms)
```

| Field | Meaning | Default |
|---|---|---|
| `workload.isl` | Expected average input sequence length | `4000` |
| `workload.osl` | Expected average output sequence length | `1000` |
| `workload.requestRate` | Target requests per second | — |
| `workload.concurrency` | Target concurrent requests (alternative to `requestRate`) | — |
| `sla.ttft` | Target Time To First Token, ms | — |
| `sla.itl` | Target Inter-Token Latency, ms | — |
| `sla.e2eLatency` | Target end-to-end latency, ms. **Cannot** be combined with `ttft`/`itl`. | — |

> [!NOTE]
> SLA targets shape which configuration the profiler selects. They are also what the [Planner](#enable-the-planner-for-runtime-autoscaling) drives toward at runtime. Set realistic targets — unreachable values push the profiler to its most expensive layout.

</Step>

<Step title="Choose a search strategy">

The `searchStrategy` field controls how the profiler explores configurations. The choice trades profiling time against how close to optimal you land.

**Rapid (default)** uses AIC-backed DynoSim-style performance modeling to search configurations without running real inference. Completes in ~30 seconds with no GPUs consumed during profiling.

```yaml
spec:
  searchStrategy: rapid
```

Use rapid when getting started, iterating quickly, or running in CI/CD — provided your GPU SKU is in the [AIC support matrix](model-deployment-guide.md#dgdr-detail-aic-support-matrix). If AIC does not support your model/hardware/backend combination, the profiler falls back to a naive memory-fit config that may not be optimal.

**Thorough** enumerates candidate parallelization configs, deploys each on real GPUs, and benchmarks them with AIPerf. Takes 2–4 hours and produces measured rather than simulated data.

```yaml
spec:
  searchStrategy: thorough
  backend: vllm      # must specify a concrete backend
```

Use thorough when tuning for production, when your hardware is not supported by AIC (for example PCIe GPUs), or when you need measured performance. It has constraints:

- **Disaggregated mode only** — thorough does not run aggregated configurations.
- **`backend: auto` is rejected** — specify `vllm`, `sglang`, or `trtllm`.
- **Requires GPU resources** — the profiler deploys real inference engines during profiling.

For the profiling algorithms, gate checks, and how to pick a mode, see the [Profiler Guide](../components/profiler/profiler-guide.md).

</Step>

<Step title="Set hardware and handle large or multinode models">

The operator auto-detects GPU SKU, VRAM, and count (capped at 32). Override any of these under `hardware` when auto-detection is wrong or you want the profiler to consider more GPUs.

```yaml
spec:
  hardware:
    gpuSku: h200_sxm   # lowercase underscore format
    totalGpus: 64      # raise above the 32-GPU auto-detect cap
    numGpusPerNode: 8
```

GPU SKUs use **lowercase underscore format** (`h100_sxm`, not `H100-SXM5-80GB`). For the full list of accepted values and the PCIe-support caveat, see the [DGDR Reference — SKU Format](dgdr-reference.mdx#sku-format).

**Large and MoE models that span nodes.** When a model needs more GPUs than one node provides, the deployment is multinode and requires a gang scheduler. Install an orchestrator first — the operator returns a hard error otherwise:

- [Multinode Orchestration](multinode-installation.md) — install-time prerequisites (Grove + KAI, or LWS + Volcano).
- [Grove](grove.md) (default) and [LWS](lws.md) — the two orchestration backends.

For **Mixture-of-Experts (MoE)** models (DeepSeek-R1, Qwen3-MoE), use **SGLang** for full support — vLLM and TensorRT-LLM have partial MoE support still under development. The profiler sweeps MoE models across up to **4 nodes**; beyond that, it selects the best config within range and you may need to adjust replica counts manually. See [Backend Selection](model-deployment-guide.md#production-detail-backend-selection).

> [!NOTE]
> **TODO (author):** Confirm whether DGDR exposes a multinode/`nodeCount` knob directly or whether node spanning is derived from `totalGpus` ÷ `numGpusPerNode` and the selected parallelism. The [DGD Guide](dgd-guide.md) sets `multinode.nodeCount` explicitly; clarify the DGDR equivalent.

</Step>

<Step title="Cache model weights">

By default the profiling job and every generated worker pod download the model from HuggingFace. For large models (>70B) this is slow, and many replicas hit HuggingFace rate limits. Point `modelCache` at a pre-populated `ReadWriteMany` PVC so weights load from shared storage instead.

```yaml
spec:
  model: meta-llama/Llama-3.1-70B-Instruct
  modelCache:
    pvcName: model-cache
    pvcMountPath: /home/dynamo/.cache/huggingface
    pvcModelPath: hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/<commit-hash>
```

The operator mounts the PVC read-only into the profiling job and passes it through to the generated DGD, so both profiling and serving use the cached weights.

`pvcModelPath` must be the HuggingFace snapshot path inside the PVC: `hub/models--<org>--<model>/snapshots/<commit-hash>`. Substitute `/` with `--` in the model ID, and replace `<commit-hash>` with the actual snapshot revision. See [Model Caching — Find the Snapshot Path](model-caching.md#find-the-snapshot-path) for how to look it up.

**Setup:** create a `ReadWriteMany` PVC ([Installation Guide — Shared Storage](installation-guide.md#shared-storage-for-model-caching)), run a one-time download Job to populate it, then reference it here. See [Model Caching](model-caching.md) for the full walkthrough.

For gated models, create the token secret the profiler and pods read automatically:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-token> \
  -n <namespace>
```

</Step>

<Step title="Enable the Planner for runtime autoscaling">

The **Planner** provides runtime autoscaling for disaggregated deployments: it adjusts prefill and decode replica counts to meet your SLA targets as traffic fluctuates. Enable it by setting `features.planner` to a PlannerConfig; DGDR passes the object through to the Planner service and generates Planner support in the final DGD.

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  features:
    planner:
      mode: disagg
      backend: vllm
  sla:
    ttft: 500
    itl: 50
```

To evaluate Planner recommendations without applying scaling changes, add `advisory: true` to the same object.

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      advisory: true
```

The Planner's `sla` optimization target reads live TTFT/ITL from Prometheus, so install [Prometheus](installation-guide.md#kube-prometheus-stack) before creating the DGDR if you want SLA-driven scaling. The `throughput` and `latency` modes use internal queue-depth signals and work without Prometheus. For scaling modes and the full PlannerConfig field reference, see the [Planner Guide](../components/planner/planner-guide.md).

> [!NOTE]
> **TODO (author):** Reconcile the PlannerConfig form. This guide and the [DGDR Reference](dgdr-reference.mdx#planner) use `features.planner.mode`/`backend`/`advisory`; the [Deployment Overview](model-deployment-guide.md#production-detail-planner) shows `features.planner.enabled: true`. Confirm which is current and align both pages (DGDR passes this object through without field-level validation, so the Planner service is the source of truth).

</Step>

<Step title="Review and customize the generated DGD">

For production, inspect the generated DGD before it deploys. Set `autoApply: false` so the DGDR stops at `Ready` and stores the config instead of deploying it.

```yaml
spec:
  autoApply: false
```

After profiling completes, extract, review, and apply the DGD yourself:

```bash
kubectl get dgdr my-model -n <namespace> \
  -o jsonpath='{.status.profilingResults.selectedConfig}' > my-dgd.yaml
# Review and edit my-dgd.yaml, then:
kubectl apply -f my-dgd.yaml -n <namespace>
```

When the generated DGD needs a field DGDR does not expose, supply a partial DGD under `overrides.dgd`. The operator merges it into the profiler-generated deployment after a configuration is selected. For example, to enable KV-aware routing on the generated `Frontend` (which defaults to `round-robin`):

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1   # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
```

DGDR does not yet expose first-class router or EPP/Gateway features, so routing is configured this way. Service names depend on the selected backend and topology — inspect the generated DGD first, especially when `autoApply: false`. For the full routing, worker KV-event, and override patterns, see [DGDR Reference — Generated DGD Overrides](dgdr-reference.mdx#generated-dgd-overrides) and the [Router Guide](../components/router/router-guide.md).

> [!IMPORTANT]
> `overrides.dgd` only patches services that already exist in the generated DGD — it cannot add a missing service (for example an `Epp` service for Gateway routing). For EPP/Gateway topologies, author a direct DGD or use a GAIE recipe; see [Inference Gateway](inference-gateway.md).

</Step>

</Steps>

## Monitor profiling and deployment

A DGDR progresses through these phases. Profiling failures are terminal — they are not retried (`backoffLimit: 0`).

| Phase | What is happening |
|---|---|
| `Pending` | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job running (sub-phases: `Initializing`, `SweepingPrefill`, `SweepingDecode`, `SelectingConfig`, `BuildingCurves`, `GeneratingDGD`, `Done`) |
| `Ready` | Profiling complete; config stored in `.status.profilingResults.selectedConfig`. Terminal when `autoApply: false`. |
| `Deploying` | Creating the DGD (only when `autoApply: true`) |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — check events and conditions |

Watch progress and read profiling logs:

```bash
# Watch phase transitions
kubectl get dgdr my-model -n <namespace> -w

# Detailed status, conditions, and events
kubectl describe dgdr my-model -n <namespace>

# Current profiling sub-phase
kubectl get dgdr my-model -n <namespace> -o jsonpath='{.status.profilingPhase}'

# Profiling job logs
kubectl get pods -n <namespace> -l nvidia.com/dgdr-name=my-model
kubectl logs -f <profiling-pod-name> -n <namespace>
```

For the full lifecycle, conditions, and monitoring command reference, see [DGDR Reference — Lifecycle](dgdr-reference.mdx#lifecycle).

## Troubleshoot

| Symptom | Cause and fix |
|---|---|
| **OOM during profiling or serving** | The model doesn't fit in GPU memory at the selected TP. Raise `hardware.totalGpus`; edge cases (long context, KV overhead) need more than the minimum. |
| **Profiler ignores extra GPUs** | Auto-detection caps at 32. Set `hardware.totalGpus` explicitly. |
| **Profiling job won't schedule** | GPU nodes are tainted. Add tolerations via `overrides.profilingJob` — see [Profiling Job Fails to Schedule](model-deployment-guide.md#profiling-job-fails-to-schedule). |
| **Spec edits rejected** | The DGDR spec is immutable once it enters `Profiling`. Delete and recreate the DGDR. |
| **Multinode deployment errors out** | Grove or LWS is missing. See [Multinode Orchestration](multinode-installation.md). |

## Clean up

Deleting the DGDR does **not** delete the DGD it created — the DGD persists so it can keep serving. To remove both:

```bash
kubectl delete dgdr my-model -n <namespace>
kubectl delete dgd <generated-dgd-name> -n <namespace>
```

> [!NOTE]
> **TODO (author):** Confirm the generated DGD's name pattern (the [Deployment Overview](model-deployment-guide.md#dgd-persists-after-dgdr-deletion) example uses `my-model-dgd`) and how to look it up via the `dgdr.nvidia.com/name` label.

## Next steps

| Goal | Guide |
|---|---|
| Copy-ready DGDR manifests | [DGDR Examples](dgdr-examples.md), [Profiler Examples](../components/profiler/profiler-examples.md) |
| Full field table and lifecycle | [DGDR Reference](dgdr-reference.mdx) |
| Author the deployment by hand instead | [DGD Guide](dgd-guide.md) |
| Profiling algorithms and modes | [Profiler Guide](../components/profiler/profiler-guide.md) |
| Runtime autoscaling details | [Planner Guide](../components/planner/planner-guide.md) |
| Full CRD reference | [API Reference](api-reference.md) |
