---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Auto Deploy with DGDR
subtitle: Deploy a model by intent — describe the model, workload, and SLA targets, and let Dynamo profile your hardware and generate the DynamoGraphDeployment for you.
---

A **DynamoGraphDeploymentRequest (DGDR)** is Dynamo's deploy-by-intent path. Instead of hand-authoring a [DynamoGraphDeployment (DGD)](../model-deployment/deploy-with-dgd.md) with explicit parallelism, replica counts, and resource limits, you describe *what* you want to run — model, backend, workload, and optional latency targets — and Dynamo's profiler analyzes your cluster's GPUs, selects a configuration, and generates the DGD that serves traffic.

This guide walks through authoring that request, starting from the smallest possible DGDR and layering on workload targets, search strategy, hardware sizing, model caching, runtime autoscaling, and review-before-deploy as you need them. Each step builds on the previous one. For the component relationships and guidance on choosing DGDR, see the [Auto Deployment overview](overview.mdx). For the full field table and lifecycle reference, see the [DGDR Reference](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx); for ready-to-copy manifests, see [DGDR Templates](../../recipes/kubernetes-templates/dgdr.mdx).

<Note>
In a release installation, when you omit `spec.image`, the DGDR webhook selects
`nvcr.io/nvidia/ai-dynamo/dynamo-planner:<operator-version>`. For a local
operator build without a known version, set `spec.image` explicitly. Use the
`dynamo-planner` image from the same release as the operator. For Dynamo
releases earlier than 1.1.0, use the matching `dynamo-frontend` image. The
generated Planner and backend images retain that registry and tag.
</Note>

## Prerequisites

Before authoring a DGDR, make sure you have:

- A Kubernetes cluster with the **Dynamo Platform installed**, including the operator and profiler. See the [Installation Guide](../installation/install-dynamo.md).
- `kubectl` access to that cluster and a target namespace.
- GPU nodes the operator can discover (via DCGM or node labels). The operator auto-detects SKU, VRAM, and GPU count; you can override any of these (see [Step 4](#set-hardware-and-handle-large-or-multinode-models)).
- A **HuggingFace token secret** named `hf-token-secret` in the namespace for gated or rate-limited models — both the profiling job and the deployed pods use it.
- New to Dynamo on Kubernetes? Run one model end to end with the [Kubernetes Quickstart](../getting-started/quickstart.mdx) first.

For SLA-driven autoscaling, also install [Prometheus](../installation/install-dynamo.md#kube-prometheus-stack) before creating the DGDR (see [Step 6](#enable-the-planner-for-runtime-autoscaling)).

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

The steps below fill in these fields for progressively more demanding deployments. For the complete field reference and defaults, see the [DGDR Reference — Field Reference](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#spec-reference).

<Steps toc={true} tocDepth={2}>

<Step title="Submit a minimal DGDR">

Start with the smallest request: a model. With `searchStrategy` and `autoApply` at their defaults, the profiler uses rapid simulation (~30 seconds, no GPUs consumed during profiling) and the operator deploys the result. In a release installation, the webhook selects the matching versioned Planner image.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-small
spec:
  model: Qwen/Qwen3-0.6B
```

Apply it and watch the request progress through its phases:

```bash
kubectl apply -f qwen-small.yaml -n <namespace>
kubectl get dgdr qwen-small -n <namespace> -w
```

The DGDR moves through `Pending` → `Profiling` → `Ready` → `Deploying` → `Deployed`. Once it reaches `Deployed`, the generated DGD is running. See [Monitor profiling and deployment](#monitor-profiling-and-deployment) for the full phase list and log commands.

<Tip>
The generated DGD does **not** get an owner reference back to the DGDR. Deleting the DGDR leaves the DGD serving traffic. To tear everything down, delete both (see [Clean up](#clean-up)).
</Tip>

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
| `sla.ttft` | Target Time To First Token, ms | `2000` |
| `sla.itl` | Target Inter-Token Latency, ms | `30` |
| `sla.e2eLatency` | Target end-to-end latency, ms. **Cannot** be combined with `ttft`/`itl`. | — |

<Note>
SLA targets shape which configuration the profiler selects. They are also what the [Planner](#enable-the-planner-for-runtime-autoscaling) drives toward at runtime. Set realistic targets — unreachable values push the profiler to its most expensive layout.
</Note>

</Step>

<Step title="Choose a search strategy">

The `searchStrategy` field controls how the profiler explores configurations. The choice trades profiling time against how close to optimal you land.

**Rapid (default)** uses AIC-backed DynoSim-style performance modeling to search configurations without running real inference. Completes in ~30 seconds with no GPUs consumed during profiling.

```yaml
spec:
  searchStrategy: rapid
```

Use rapid when getting started, iterating quickly, or running in CI/CD — provided your GPU SKU is in the [Profiler support matrix](../../developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md#support-matrix). If AIC does not support your model/hardware/backend combination, the profiler falls back to a naive memory-fit config that may not be optimal.

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

For the profiling algorithms, gate checks, and how to pick a mode, see the [Profiler Guide](../../developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md).

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

GPU SKUs use **lowercase underscore format** (`h100_sxm`, not `H100-SXM5-80GB`). For the full list of accepted values and the PCIe-support caveat, see the [DGDR Reference — SKU Format](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#sku-format).

**Large and MoE models that span nodes.** When a model needs more GPUs than one node provides, the deployment is multinode and requires a gang scheduler. Install an orchestrator first — the operator returns a hard error otherwise:

- [Multinode Orchestration](../installation/multinode-orchestration.md) — install-time prerequisites (Grove + KAI, or LWS + Volcano).
- [Grove](../../developer-guide/knowledge-base/kubernetes/multinode/grove.md) (default) and [LWS](../../developer-guide/knowledge-base/kubernetes/multinode/lws.md) — the two orchestration backends.

For **Mixture-of-Experts (MoE)** models (DeepSeek-R1, Qwen3-MoE), use **SGLang** for full support — vLLM and TensorRT-LLM have partial MoE support still under development. The profiler sweeps MoE models across up to **4 nodes**; beyond that, it selects the best config within range and you may need to adjust replica counts manually. See the [Profiler support matrix](../../developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md#support-matrix).

<Note>
DGDR does not expose `multinode.nodeCount`. `hardware.totalGpus` sets the
overall profiling and deployment budget, while `hardware.numGpusPerNode`
describes per-node capacity. After selecting each worker's parallelism, the
profiler sets the generated DGD's `multinode.nodeCount` to the per-instance GPU
count divided by `numGpusPerNode`, rounded up. It omits multinode configuration
when the worker fits on one node.
</Note>

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

`pvcModelPath` must be the HuggingFace snapshot path inside the PVC: `hub/models--<org>--<model>/snapshots/<commit-hash>`. Substitute `/` with `--` in the model ID, and replace `<commit-hash>` with the actual snapshot revision. See [Model Caching — Find the Snapshot Path](../model-deployment/model-loading/model-caching.mdx#find-the-snapshot-path) for how to look it up.

**Setup:** create a `ReadWriteMany` PVC ([Installation Guide — Shared Storage](../installation/install-dynamo.md#shared-storage-for-model-caching)), run a one-time download Job to populate it, then reference it here. See [Model Caching](../model-deployment/model-loading/model-caching.mdx) for the full walkthrough.

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

The Planner's `sla` optimization target reads live TTFT/ITL from Prometheus, so install [Prometheus](../installation/install-dynamo.md#kube-prometheus-stack) before creating the DGDR if you want SLA-driven scaling. The `throughput` and `latency` modes use internal queue-depth signals and work without Prometheus. For scaling modes and the full PlannerConfig field reference, see the [Planner Guide](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md).

<Note>
`features.planner` is the PlannerConfig object itself; it does not use an
`enabled` wrapper. The DGDR API passes the object through without field-level
validation. The profiler copies it, adds GPU counts from the selected
configuration, and propagates explicit DGDR latency targets unless
`features.planner.ttft_ms` or `features.planner.itl_ms` overrides them. It then
writes the reconciled object to `planner_config.json` in a ConfigMap, mounts the
file into the generated Planner service, and starts the service with `--config`.
For non-advisory operation, it also enables scaling adapters on the worker
services selected by `mode`.
</Note>

</Step>

<Step title="Review the generated DGD">

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

While `autoApply` is disabled, you may set or change
`runtimeVersionOverride` during `Profiling` or `Ready`. To have the operator
deploy the reviewed snapshot, enable `autoApply`:

```bash
kubectl patch dgdr my-model -n <namespace> --type=merge \
  -p '{"spec":{"runtimeVersionOverride":"1.4.0"}}'
kubectl patch dgdr my-model -n <namespace> --type=merge \
  -p '{"spec":{"autoApply":true}}'
```

You may also set or change the override in the update that enables
`autoApply`. The operator applies the current value to the new DGD without
changing the stored snapshot.

</Step>

</Steps>

## Monitor profiling and deployment

A DGDR progresses through these phases. Profiling failures are terminal — they are not retried (`backoffLimit: 0`).

| Phase | What is happening |
|---|---|
| `Pending` | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job running (sub-phases: `Initializing`, `SweepingPrefill`, `SweepingDecode`, `SelectingConfig`, `BuildingCurves`, `GeneratingDGD`, `Done`) |
| `Ready` | Profiling complete; config stored in `.status.profilingResults.selectedConfig`. Waits for manual application or for `autoApply` to be enabled. |
| `Deploying` | Creating the DGD (only when `autoApply: true`) |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — check events and conditions |

Watch progress and read profiling logs:

```bash
NAMESPACE=your-namespace

# Watch phase transitions
kubectl get dgdr my-model -n "$NAMESPACE" -w

# Detailed status, conditions, and events
kubectl describe dgdr my-model -n "$NAMESPACE"

# Current profiling sub-phase
kubectl get dgdr my-model -n "$NAMESPACE" -o jsonpath='{.status.profilingPhase}'

# Profiling job logs
PROFILING_JOB=$(kubectl get dgdr my-model -n "$NAMESPACE" -o jsonpath='{.status.profilingJobName}')
kubectl logs -f "job/${PROFILING_JOB}" -c profiler -n "$NAMESPACE"
```

For the full lifecycle, conditions, and monitoring command reference, see [DGDR Reference — Lifecycle](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#lifecycle).

## Troubleshoot

| Symptom | Cause and fix |
|---|---|
| **OOM during profiling or serving** | The model doesn't fit in GPU memory at the selected TP. Raise `hardware.totalGpus`; edge cases (long context, KV overhead) need more than the minimum. |
| **Profiler ignores extra GPUs** | Auto-detection caps at 32. Set `hardware.totalGpus` explicitly. |
| **Profiling job won't schedule** | GPU nodes are tainted. Add tolerations through `overrides.profilingJob`. |
| **Spec edits rejected** | The DGDR spec is immutable once it enters `Profiling`, except that a request with `autoApply: false` may set or change `runtimeVersionOverride` during `Profiling` or `Ready`. In `Ready`, it may also enable `autoApply`. Delete and recreate the DGDR for other changes. |
| **Multinode deployment errors out** | Grove or LWS is missing. See [Multinode Orchestration](../installation/multinode-orchestration.md). |

### Profiling Job Fails to Schedule

GPU nodes commonly use taints. Add the required tolerations to the generated profiling Job through
`overrides.profilingJob`:

```yaml
spec:
  overrides:
    profilingJob:
      template:
        spec:
          containers: []
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
```

The empty `containers` list is a required placeholder. The operator merges the override into the
container configuration it generates for the profiling Job.

## Clean up

Deleting the DGDR does **not** delete the DGD it created — the DGD persists so it can keep serving. To remove both:

```bash
kubectl delete dgdr my-model -n <namespace>
kubectl delete dgd my-model-dgd -n <namespace>
```

<Note>
By default, the generated DGD is named `<dgdr-name>-dgd`. An explicit
`spec.overrides.dgd.metadata.name` replaces that default. To look up the DGD, run
`kubectl get dgd -n <namespace> -l dgdr.nvidia.com/name=my-model`. The selected
name is also recorded in `.status.dgdName`.
</Note>

## Optional: Customize the generated DGD

Use `spec.overrides.dgd` when the generated DGD needs a field that DGDR does not expose directly.
Provide a partial `nvidia.com/v1beta1` DGD. DGDR merges matching components, containers, and
environment variables by `name` after profiling selects a configuration.

For example, enable KV-aware routing on the generated `Frontend` component:

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        components:
        - name: Frontend
          podTemplate:
            spec:
              containers:
              - name: main
                env:
                - name: DYN_ROUTER_MODE
                  value: kv
```

Inspect `.status.profilingResults.selectedConfig` with `autoApply: false` to find the generated
component names. An override can modify only components already present in that generated DGD; it
cannot add a new worker, EPP, or other topology component.

<Info>
Older overrides used the `nvidia.com/v1alpha1` DGD shape. They remain supported for compatibility,
but their merge behavior differs: `spec.services` entries merge by service name, and worker
`extraPodSpec.mainContainer.args` values append to the generated arguments. In `v1beta1`, map lists
such as components, containers, and environment variables merge by `name`, while atomic lists such
as graph-level `spec.env` and container `args` replace the generated list. Use `v1beta1` for new
overrides. Include the complete desired argument list for replacement, or use the append modifier
described below.
</Info>

For example, append KV routing flags to the generated frontend arguments:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        components:
        - name: Frontend
          podTemplate:
            spec:
              containers:
              - name: main
                $patch:
                  args: append
                args:
                - --router-mode
                - kv
```

To add flags to a container with explicit generated arguments, set `$patch.args` to `append` on that
named container. The override helper combines the argument lists before returning the final DGD and
removes the `$patch` directive. The append modifier supports only `args: append`, requires a
non-empty argument list, and fails when the generated DGD does not contain the target container or
the target does not define `args`. Without `$patch`, specifying `args` retains the standard
replacement behavior and requires the complete desired argument list.

For the complete merge, metadata, and validation rules, see
[DGDR Reference — Generated DGD overrides](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#generated-dgd-overrides).

## Next steps

| Goal | Guide |
|---|---|
| Copy-ready DGDR manifests | [DGDR Templates](../../recipes/kubernetes-templates/dgdr.mdx), [Profiler Examples](../../developer-guide/knowledge-base/modular-components/profiler/profiler-examples.md) |
| Full field table and lifecycle | [DGDR Reference](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx) |
| Author the deployment by hand instead | [DGD Guide](../model-deployment/deploy-with-dgd.md) |
| Profiling algorithms and modes | [Profiler Guide](../../developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md) |
| Runtime autoscaling details | [Planner Guide](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md) |
| Full CRD reference | [API Reference](../../reference/kubernetes-api/full-api-reference.mdx) |
