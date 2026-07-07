---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Dynamo Profiler
subtitle: How the profiler turns a DynamoGraphDeploymentRequest into a sized deployment — the pipeline, the two search modes, and what it optimizes for.
---

The profiler is the engine behind [Auto Deploy with DGDR](dgdr-guide.md). When you submit a [DynamoGraphDeploymentRequest (DGDR)](dgdr-reference.mdx), the profiler analyzes your model, hardware, and SLA targets, chooses a parallelization strategy for the prefill and decode engines, and generates the [DynamoGraphDeployment (DGD)](dgd-reference.mdx) that serves traffic.

The [DGDR walkthrough](dgdr-guide.md) is enough to deploy a model. This page is for the next level: understanding what the profiler does with your request, choosing between the two search modes deliberately, and reading what the profiler decided. It assumes you have already authored a DGDR — it does not re-cover spec fields, `autoApply`, model caching, or monitoring, which the walkthrough and the [DGDR Reference](dgdr-reference.mdx) already document.

## The profiling pipeline

The profiler runs as a Kubernetes Job while the DGDR sits in the `Profiling` phase. The stages below map to the sub-phases you can watch with `kubectl get dgdr <name> -o jsonpath='{.status.profilingPhase}'`.

<Steps toc={true} tocDepth={2}>

<Step title="Validate the request">

*Sub-phase: `Initializing`.*

The profiler checks required fields (`model`, and resolved `hardware`), verifies the SLA is internally consistent, and applies the [gate checks](#constraints-and-gate-checks). Invalid requests fail here before any GPU is touched — `backend: auto` with `searchStrategy: thorough`, or `sla.e2eLatency` combined with an explicit `sla.ttft`/`sla.itl`, are rejected at this stage.

</Step>

<Step title="Search for candidate configurations">

*Sub-phases: `SweepingPrefill`, `SweepingDecode`.*

The profiler sweeps parallelization mappings for the prefill and decode engines independently, bounded by a minimum GPU count (set by the model size and GPU VRAM) and a maximum (one node for dense models, up to four nodes for MoE). Which mappings it tries depends on the model architecture — see [Model and parallelization support](#model-and-parallelization-support).

How each candidate is scored depends on the search mode:

- **[Rapid](#rapid-mode)** estimates each candidate with AI Configurator (AIC) simulation — no engines are deployed.
- **[Thorough](#thorough-mode)** deploys each candidate as a real engine and measures it: prefill TTFT at batch size 1 (assuming the input is long enough to saturate compute), and decode ITL across a range of in-flight requests up to the KV-cache limit.

</Step>

<Step title="Pick the best configuration">

*Sub-phase: `SelectingConfig`.*

The profiler selects the prefill and decode engine configuration that satisfies your SLA. *What* it optimizes for — minimum GPUs, maximum throughput, or independent engines for the Planner to scale — is chosen automatically from your spec. See [How the profiler picks a configuration](#how-the-profiler-picks-a-configuration).

</Step>

<Step title="Build interpolation curves">

*Sub-phase: `BuildingCurves`.*

When the deployment will run the [Planner](../components/planner/planner-guide.md) or a [mocker backend](#simulate-without-gpus), the profiler builds detailed performance curves for the selected engines — TTFT versus input length for prefill, and ITL versus KV-cache utilization and context length for decode. Rapid mode derives these from AIC; thorough mode measures them on the selected engines. Deployments that need neither the Planner nor mocker skip this stage.

</Step>

<Step title="Generate the DGD">

*Sub-phases: `GeneratingDGD`, `Done`.*

The picked configuration is rendered into a complete DGD — parallelism, replica counts, image, and volume mounts. The profiler then assembles the final output in layers: a mocker template replaces the engine base if mocker is enabled, a `Planner` component and its config are injected if the Planner is enabled, and interpolation data is attached as a ConfigMap where the Planner or mocker needs it. The result lands in `status.profilingResults.selectedConfig`, and the operator deploys it when `autoApply: true`.

</Step>

</Steps>

## Rapid mode

Rapid is the default. It scores candidate configurations with AIC simulation instead of deploying engines, so it finishes in about 30 seconds and consumes no GPUs during profiling.

```yaml
spec:
  searchStrategy: rapid
```

Reach for rapid when you are getting started, iterating on SLA targets, or running in CI. It supports all three backends. The tradeoff is accuracy: results are estimated, and unusual configurations can carry error. If AIC does not support your model, GPU SKU, and backend combination, the profiler falls back to a naive memory-fit calculation that may not be optimal — confirm your SKU is in the [AIC support matrix](model-deployment-guide.md#dgdr-detail-aic-support-matrix) first.

## Thorough mode

Thorough enumerates candidate parallelization mappings, deploys each as a real Kubernetes workload, and benchmarks it with AIPerf. It produces measured rather than estimated data, at the cost of 2–4 hours and real GPU capacity during profiling.

```yaml
spec:
  searchStrategy: thorough
  backend: vllm      # a concrete backend is required
```

Use thorough when tuning for production, when your hardware is outside the AIC support matrix, or when you need measured numbers. It has three constraints:

- **Disaggregated only** — thorough does not evaluate aggregated topologies.
- **`backend: auto` is rejected** — name `vllm`, `sglang`, or `trtllm`.
- **GPUs are consumed during profiling** — the profiler deploys and benchmarks real engines for each candidate.

After picking, thorough mode also runs in-depth profiling on the selected prefill and decode engines to produce the interpolation curves described in the pipeline above — sweeping input lengths for prefill and KV-cache load and context length for decode.

## How the profiler picks a configuration

You do not choose a picking mode. The profiler derives it from two things you already set — whether the Planner is enabled, and whether you gave it a target load — and optimizes accordingly.

| The profiler optimizes for | When | Result |
|---|---|---|
| **Independent engines for autoscaling** | The Planner is enabled (`features.planner` is set) | Picks prefill and decode engines separately, each at 1 replica; the Planner scales them at runtime |
| **Fewest GPUs that meet the load** | A target load is set (`workload.requestRate` or `workload.concurrency`) and the Planner is not enabled | Finds the configuration that serves the target load under SLA with the minimum GPUs |
| **Maximum throughput** | Neither of the above | Maximizes throughput for the available GPU budget under SLA |

The Planner takes precedence: if you set both a Planner and a target load, the profiler optimizes for autoscaling. To size for a fixed load instead, leave the Planner out and set `workload.requestRate` or `workload.concurrency`.

## Model and parallelization support

The profiler supports dense and MoE models, with MoE coverage depending on the backend:

| Backend | Dense models | MoE models |
|---|---|---|
| vLLM | ✅ | 🚧 |
| SGLang | ✅ | ✅ |
| TensorRT-LLM | ✅ | 🚧 |

Which parallelization mappings it sweeps depends on the model architecture:

| Model architecture | Prefill mappings | Decode mappings |
|---|---|---|
| MLA + MoE (DeepSeek-V3, DeepSeek-V3.2) | TEP, DEP | TEP, DEP |
| GQA + MoE (Qwen3-MoE) | TP, TEP, DEP | TP, TEP, DEP |
| Dense | TP | TP |

> [!NOTE]
> Exact model × mapping support depends on the backend. The profiler does not guarantee that the recommended prefill/decode configuration is supported and bug-free on the chosen backend. For MoE, prefer SGLang for full support.

## Constraints and gate checks

The profiler enforces these rules. Most fail validation early — before any GPUs are used.

| Condition | Behavior |
|---|---|
| `searchStrategy: thorough` with `backend: auto` | Rejected — name a concrete backend |
| `searchStrategy: thorough` with an aggregated topology | Rejected — thorough is disaggregated-only |
| `sla.e2eLatency` set together with an explicit `sla.ttft` or `sla.itl` | Rejected — provide only `e2eLatency` |
| SLA is unachievable on the available hardware | Warning logged; the SLA is relaxed to the best achievable value and profiling continues |
| Target load needs more GPUs than available | Warning logged; the profiler returns its best effort |
| Model spans more GPUs than one node provides | Requires a gang scheduler (Grove or LWS); see [Multinode Orchestration](multinode-installation.md) |

## Simulate without GPUs

To validate a configuration or exercise Planner behavior at scale without real engines, enable the mocker backend. The profiler swaps the real-engine base for a mocker template that registers workers and replays modeled performance, so no GPUs are needed to run the deployment.

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  features:
    mocker:
      enabled: true
```

Mocker is a testing and experimentation path, not a serving deployment. When you enable it **alongside the Planner**, the Planner's pre-deployment sweeping cannot be `none` — the mocker needs the simulated performance data that sweeping produces. Without a Planner, mocker carries no such requirement. For how the simulated backend works and how it models performance, see [Live Simulation with Mocker](../dynosim/mocker.md).

## Accessing profiling artifacts

By default the profiler writes only its output — the generated DGD, and interpolation data when the Planner or mocker needs it — to ConfigMaps. For the full record of a run (performance plots, per-candidate configs, AIPerf results, raw `.npz` data, and logs), attach a PVC to the profiling Job through `overrides.profilingJob`:

```yaml
spec:
  overrides:
    profilingJob:
      template:
        spec:
          containers: []    # required placeholder; leave empty to inherit defaults
          volumes:
            - name: profiling-output
              persistentVolumeClaim:
                claimName: dynamo-pvc
```

Copy the results off the PVC with a temporary access pod:

```bash
kubectl apply -f deploy/utils/manifests/pvc-access-pod.yaml -n <namespace>
kubectl wait --for=condition=Ready pod/pvc-access-pod -n <namespace> --timeout=60s
kubectl cp <namespace>/pvc-access-pod:/data ./profiling-results
kubectl delete pod pvc-access-pod -n <namespace>
```

The interpolation `.npz` files are the same data the Planner consumes for autoscaling decisions. For their array schema and how the Planner uses them, see the [Planner Guide](../components/planner/planner-guide.md).

## Related pages

| Goal | Guide |
|---|---|
| Author a DGDR step by step | [Auto Deploy with DGDR](dgdr-guide.md) |
| Full field table and lifecycle | [DGDR Reference](dgdr-reference.mdx) |
| Copy-ready DGDR manifests | [DGDR Examples](dgdr-examples.md) |
| Runtime autoscaling from profiling data | [Planner Guide](../components/planner/planner-guide.md) |
| Simulate engines without GPUs | [Live Simulation with Mocker](../dynosim/mocker.md) |
| Profiling algorithm internals and interpolation schema | [Profiler Guide](../components/profiler/profiler-guide.md) |
