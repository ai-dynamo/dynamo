---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DGDR Reference
---

A `DynamoGraphDeploymentRequest` (DGDR) is Dynamo's **deploy-by-intent** API.
You describe what you want to run and your performance targets; the profiler
determines the optimal configuration and creates the live deployment.

For a step-by-step walkthrough of deploying your model — including strategy
selection, model caching, planner setup, and common pitfalls — see the
[Model Deployment Guide](model-deployment-guide.md).

## DGDR vs DGD

Dynamo provides two Custom Resources for deploying inference graphs:

| | DGDR (recommended) | DGD (manual) |
|---|---|---|
| **You provide** | Model + optional SLA targets | Full deployment spec (parallelism, replicas, resource limits, etc.) |
| **Profiling** | Automated — sweeps configurations to find optimal setup | None — you bring your own config |
| **Hardware portability** | Adapts to whatever GPUs are in your cluster | Tied to the hardware you configured for |
| **Best for** | Most deployments, SLA-driven optimization | Known-good configs, pinned recipes |

**When to use DGD instead**: Use DGD when you have a hand-crafted configuration
for a specific model/hardware combination (e.g., from `recipes/`). These configs
may be more optimal for known setups but require understanding of what
parallelism parameters (TP, PP, EP) are appropriate and don't generalize across
different hardware.

For DGD deployment details, see [Creating Deployments](deployment/create-deployment.md).

## Spec Reference

### Minimal Example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.0.0"
```

### Field Reference

| Field | Required | Default | Purpose |
|---|---|---|---|
| `model` | Yes | — | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `image` | No | — | Container image for the profiling job. Dynamo >= 1.1.0: use `dynamo-planner`; earlier versions: use `dynamo-frontend`. |
| `backend` | No | `auto` | Inference engine: `auto`, `vllm`, `sglang`, `trtllm` |
| `searchStrategy` | No | `rapid` | Profiling depth: `rapid` (AIC simulation, ~30s) or `thorough` (real GPU, 2–4h) |
| `autoApply` | No | `true` | Automatically deploy the profiler's recommended config |
| `sla.ttft` | No | — | Target time to first token (ms) |
| `sla.itl` | No | — | Target inter-token latency (ms) |
| `sla.e2eLatency` | No | — | Target end-to-end latency (ms). Cannot be combined with explicit `ttft`/`itl`. |
| `workload.isl` | No | `4000` | Expected average input sequence length |
| `workload.osl` | No | `1000` | Expected average output sequence length |
| `workload.requestRate` | No | — | Target requests per second |
| `workload.concurrency` | No | — | Target concurrent requests |
| `hardware.gpuSku` | No | auto-detected | GPU SKU (see [SKU Format](#sku-format)) |
| `hardware.vramMb` | No | auto-detected | GPU VRAM in MB |
| `hardware.totalGpus` | No | auto-detected (capped at 32) | Total GPUs available to the deployment |
| `hardware.numGpusPerNode` | No | auto-detected | GPUs per node |
| `hardware.interconnect` | No | auto-detected | Interconnect type |
| `hardware.rdma` | No | auto-detected | Whether RDMA is available |
| `modelCache.pvcName` | No | — | Name of a `ReadWriteMany` PVC containing cached model weights |
| `modelCache.pvcModelPath` | No | — | Path to the model directory inside the PVC |
| `modelCache.pvcMountPath` | No | `/opt/model-cache` | Mount path inside containers |
| `features.planner` | No | disabled | Enable the SLA-aware Planner (raw JSON config) |
| `features.mocker` | No | disabled | Enable mocker mode for testing |
| `overrides.profilingJob` | No | — | `batchv1.JobSpec` overrides for the profiling job (e.g., tolerations) |
| `overrides.dgd` | No | — | Raw DGD override base applied to the generated deployment |

For the complete CRD spec, see the [API Reference](api-reference.md).

### SKU Format

When providing hardware configuration manually, use lowercase underscore format:

| Correct | Incorrect |
|---|---|
| `h100_sxm` | `H100-SXM5-80GB` |
| `h200_sxm` | `H200-SXM-141GB` |
| `a100_sxm` | `A100-SXM4-80GB` |
| `l40s` | `L40S` |

All supported values: `gb200_sxm`, `b200_sxm`, `h200_sxm`, `h100_sxm`,
`h100_pcie`, `a100_sxm`, `a100_pcie`, `l40s`, `l40`, `l4`, `v100_sxm`,
`v100_pcie`, `t4`, `mi200`, `mi300`.

> [!NOTE]
> Not all SKUs are supported by the AIC profiler for `rapid` mode. See
> [AIC Support Matrix](model-deployment-guide.md#aic-support-matrix) for details.

## Lifecycle

When you create a DGDR, it progresses through these phases:

| Phase | What is happening |
|---|---|
| `Pending` | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job running — sub-phases: `Initializing`, `SweepingPrefill`, `SweepingDecode`, `SelectingConfig`, `BuildingCurves`, `GeneratingDGD`, `Done` |
| `Ready` | Profiling complete; optimal config stored in `.status.profilingResults.selectedConfig`. Terminal state when `autoApply: false`. |
| `Deploying` | Creating the `DynamoGraphDeployment` (only when `autoApply: true`) |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — check events and conditions for details |

### Conditions

The operator maintains these conditions on the DGDR status:

| Condition | Meaning |
|---|---|
| `Validation` | Spec validation passed or failed |
| `Profiling` | Profiling job is running, succeeded, or failed |
| `SpecGenerated` | Generated DGD spec is available |
| `DeploymentReady` | DGD is deployed and healthy |
| `Succeeded` | Aggregate condition — true when the DGDR has reached its target state |

### Monitoring

```bash
# Watch phase transitions
kubectl get dgdr my-model -n $NAMESPACE -w

# Detailed status, conditions, and events
kubectl describe dgdr my-model -n $NAMESPACE

# Profiling sub-phase
kubectl get dgdr my-model -n $NAMESPACE -o jsonpath='{.status.profilingPhase}'

# Profiling job logs
kubectl get pods -n $NAMESPACE -l nvidia.com/dgdr-name=my-model
kubectl logs -f <profiling-pod-name> -n $NAMESPACE

# View generated DGD spec (when autoApply: false)
kubectl get dgdr my-model -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.selectedConfig}' | python3 -m json.tool

# View Pareto-optimal configs from profiling
kubectl get dgdr my-model -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.pareto}'
```

### Resource Ownership

- The DGDR does **not** set an owner reference on the DGD it creates. Deleting
  a DGDR does not delete the DGD — it persists independently so it can continue
  serving traffic.
- The relationship is tracked via labels: `dgdr.nvidia.com/name` and
  `dgdr.nvidia.com/namespace`.
- Additional resources (planner ConfigMaps) are created in the same namespace
  and labeled with `dgdr.nvidia.com/name`.

## Further Reading

- [Model Deployment Guide](model-deployment-guide.md) — How to deploy your model, strategy selection, pitfalls, examples
- [Profiler Guide](../components/profiler/profiler-guide.md) — Profiling algorithms, picking modes, gate checks
- [Profiler Examples](../components/profiler/profiler-examples.md) — Ready-to-use YAML for SLA targets, private models, MoE, overrides
- [Planner Guide](../components/planner/planner-guide.md) — Scaling modes, PlannerConfig reference
- [API Reference](api-reference.md) — Complete CRD field specifications
- [Creating Deployments](deployment/create-deployment.md) — DGD spec for full manual control
