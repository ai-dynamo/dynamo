---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DGDR Deployment Guide
---

# DGDR Deployment Guide

A `DynamoGraphDeploymentRequest` (DGDR) is Dynamo's **deploy-by-intent** API. You describe
what you want to run and your performance targets; Dynamo's profiler determines the optimal
configuration automatically, then creates the live deployment for you.

If you haven't deployed a model yet, start with the [Quickstart](README.md).

## DGDR vs DGD

Dynamo provides two Custom Resources for deploying inference graphs:

| | DGDR (recommended) | DGD (manual) |
|---|---|---|
| **You provide** | Model + optional SLA targets | Full deployment spec (parallelism, replicas, resource limits, etc.) |
| **Profiling** | Automated — sweeps configurations to find optimal setup | None — you bring your own config |
| **Hardware portability** | Adapts to whatever GPUs are in your cluster | Tied to the hardware you configured for |
| **Best for** | Most deployments, SLA-driven optimization | Known-good configs, pinned recipes |

**When to use DGD instead**: Use DGD when you have a hand-crafted configuration for a specific
model/hardware combination (e.g., from `recipes/`). These configs may be more optimal for known
setups but require understanding of what parallelism parameters (TP, PP, EP) are appropriate
and don't generalize across different hardware.

For DGD deployment details, see [Creating Deployments](deployment/create-deployment.md).

## DGDR Spec Reference

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

### Production Example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model-prod
spec:
  model: meta-llama/Llama-3.1-70B-Instruct
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.0.0"
  backend: vllm
  searchStrategy: thorough
  autoApply: false
  sla:
    ttft: 500
    itl: 50
  workload:
    isl: 4000
    osl: 500
    requestRate: 10
  features:
    planner:
      enabled: true
```

### Field Reference

| Field | Required | Default | Purpose |
|---|---|---|---|
| `model` | Yes | — | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `image` | No | — | Container image for the profiling job. Dynamo >= 1.1.0: use `dynamo-planner`; earlier versions: use `dynamo-frontend`. |
| `backend` | No | `auto` | Inference engine: `auto`, `vllm`, `sglang`, `trtllm` |
| `searchStrategy` | No | `rapid` | Profiling depth — see [Search Strategy](#search-strategy) |
| `autoApply` | No | `true` | Automatically deploy after profiling — see [autoApply](#autoapply) |
| `sla` | No | — | Target latency (TTFT, ITL in ms) |
| `workload` | No | — | Expected traffic shape (ISL, OSL, request rate) |
| `hardware` | No | auto-detected | GPU SKU and count override. Auto-detected count is capped at 32 — set `hardware.totalGpus` explicitly to use more. |
| `features.planner` | No | disabled | Enable the SLA-aware Planner — see [Planner](#planner) |

For the complete CRD spec, see the [API Reference](api-reference.md).

## Search Strategy

The `searchStrategy` field controls how the profiler explores configuration options:

| Strategy | Time | Method | Best for |
|---|---|---|---|
| `rapid` (default) | ~30 seconds | AIC simulation — models GPU behavior without running real inference | Getting started, fast iteration, CI/CD |
| `thorough` | 2–4 hours | Real GPU profiling — runs actual inference across candidate configs | Production tuning, maximizing performance |

**`rapid`** uses the AI Configurator (AIC) to simulate performance across backend ×
parallelism combinations. It's fast enough for development and produces good configs for most
models.

**`thorough`** runs real inference workloads on your cluster's GPUs, profiling each candidate
configuration with actual model execution. Use this when you need the most optimal configuration
and can afford the profiling time.

## autoApply

Controls what happens after profiling completes:

- **`autoApply: true`** (default) — The profiler's recommended configuration is automatically
  deployed as a `DynamoGraphDeployment` (DGD). The DGDR transitions through
  `Profiling` → `Deploying` → `Deployed`.

- **`autoApply: false`** — Profiling completes and the DGDR reaches `Ready`. The generated
  config is stored in `.status.profilingResults.selectedConfig` but nothing is deployed. This
  lets you inspect and optionally modify the config before deploying:

```bash
# View the generated DGD spec
kubectl get dgdr my-model -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.selectedConfig}' | python3 -m json.tool

# Save it, review/edit, then apply
kubectl get dgdr my-model -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.selectedConfig}' > generated-dgd.yaml
kubectl apply -f generated-dgd.yaml -n $NAMESPACE
```

Use `autoApply: false` when you want to review the profiler's choices, tweak parallelism
settings, or save the generated config as a DGD recipe for future use.

## Planner

The Planner provides **SLA-aware autoscaling** for disaggregated deployments. When enabled,
it monitors live traffic and adjusts the prefill/decode split, replica counts, and resource
allocation to meet your latency targets.

```yaml
spec:
  features:
    planner:
      enabled: true
  sla:
    ttft: 500    # Target TTFT in ms
    itl: 50      # Target ITL in ms
```

The Planner requires:
- Prometheus installed and configured (see [Installation Guide — Prometheus](installation-guide.md#prometheus--observability))
- SLA targets in the DGDR spec (otherwise there's nothing to optimize for)

See the [Planner Guide](../components/planner/planner-guide.md) for details on optimization
behavior and advanced configuration.

## DGDR Lifecycle

When you create a DGDR, it progresses through these phases:

| Phase | What is happening |
|---|---|
| `Pending` (condition: `DiscoveringHardware`) | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job running (AIC simulation or real-GPU sweep) |
| `Ready` | Profiling complete; optimal config stored in `.status`. Terminal state when `autoApply: false` |
| `Deploying` | Creating the `DynamoGraphDeployment` (only when `autoApply: true`) |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — check events for details |

### Monitoring Progress

```bash
# Watch phase transitions
kubectl get dgdr my-model -n $NAMESPACE -w

# Full status and events
kubectl describe dgdr my-model -n $NAMESPACE

# Profiling job logs
kubectl get pods -n $NAMESPACE -l nvidia.com/dgdr-name=my-model
kubectl logs -f <profiling-pod-name> -n $NAMESPACE
```

## Model Caching

If you are deploying a **large model (>70B parameters)** or scaling to **many replicas**, each
pod downloads the full model independently from HuggingFace. This can take hours and trigger
rate limits.

**Set up shared storage before deploying large models.** Create a `ReadWriteMany` PVC, run a
one-time download Job, then mount the PVC in your deployment. See
[Model Caching](model-caching.md) for the full walkthrough.

For cloud-provider storage options (EFS, Azure Files/Lustre, GKE Filestore), see the
[Installation Guide — Model Caching](installation-guide.md#model-caching--shared-storage).

## GPU SKUs and Hardware

### Supported SKUs

The AIC profiler (`rapid` mode) currently supports a limited set of GPU SKUs — primarily
SXM variants (H100, H200, A100, B100, B200) and L40S. PCIe variants are not yet supported.

If your GPU is not recognized, the DGDR will report a validation error. Provide the exact SKU
in `hardware.gpuSku` if auto-detection doesn't match a supported value.

### SKU Format

When providing hardware configuration manually, use lowercase underscore format:

| Correct | Incorrect |
|---|---|
| `h100_sxm` | `H100-SXM5-80GB` |
| `h200_sxm` | `H200-SXM-141GB` |
| `l40s` | `L40S` |

### GPU Count and Parallelism

- `hardware.totalGpus` determines how many GPUs the profiler can allocate across the deployment.
  Auto-detection caps at 32 — set this explicitly if you need more.
- Within a single worker pod, `resources.limits.gpu` must equal the tensor parallel size —
  all TP GPUs must be visible to the same process.
- For multinode deployments, ensure you have [Grove](grove.md) and
  [KAI Scheduler](installation-guide.md#grove-and-kai-scheduler) installed.

## Overrides

Use `overrides` to customize the profiling job or generated deployment without modifying the
core spec. Common use case — adding GPU node tolerations:

```yaml
spec:
  overrides:
    profilingJob:
      template:
        spec:
          containers: []    # required placeholder; leave empty to inherit defaults
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
```

The operator forwards profiling job tolerations to every candidate and deployed pod.

## Troubleshooting

**DGDR stuck in `Pending`**

```bash
kubectl describe dgdr my-model -n $NAMESPACE
# Check the Events section
```

Common causes:
- No available GPU nodes — check `kubectl get nodes -l nvidia.com/gpu.present=true`
- Image pull failure — verify the image tag matches your platform version
- GPU node taints — add tolerations via `overrides.profilingJob` (see above)

**Profiling job fails**

```bash
kubectl get pods -n $NAMESPACE -l nvidia.com/dgdr-name=my-model
kubectl logs <profiling-pod-name> -n $NAMESPACE
# If the pod already exited:
kubectl logs <profiling-pod-name> -n $NAMESPACE --previous
```

**Pods not starting after profiling**

```bash
kubectl describe pod <pod-name> -n $NAMESPACE
# Look for ImagePullBackOff, OOMKilled, or Insufficient resources
```

**Version mismatch errors**

Ensure container image versions match your installed Helm chart:

```bash
helm list -n $NAMESPACE
kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u
```

**Deleting a DGDR**

```bash
kubectl delete dgdr my-model -n $NAMESPACE
```

Deleting a DGDR does **not** delete the `DynamoGraphDeployment` it created. The DGD persists
independently so it can continue serving traffic.

## Further Reading

- [Profiler Guide](../components/profiler/profiler-guide.md) — Profiling algorithms and configuration
- [DGDR Examples](../components/profiler/profiler-examples.md) — Ready-to-use YAML for SLA targets, private models, MoE, overrides
- [Planner Guide](../components/planner/planner-guide.md) — SLA optimization details
- [API Reference](api-reference.md) — Complete CRD field specifications
- [Creating Deployments](deployment/create-deployment.md) — DGD spec for full manual control
- [Autoscaling](autoscaling.md) — HPA and KEDA integration
