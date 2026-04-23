---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Supported GPU SKUs
---

# Supported GPU SKUs

This page lists all GPU SKUs accepted by DGDR (`DynamoGraphDeploymentRequest`) and explains which search strategies are available for each.

## GPU SKU Format

GPU SKUs in DGDR use **lowercase underscore format**, not the node discovery labels. For example:

- ✅ **Correct**: `h100_sxm`, `a100_sxm`, `l40s`
- ❌ **Incorrect**: `H100-SXM5-80GB`, `A100_80GB`

The `H100-SXM5-80GB` style is a Kubernetes node label produced by DCGM — it is **not** a valid `gpuSku` value.

## Search Strategy Overview

DGDR supports two profiling search strategies:

- **`rapid`** (default): Uses the AI Configurator (AIC) simulator — runs in ~20–30 seconds, no real GPUs needed. Only available for AIC-supported SKUs.
- **`thorough`**: Uses AIPerf real hardware profiling — takes 2–4 hours, requires real GPUs. Available for all accepted SKUs.

## AIC-Supported SKUs (rapid + thorough)

These SKUs are supported by the AI Configurator and work with the default `rapid` search strategy:

| SKU | GPU | Form Factor |
|-----|-----|-------------|
| `gb200_sxm` | NVIDIA GB200 NVL72 | SXM |
| `b200_sxm` | NVIDIA B200 | SXM |
| `h200_sxm` | NVIDIA H200 | SXM |
| `h100_sxm` | NVIDIA H100 | SXM |
| `a100_sxm` | NVIDIA A100 | SXM |
| `l40s` | NVIDIA L40S | PCIe |

## Other Accepted SKUs (thorough only)

These SKUs are accepted by the DGDR webhook but are **not** AIC-supported. They require `searchStrategy: thorough` and real GPUs:

| SKU | GPU | Form Factor |
|-----|-----|-------------|
| `h100_pcie` | NVIDIA H100 | PCIe |
| `a100_pcie` | NVIDIA A100 | PCIe |
| `l40` | NVIDIA L40 | PCIe |
| `l4` | NVIDIA L4 | PCIe |
| `v100_sxm` | NVIDIA V100 | SXM |
| `v100_pcie` | NVIDIA V100 | PCIe |
| `t4` | NVIDIA T4 | PCIe |
| `mi200` | AMD Instinct MI200 | SXM |
| `mi300` | AMD Instinct MI300 | SXM |

## Usage in DGDR

### Auto-Detection (Recommended)

If your cluster has GPU node labels, DGDR will automatically detect available GPUs. You can omit the `gpuSku` field:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1"

  workload:
    isl: 3000
    osl: 150

  sla:
    ttft: 200.0
    itl: 20.0
```

### Manual GPU Specification

If GPU discovery is unavailable or you want to target a specific GPU:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1"

  hardware:
    gpuSku: h100_sxm        # <- Use lowercase underscore format
    numGpusPerNode: 8
    vramMb: 81920

  workload:
    isl: 3000
    osl: 150

  sla:
    ttft: 200.0
    itl: 20.0
```

### Using Thorough Search for Non-AIC SKUs

PCIe and older GPU variants require `searchStrategy: thorough`:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: cloud-deployment
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1"

  searchStrategy: thorough   # Required for non-AIC SKUs

  hardware:
    gpuSku: h100_pcie
    numGpusPerNode: 4
    vramMb: 81920

  workload:
    isl: 3000
    osl: 150

  sla:
    ttft: 200.0
    itl: 20.0
```

## Troubleshooting

### "Unknown GPU SKU" Error

If you receive a webhook validation error, verify:

1. **Format**: Use lowercase with underscores (e.g., `h100_sxm` not `H100-SXM5-80GB`)
2. **Spelling**: Check against the tables above

### GPU Not Auto-Detected

If auto-detection is not working:

1. Verify cluster nodes have GPU labels
2. Check RBAC permissions: the operator needs node read access
3. Manually specify `hardware.gpuSku` in your DGDR
