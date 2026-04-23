---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Supported GPU SKUs
---

# Supported GPU SKUs

This page lists all GPU SKUs supported by DGDR (DynamoGraphDeploymentRequest) for profiling and deployment.

## GPU SKU Format

GPU SKUs in DGDR use **lowercase hyphenated format**, not the node discovery labels. For example:

- ✅ **Correct**: `h100_sxm`, `a100_pcie`, `l40`
- ❌ **Incorrect**: `H100-SXM5-80GB`, `A100_80GB`

## Supported GPU SKUs

### NVIDIA Blackwell
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `gb200_sxm` | SXM | Data center |
| `b200_sxm` | SXM | Data center |

### NVIDIA Hopper
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `h200_sxm` | SXM | Data center |
| `h100_sxm` | SXM | Data center |
| `h100_pcie` | PCIe | Cloud / On-prem |

### NVIDIA Ampere
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `a100_sxm` | SXM | Data center |
| `a100_pcie` | PCIe | Cloud / On-prem |

### NVIDIA Ada
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `l40s` | PCIe | Cloud / On-prem |
| `l40` | PCIe | Cloud / On-prem |
| `l4` | PCIe | Cloud / On-prem |

### NVIDIA Older Generations
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `v100_sxm` | SXM | Legacy data center |
| `v100_pcie` | PCIe | Legacy on-prem |
| `t4` | PCIe | Cloud / On-prem |

### AMD EPYC Accelerators
| SKU | Form Factor | Availability |
|-----|-------------|--------------|
| `mi200` | SXM | Data center |
| `mi300` | SXM | Data center |

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

If GPU discovery is unavailable or you want to specify a particular GPU:

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
    gpuSku: h100_sxm        # ← Use lowercase, underscore format
    numGpusPerNode: 8
    vramMb: 81920
  
  workload:
    isl: 3000
    osl: 150
  
  sla:
    ttft: 200.0
    itl: 20.0
```

## Cloud Deployment with PCIe GPUs

PCIe GPU variants (h100_pcie, a100_pcie, l40, l4, t4) are fully supported and recommended for cloud and colocation deployments:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: cloud-deployment
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1"
  
  hardware:
    gpuSku: h100_pcie       # PCIe variant for cloud
    numGpusPerNode: 4
    vramMb: 81920
  
  # ... rest of spec
```

## Troubleshooting

### "Unknown GPU SKU" Error

If you receive a validation error with an unknown GPU SKU, verify:

1. **Format**: Use lowercase with underscores (e.g., `h100_sxm` not `H100-SXM5-80GB`)
2. **Spelling**: Check against the table above
3. **Availability**: Ensure the GPU is available in your cluster

### GPU Not Auto-Detected

If auto-detection isn't working:

1. Verify cluster nodes have GPU labels
2. Check RBAC permissions: the operator needs node read access
3. Manually specify `hardware.gpuSku` in your DGDR
