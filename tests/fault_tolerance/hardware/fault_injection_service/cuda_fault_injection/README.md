# CUDA Fault Injection - Test Library

**Purpose**: Safely simulate GPU failures (XID errors) in tests without breaking real hardware.

> **⚠️ Note**: This directory contains the **C library source code** only. The library is **compiled in-pod** during Kubernetes tests for Linux compatibility. You do **not** need to build it locally unless doing standalone local testing.

## What This Does

Intercepts CUDA calls to simulate GPU failures using LD_PRELOAD. Faults persist across pod restarts via hostPath volumes, enabling realistic hardware failure testing.

```
Pod calls cudaMalloc() → LD_PRELOAD intercepts → Checks /host-fault/cuda_fault_enabled → Returns error → Pod crashes
```

**Key Features**:
- **Persistent faults**: hostPath volume (`/var/lib/cuda-fault-test`) survives pod restarts on same node
- **Runtime toggle**: Enable/disable faults without pod restarts via `/host-fault/cuda_fault_enabled`
- **Node-specific**: Faults only on target node, healthy nodes unaffected

## Scope

This library simulates **software/orchestration-level failures** that occur when GPU hardware becomes inaccessible or unusable:
- ✅ **In scope**: CUDA API failures due to GPU becoming unavailable (XID errors, device not found, ECC errors)
- ✅ **Use case**: Testing Kubernetes pod rescheduling, inference failover, recovery orchestration
- ❌ **Out of scope**: Bit-level Silent Data Corruption (SDC), compute errors, incorrect results
- ❌ **Not modeled**: General GPU faulting phenomena at the computation/memory level

**Note**: SDC detection mechanisms will not trigger with this approach, as we intercept at the CUDA API layer, not at the hardware/computation layer.

## Supported XID Errors

| XID | Description | CUDA Error | Use Case |
|-----|-------------|------------|----------|
| **79** | GPU fell off bus | `CUDA_ERROR_NO_DEVICE` | Most common, node-level failure |
| **48** | Double-bit ECC error | `CUDA_ERROR_ECC_UNCORRECTABLE` | Memory corruption |
| **94** | Contained ECC error | `CUDA_ERROR_ECC_UNCORRECTABLE` | Recoverable memory error |
| **95** | Uncontained error | `CUDA_ERROR_UNKNOWN` | Fatal GPU error |
| **43** | GPU stopped responding | `CUDA_ERROR_LAUNCH_TIMEOUT` | Hung kernel |
| **74** | NVLink error | `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` | Multi-GPU communication failure |

## How It Works

1. **Deployment patching**: Adds hostPath volume + init container to compile library
2. **LD_PRELOAD injection**: Environment variable loads library before CUDA
3. **Runtime control**: Toggle file (`/host-fault/cuda_fault_enabled`) controls fault state
4. **Node persistence**: hostPath ensures faults survive pod restarts on same node

## Files in This Directory

| File | Purpose |
|------|---------|
| `cuda_intercept.c` | C library that intercepts CUDA calls and checks fault markers |
| `inject_into_pods.py` | Kubernetes deployment patcher (adds hostPath volume + library) |
| `Makefile` | Local build (optional, for testing) |

## Prerequisites

- **gcc compiler** (for building the library)
- **kubectl** with cluster access
- Python packages: `kubernetes`, `requests`
- No local compilation needed (compiled in-pod)

### For Standalone Local Testing (Optional)
- **gcc** (version 7.5+ recommended, any modern gcc works)
- **CUDA development headers** (optional, uses runtime API only)

## Writing Your Own Test

### Import Helper Functions

```python
import sys
from pathlib import Path

# Add cuda_fault_injection to path
cuda_injection_dir = Path(__file__).parent.parent / "cuda_fault_injection"
sys.path.insert(0, str(cuda_injection_dir))

from inject_into_pods import (
    create_cuda_fault_configmap,      # Step 1: Create ConfigMap with library source
    patch_deployment_env,             # Step 2: Patch deployment to use it
    delete_cuda_fault_configmap       # Cleanup: Remove ConfigMap
)
```