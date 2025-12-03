# CUDA Fault Injection - Test Library

**Purpose**: Safely simulate GPU failures (XID errors) in tests without breaking real hardware.

> **⚠️ Note**: This directory contains the **C library source code** only. The library is **compiled in-pod** during Kubernetes tests for Linux compatibility. You do **not** need to build it locally unless doing standalone local testing.

## What This Does

Makes CUDA calls return error codes to simulate various GPU failures. Uses LD_PRELOAD to intercept CUDA library calls.

```
Pod calls cudaMalloc() → LD_PRELOAD intercepts → Returns error → Pod crashes
```

**Result**: Realistic GPU failure testing without hardware damage.

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

## Files in This Directory

| File | Purpose |
|------|---------|
| `cuda_intercept.c` | C library source that intercepts CUDA calls |
| `inject_into_pods.py` | Helper functions for patching Kubernetes deployments |
| `Makefile` | Builds the `.so` library locally (optional, for standalone testing) |

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