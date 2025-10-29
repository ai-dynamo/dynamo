# CUDA Fault Injection - Test Library

**Purpose**: Safely simulate GPU failures (XID errors) in tests without breaking real hardware.

## What This Does

Makes CUDA calls return error codes to simulate various GPU failures. Uses LD_PRELOAD to intercept CUDA library calls.

```
Pod calls cudaMalloc() → LD_PRELOAD intercepts → Returns error → Pod crashes
```

**Result**: Realistic GPU failure testing without hardware damage.

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
| `fake_cuda_xid79.c` | C library that intercepts CUDA calls |
| `inject_into_pods.py` | Helper functions for patching Kubernetes deployments |
| `Makefile` | Builds the `.so` library (runs `gcc`) |
| `test_xid_template.py` | Example test templates for all supported XID types |

## Prerequisites

- **gcc compiler** (for building the library)
- **kubectl** with cluster access
- Python packages: `kubernetes`, `requests`

## Writing Your Own Test

### 1. Build the Library (First Time Only)

```bash
cd cuda-fault-injection
make  # Creates fake_cuda_xid79.so
```

### 2. Import Helper Functions

```python
import sys
import os
from pathlib import Path

# Add cuda-fault-injection to path
cuda_injection_dir = Path(__file__).parent.parent / "cuda-fault-injection"
sys.path.insert(0, str(cuda_injection_dir))

from inject_into_pods import (
    create_cuda_fault_configmap,      # Step 1: Create ConfigMap with library
    patch_deployment_env,             # Step 2: Patch deployment to use it
    delete_cuda_fault_configmap       # Cleanup: Remove ConfigMap
)
```

### 3. Use in Your Test

```python
def test_gpu_failure_xid79():
    """Test XID 79 - GPU fell off bus"""
    deployment_name = "vllm-worker"
    namespace = "default"
    target_node = "node-with-gpu"
    
    # Enable CUDA fault injection for XID 79 (default)
    create_cuda_fault_configmap(namespace)
    patch_deployment_env(deployment_name, namespace, enable=True, 
                        use_configmap=True, target_node=target_node,
                        xid_type=79)
    
    # Pods will crash with CUDA_ERROR_NO_DEVICE
    # ... your test logic here ...
    
    # Cleanup
    patch_deployment_env(deployment_name, namespace, enable=False, use_configmap=True)
    delete_cuda_fault_configmap(namespace)

def test_ecc_error_xid48():
    """Test XID 48 - Double-bit ECC error"""
    create_cuda_fault_configmap(namespace)
    patch_deployment_env(deployment_name, namespace, enable=True,
                        use_configmap=True, xid_type=48)
    
    # Pods will crash with CUDA_ERROR_ECC_UNCORRECTABLE
    # ... your test logic here ...
    
    # Cleanup
    patch_deployment_env(deployment_name, namespace, enable=False, use_configmap=True)
    delete_cuda_fault_configmap(namespace)
```

## Key Functions

### `create_cuda_fault_configmap(namespace)`
Creates a ConfigMap containing the C source code. An init container compiles it in-pod (Linux-compatible).

### `patch_deployment_env(deployment_name, namespace, enable, use_configmap, target_node, xid_type)`
- **enable=True**: Adds LD_PRELOAD env var, mounts ConfigMap, adds init container
- **enable=False**: Removes all CUDA fault artifacts
- **target_node**: Pins pods to specific node (simulates real XID where failure is localized)
- **xid_type**: XID error to simulate (79, 48, 94, 95, 43, 74). Default: 79

Works with both standard Deployments and DynamoGraphDeployments.

### `delete_cuda_fault_configmap(namespace)`
Removes the ConfigMap after test.

## How It Works

1. **ConfigMap** stores `fake_cuda_xid79.c` source code (not compiled `.so` - ensures Linux compatibility since local build is macOS)
2. **Init container** compiles source in-pod to `/cuda-fault/fake_cuda_xid79.so`
3. **LD_PRELOAD** env var tells the OS to load our library before CUDA
4. **CUDA calls** get intercepted and return error code based on `CUDA_XID_TYPE` env var
5. **App crashes** naturally just like with real GPU failure

**DynamoGraphDeployment**: For vLLM disaggregated serving (custom resource), patches only GPU workers (PrefillWorker, DecodeWorker), not Frontend (CPU-only). Falls back to standard Deployment patching if not DGD.

## Environment Variables (Automatic)

Set by `patch_deployment_env()`:
- `LD_PRELOAD=/cuda-fault/fake_cuda_xid79.so` - Activates interception
- `CUDA_FAULT_INJECTION_ENABLED=1` - Enables fault injection
- `CUDA_XID_TYPE=<number>` - Which XID error to simulate (default: 79)

## Examples

**Full E2E Test**: `examples/test_xid79_cuda_fault.py`
- Complete workflow for XID 79
- Building library, injection, crash detection
- Simulating NVSentinel recovery
- Cleanup and verification

**Test Templates**: `test_xid_template.py`
- Ready-to-copy functions for all XID types
- Shows expected behavior for each XID
- Includes pytest fixture example

## Why This Approach?

Real life:
- GPU Falls off bus
- All requests immediately begin failing as the CUDA calls to the pods return an error since the device is not found, causing pods to crash
- XID Error detected and logged
- NVSentinel kicks in (drains & cordons node, restarts GPU driver)
- Pods reschedule onto healthy node with accessible GPU
- Inference requests begin working again

Simulation
- XID Error inserted (inference requests continue to succeed since processes are still running)
- NVSentinel kicks in (cordons & drains node, restarts GPU driver). Inference requests still succeed until drainage.
- Pods reschedule onto healthy node with accessible GPU.
- Inference requests begin working again

This fake CUDA library simulates the CUDA call failures so that we induce pod crashing earlier and inference fails where it is expected to.


**LD_PRELOAD makes pods crash naturally from CUDA errors, not from SIGTERM.**

Tests pod crash behavior, Kubernetes rescheduling, inference failover, and NVSentinel remediation with realistic failure simulation.

## References

- [CUDA Error Codes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)
- [LD_PRELOAD man page](https://man7.org/linux/man-pages/man8/ld.so.8.html)
- [Function Interposition](https://en.wikipedia.org/wiki/Interposition)

