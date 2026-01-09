# CUDA Fault Injection Library

LD_PRELOAD library that intercepts CUDA API calls to simulate GPU failures.

## Why This Component

The CUDA shim library simulates GPU unavailability at the application level:
- Intercepts CUDA calls (`cudaMalloc`, `cudaLaunchKernel`, etc.)
- Returns error codes matching real XID failures
- Causes pods to crash and trigger Kubernetes reschedule

**Recovery method:** Pod crashes → Kubernetes restarts → Scheduler places on healthy node (if original node is cordoned by NVSentinel)

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Worker Pod                                │
│  ┌───────────────────┐    ┌───────────────────────────────────┐ │
│  │  LD_PRELOAD       │───▶│  cuda_intercept.so                │ │
│  │  env var          │    │  - Intercepts CUDA calls          │ │
│  └───────────────────┘    │  - Reads toggle file              │ │
│                           │  - Returns error codes            │ │
│  ┌───────────────────┐    └───────────────────────────────────┘ │
│  │  /host-fault/     │◀───── hostPath volume (persists across   │
│  │  cuda_fault_      │       pod restarts on same node)         │
│  │  enabled          │                                          │
│  └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key feature:** Fault marker persists via hostPath (`/var/lib/cuda-fault-test`), so pods keep crashing on the "faulty" node until cleanup.

## Supported XID Types

| XID | CUDA Error | Description |
|-----|------------|-------------|
| 79 | `cudaErrorNoDevice` | GPU fell off bus (default) |
| 48 | `cudaErrorEccUncorrectable` | Double-bit ECC error |
| 94 | `cudaErrorEccUncorrectable` | Contained ECC error |
| 95 | `cudaErrorUnknown` | Uncontained error |
| 43 | `cudaErrorLaunchTimeout` | GPU stopped responding |
| 74 | `cudaErrorPeerAccessUnsupported` | NVLink error |

## Build

```bash
make
# Creates: cuda_intercept.so
```

Or manually:
```bash
gcc -shared -fPIC -ldl cuda_intercept.c -o cuda_intercept.so
```

## Usage

### Method 1: Deployment Patch (Recommended)

Patches the DynamoGraphDeployment to load the library in all worker pods:

```bash
# Setup with passthrough mode (library loaded, faults disabled)
python inject_into_pods.py \
  --deployment vllm-v1-disagg-router \
  --namespace dynamo \
  --patch-deployment \
  --xid-type 79 \
  --passthrough \
  --node gpu-node-001  # Optional: pin pods to specific node

# Verify
python inject_into_pods.py \
  --deployment vllm-v1-disagg-router \
  --namespace dynamo \
  --verify

# Remove injection
python inject_into_pods.py \
  --deployment vllm-v1-disagg-router \
  --namespace dynamo \
  --remove
```

### Method 2: Direct Pod Injection (Temporary)

Copies library to running pods (lost on restart):

```bash
python inject_into_pods.py \
  --deployment vllm-v1-disagg-router \
  --namespace dynamo
```

### Method 3: Python API

```python
from cuda_fault_injection import CUDAFaultInjector

injector = CUDAFaultInjector()

# Build library
injector.build_library()

# Create ConfigMap with source
injector.create_configmap_with_library("dynamo")

# Patch deployment (passthrough mode)
injector.patch_deployment_for_cuda_fault(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo",
    target_node="gpu-node-001",
    xid_type=79,
    passthrough_mode=True  # Faults disabled until toggled
)

# Toggle faults ON (no pod restart)
pods = k8s_core.list_namespaced_pod(namespace, label_selector="...")
injector.enable_cuda_faults_via_toggle(pods.items, namespace, enable=True, target_node="gpu-node-001")

# Toggle faults OFF
injector.enable_cuda_faults_via_toggle(pods.items, namespace, enable=False)

# Cleanup
injector.cleanup_cuda_fault_injection(deployment_name, namespace)
```

## Toggle Mechanism

The library checks a file at runtime to enable/disable faults:

```c
// In cuda_intercept.c
FILE* toggle_file = fopen("/host-fault/cuda_fault_enabled", "r");
if (toggle_file) {
    char toggle_value[4] = {0};
    if (fgets(toggle_value, sizeof(toggle_value), toggle_file)) {
        runtime_inject = (toggle_value[0] == '1');
    }
    fclose(toggle_file);
}
```

**Toggle via kubectl:**
```bash
# Enable faults
kubectl exec -n dynamo $POD_NAME -- sh -c 'echo 1 > /host-fault/cuda_fault_enabled'

# Disable faults
kubectl exec -n dynamo $POD_NAME -- sh -c 'echo 0 > /host-fault/cuda_fault_enabled'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_FAULT_INJECTION_ENABLED` | Initial state (`0` or `1`) | `0` in passthrough |
| `CUDA_XID_TYPE` | XID type to simulate | `79` |
| `LD_PRELOAD` | Path to library | `/cuda-fault/cuda_intercept.so` |

## What Gets Patched

When using `--patch-deployment`, the following is added to worker services:

1. **ConfigMap volume** - Library source code
2. **Init container** - Compiles library (Linux-compatible)
3. **Volume mounts** - Library + hostPath fault marker
4. **Environment variables** - `LD_PRELOAD`, `CUDA_XID_TYPE`, `CUDA_FAULT_INJECTION_ENABLED`
5. **Node affinity** (optional) - Pins pods to target node

## Cleanup

Full cleanup removes:
- LD_PRELOAD environment variable
- ConfigMap volume and mounts
- Init container
- Node affinity
- ConfigMap resource
- hostPath fault marker files

```python
injector.cleanup_cuda_fault_injection(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo",
    force_delete_pods=True  # Restart pods with clean spec
)
```

## Troubleshooting

### Library not loading
```bash
# Check LD_PRELOAD is set
kubectl exec -n dynamo $POD -- env | grep LD_PRELOAD

# Check library exists
kubectl exec -n dynamo $POD -- ls -la /cuda-fault/
```

### Faults not triggering
```bash
# Check toggle file
kubectl exec -n dynamo $POD -- cat /host-fault/cuda_fault_enabled
# Should show "1" if enabled

# Check init container logs
kubectl logs -n dynamo $POD -c compile-cuda-fault-lib
```

### Pods not crashing
- Verify `CUDA_FAULT_INJECTION_ENABLED=1` in pod env
- Check pod logs for `[XID 79 SIM]` messages
- Ensure application actually calls CUDA APIs
