# GPU Fault Injection - User Guide

Test GPU hardware failure scenarios without damaging real hardware. Inject XID errors, simulate CUDA failures, and validate NVSentinel automated recovery workflows.

## What You Can Do

### XID Error Injection
- **Inject GPU errors into kernel logs** - Simulate real hardware failures
- **Multiple XID types supported** - XID 79 (GPU off bus), 48/94 (ECC), 43 (hang), 74 (NVLink)
- **NVSentinel integration** - Validate automated detection and remediation
- **No GPU hardware required** - Safe testing in any environment

### CUDA Fault Injection (Advanced)
- **Realistic pod crashes** - Pods experience actual CUDA errors
- **LD_PRELOAD library** - Intercepts CUDA API calls
- **Multiple error types** - Simulate XID 79, 48, 94, 95, 43, 74 behaviors
- **Automatic cleanup** - Removes all artifacts after testing

### Combined Testing (Recommended)
- **XID + CUDA together** - Most realistic GPU failure simulation
- **Validates full pipeline** - Detection → Quarantine → Drain → Remediation → Recovery
- **NVSentinel E2E** - Tests complete automated workflow

## Prerequisites

### Required
- Fault Injection API deployed (see [README.md](README.md))
- GPU fault injector agent running as DaemonSet on GPU nodes
- Target workload using CUDA (e.g., vLLM workers)

### Optional
- **NVSentinel deployed** - For automated detection and remediation testing
- **kubectl access** - For checking kernel logs and pod status

### Installation Check
```bash
# Verify fault injection API
kubectl get pods -n fault-injection-system -l app=fault-injection-api

# Verify GPU fault injector agents
kubectl get pods -n fault-injection-system -l app=gpu-fault-injector

# Check NVSentinel (optional)
kubectl get pods -n nvsentinel
```

## Quick Start

### Run Pre-Built Tests

```bash
# XID 79 with NVSentinel automation (requires NVSentinel deployed)
python scripts/run_test_incluster.py examples/test_xid79_nvsentinel_automated.py

# XID 79 with CUDA fault injection (realistic pod crashes)
python scripts/run_test_incluster.py examples/test_xid79_cuda_fault_refactored.py

# XID 94 ECC error test
python scripts/run_test_incluster.py examples/test_xid_94_ecc_error.py

# All XID types test
python scripts/run_test_incluster.py examples/test_all_xid_errors.py
```

### Test Helpers

```python
from test_helpers import Colors, get_config_from_env
from helpers.cuda_fault_injection import CUDAFaultInjector
from fault_injection_client import FaultInjectionClient

# Load configuration
config = get_config_from_env()

# Initialize clients
api_client = FaultInjectionClient(api_url=config['api_url'])
cuda_injector = CUDAFaultInjector()
```

## Supported XID Types

| XID | Description | CUDA Error | When to Use |
|-----|-------------|------------|-------------|
| **79** | GPU fell off bus | `CUDA_ERROR_NO_DEVICE` | Most common failure, node-level, NVSentinel testing |
| **48** | Double-bit ECC error | `CUDA_ERROR_ECC_UNCORRECTABLE` | Memory corruption scenarios |
| **94** | Contained ECC error | `CUDA_ERROR_ECC_UNCORRECTABLE` | Recoverable memory errors |
| **95** | Uncontained error | `CUDA_ERROR_UNKNOWN` | Fatal GPU errors |
| **43** | GPU stopped responding | `CUDA_ERROR_LAUNCH_TIMEOUT` | Hung kernel detection |
| **74** | NVLink error | `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` | Multi-GPU communication failures |

## Usage Approaches

### Approach 1: XID Injection Only (Kernel Logs)

Best for: NVSentinel detection testing

**What it does:**
- Writes XID error to kernel logs via `nsenter`
- NVSentinel's syslog-health-monitor detects it
- Triggers automated workflow (cordon → drain → remediate)
- Pods keep running normally (no crashes)

**Example:**
```python
from fault_injection_client import FaultInjectionClient
import time

client = FaultInjectionClient(api_url="http://localhost:8080")

# Inject XID 79 on specific node
fault = client.inject_xid_error(
    node_name="worker-node-1",
    xid_type=79,
    gpu_id=0
)

print(f"XID injected: {fault.fault_id}")

# Wait for NVSentinel to detect
time.sleep(10)

# Check kernel logs
# kubectl exec <gpu-operator-pod> -- journalctl -k | grep "Xid.*79"

# Cleanup
client.delete_fault(fault.fault_id)
```

**Verification:**
```bash
# Check kernel logs for XID
kubectl exec -n gpu-operator <gpu-operator-pod> -- journalctl -k | grep "Xid.*79"

# Check NVSentinel detection
kubectl logs -n nvsentinel -l app=syslog-health-monitor | grep "XID"
```

### Approach 2: CUDA Fault Injection Only (Realistic Crashes)

Best for: Pod crash behavior testing

**What it does:**
- Injects CUDA fault library into pods via LD_PRELOAD
- CUDA API calls return error codes
- Pods crash naturally with CUDA errors
- No kernel log XID (unless combined with Approach 1)

**Example:**
```python
from helpers.cuda_fault_injection import CUDAFaultInjector

cuda_injector = CUDAFaultInjector()

# Step 1: Build library (first time only)
cuda_injector.build_library()

# Step 2: Create ConfigMap with library source
cuda_injector.create_configmap_with_library("dynamo-oviya")

# Step 3: Patch deployment to inject CUDA faults
cuda_injector.patch_deployment_for_cuda_fault(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo-oviya",
    target_node="worker-node-1",  # Pin pods to specific node
    xid_type=79                    # Simulate XID 79 behavior
)

# Pods will restart with CUDA faults injected
# They will crash with CUDA_ERROR_NO_DEVICE when they try to use GPU

# Wait for pods to crash
time.sleep(60)

# Step 4: Cleanup (removes ConfigMap, env vars, volumes, restarts pods)
cuda_injector.cleanup_cuda_fault_injection(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo-oviya",
    force_delete_pods=True  # Force-delete stuck pods
)
```

**What gets modified:**
- ConfigMap with `.c` source code created
- Init container added to compile library
- Volume mounts added for library
- Environment variables: `LD_PRELOAD`, `CUDA_FAULT_INJECTION_ENABLED`, `CUDA_XID_TYPE`
- Node affinity (pins pods to target node)

**Cleanup removes:**
- ConfigMap
- Init container
- Volume mounts
- Environment variables
- Node affinity

### Approach 3: Combined XID + CUDA (Most Realistic)

Best for: Full E2E testing with NVSentinel

**What it does:**
- XID in kernel logs (for NVSentinel detection)
- CUDA faults in pods (for realistic crashes)
- Complete simulation of real GPU hardware failure

**Example:**
```python
from fault_injection_client import FaultInjectionClient
from helpers.cuda_fault_injection import CUDAFaultInjector
import time

api_client = FaultInjectionClient(api_url="http://localhost:8080")
cuda_injector = CUDAFaultInjector()

# Step 1: Inject XID 79 (kernel logs for NVSentinel)
fault = api_client.inject_xid_error(
    node_name="worker-node-1",
    xid_type=79,
    gpu_id=0
)
print(f"[1/3] XID 79 injected in kernel logs")

# Step 2: Inject CUDA faults (pods will crash)
cuda_injector.build_library()
cuda_injector.create_configmap_with_library("dynamo-oviya")
cuda_injector.patch_deployment_for_cuda_fault(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo-oviya",
    target_node="worker-node-1",
    xid_type=79
)
print(f"[2/3] CUDA faults injected - pods will crash naturally")

# Pods crash, NVSentinel detects, automated workflow begins
# - fault-quarantine-module: Cordons node
# - node-drainer-module: Drains pods
# - fault-remediation-module: Restarts GPU driver (optional)
# - Automatic uncordon: Node returns to service

# Wait for workflow to complete
time.sleep(300)  # 5 minutes

# Step 3: Cleanup
cuda_injector.cleanup_cuda_fault_injection(
    "vllm-v1-disagg-router", "dynamo-oviya", force_delete_pods=True
)
api_client.delete_fault(fault.fault_id)
print(f"[3/3] Cleanup complete")
```

## Integration with NVSentinel

When NVSentinel is deployed, the automated workflow triggers:

**Complete Workflow:**
1. **XID Detection** - syslog-health-monitor detects XID from kernel logs
2. **Pod Crashes** - CUDA faults cause pods to crash (realistic behavior)
3. **Quarantine** - fault-quarantine-module cordons the node
4. **Drain** - node-drainer-module drains crashing pods
5. **Remediation** - fault-remediation-module restarts GPU driver (optional)
6. **Uncordon** - Node automatically uncordoned after remediation
7. **Recovery** - Pods reschedule, inference resumes

**Example Test:**
```bash
# Runs complete automated workflow validation
python scripts/run_test_incluster.py examples/test_xid79_nvsentinel_automated.py
```

See `examples/test_xid79_nvsentinel_automated.py` for full E2E test implementation.

## CUDA Fault Injection Deep Dive

### How It Works

1. **ConfigMap Creation**
   - Stores `.c` source code (not compiled `.so`)
   - Ensures Linux compatibility (macOS builds won't work)

2. **Init Container Compilation**
   - Uses `gcc:latest` image
   - Compiles source in-pod: `/source/fake_cuda_xid79.c` → `/dest/fake_cuda_xid79.so`
   - Output stored in emptyDir volume

3. **Main Container Injection**
   - Volume mounted at `/cuda-fault/fake_cuda_xid79.so`
   - `LD_PRELOAD=/cuda-fault/fake_cuda_xid79.so` set
   - `CUDA_FAULT_INJECTION_ENABLED=1` set
   - `CUDA_XID_TYPE=79` set (or other XID type)

4. **CUDA Interception**
   - Library intercepts: `cuInit()`, `cuMemAlloc()`, `cudaMalloc()`, etc.
   - Returns appropriate error codes based on XID type
   - Pod crashes just like real GPU failure

### Configuration Options

```python
cuda_injector.patch_deployment_for_cuda_fault(
    deployment_name="vllm-v1-disagg-router",  # DynamoGraphDeployment name
    namespace="dynamo-oviya",                  # Kubernetes namespace
    target_node="worker-node-1",               # Pin pods to this node (optional)
    xid_type=79,                               # XID type to simulate (79, 48, 94, 95, 43, 74)
)
```

### Cleanup

**Automatic cleanup removes:**
- ConfigMap: `cuda-fault-injection-lib`
- Init container: `compile-cuda-fault-lib`
- Volume: `cuda-fault-lib`, `cuda-fault-lib-source`
- Volume mounts in main container
- Environment variables: `LD_PRELOAD`, `CUDA_FAULT_INJECTION_ENABLED`, `CUDA_XID_TYPE`
- Node affinity constraints
- Restarts pods to apply clean configuration

```python
cuda_injector.cleanup_cuda_fault_injection(
    deployment_name="vllm-v1-disagg-router",
    namespace="dynamo-oviya",
    force_delete_pods=True  # Force-delete if pods are stuck
)
```

## Best Practices

1. **Use XID 79 for most tests**
   - Most common GPU failure
   - Well-supported by NVSentinel
   - Clear failure mode (GPU off bus)

2. **Combine XID + CUDA for realism**
   - XID: Triggers NVSentinel detection
   - CUDA: Causes realistic pod crashes
   - Together: Most accurate GPU failure simulation

3. **Test with NVSentinel deployed**
   - Validates automated detection
   - Validates automated remediation
   - Validates full recovery workflow

4. **Always cleanup**
   - Remove CUDA artifacts after tests
   - Delete API faults
   - Verify pods return to normal

5. **Pin pods to specific node**
   - Use `target_node` parameter
   - Simulates node-specific GPU failure
   - Prevents affecting all workers

6. **Verify XID in kernel logs**
   ```bash
   kubectl exec -n gpu-operator <pod> -- journalctl -k | grep "Xid.*79"
   ```

7. **Monitor pod crashes**
   ```bash
   kubectl get pods -n <namespace> -w
   kubectl logs <pod-name> | grep CUDA
   ```

8. **Test recovery completely**
   - Verify pods restart
   - Verify inference works
   - Check all CUDA artifacts removed

## Troubleshooting

### XID Not Appearing in Kernel Logs

**Check agent pod:**
```bash
kubectl logs -n fault-injection-system -l app=gpu-fault-injector
```

**Verify node name:**
```bash
kubectl get nodes
# Node name must match exactly
```

**Check GPU ID:**
```bash
# On the node
nvidia-smi -L
# GPU ID must exist (0-7 typically)
```

### CUDA Faults Not Working

**Verify library compiled:**
```bash
# Check init container logs
kubectl logs <pod-name> -c compile-cuda-fault-lib
```

**Check LD_PRELOAD set:**
```bash
kubectl exec <pod-name> -- env | grep LD_PRELOAD
# Should show: LD_PRELOAD=/cuda-fault/fake_cuda_xid79.so
```

**Verify environment variables:**
```bash
kubectl exec <pod-name> -- env | grep CUDA
# Should show:
# CUDA_FAULT_INJECTION_ENABLED=1
# CUDA_XID_TYPE=79
```

**Check library exists in pod:**
```bash
kubectl exec <pod-name> -- ls -la /cuda-fault/
# Should show: fake_cuda_xid79.so
```

### Pods Not Crashing

**Restart pods after CUDA injection:**
```bash
# Pods need to restart to load LD_PRELOAD
kubectl delete pod <pod-name> -n <namespace>
```

**Verify pod uses CUDA:**
```bash
# Only worker pods use CUDA (not frontend, planner, router)
kubectl get pods -n <namespace> -l nvidia.com/dynamo-component-type=worker
```

**Check library actually loaded:**
```bash
kubectl exec <pod-name> -- ldd /usr/bin/python3 | grep fake
# Should show fake_cuda_xid79.so
```

### Cleanup Issues

**ConfigMap stuck:**
```bash
kubectl delete configmap cuda-fault-injection-lib -n <namespace> --force --grace-period=0
```

**Pods stuck terminating:**
```bash
# Use force_delete_pods=True in cleanup
cuda_injector.cleanup_cuda_fault_injection(
    deployment_name, namespace, force_delete_pods=True
)
```

**Deployment not updating:**
```bash
# Check for 409 Conflicts (handled by retry logic)
kubectl logs -n fault-injection-system -l app=fault-injection-api | grep 409
```

### NVSentinel Not Detecting XID

**Check syslog-health-monitor:**
```bash
kubectl logs -n nvsentinel -l app=syslog-health-monitor | grep -i xid
```

**Verify health event created:**
```bash
kubectl get healthevents -n nvsentinel
```

**Check node annotations:**
```bash
kubectl get node <node-name> -o jsonpath='{.metadata.annotations}' | grep quarantine
```

## Advanced Topics

### Testing Multiple XIDs

```python
xid_types = [79, 48, 94, 95, 43, 74]

for xid in xid_types:
    print(f"Testing XID {xid}...")
    fault = client.inject_xid_error(node_name="worker-node-1", xid_type=xid)
    time.sleep(10)
    client.delete_fault(fault.fault_id)
    time.sleep(5)
```

### Testing Across Multiple Nodes

```python
from kubernetes import client, config

config.load_kube_config()
k8s = client.CoreV1Api()

# Get all GPU nodes
nodes = k8s.list_node(label_selector="nvidia.com/gpu.present=true")

for node in nodes.items:
    node_name = node.metadata.name
    print(f"Testing {node_name}...")
    
    fault = api_client.inject_xid_error(node_name=node_name, xid_type=79)
    # Run tests...
    api_client.delete_fault(fault.fault_id)
```

### Custom CUDA Error Codes

Modify `fake_cuda_xid79.c` to add custom error codes:

```c
// Add new XID type
{99, cudaErrorCustom, "Custom error description"},
```

Rebuild and redeploy:
```bash
cd cuda-fault-injection
make clean && make
# Update ConfigMap with new source
```

## Additional Resources

- **CUDA Fault Library Details**: [cuda-fault-injection/README.md](cuda-fault-injection/README.md)
- **Network Fault Testing**: [NETWORK_FAULT_GUIDE.md](NETWORK_FAULT_GUIDE.md)
- **General Setup**: [README.md](README.md)
- **Example Tests**: [examples/](examples/)

## Create Your Own Test

```python
# examples/my_gpu_test.py
from fault_injection_client import FaultInjectionClient
from helpers.cuda_fault_injection import CUDAFaultInjector
from test_helpers import Colors, get_config_from_env
import time

def test_my_gpu_fault():
    config = get_config_from_env()
    api_client = FaultInjectionClient(api_url=config['api_url'])
    cuda_injector = CUDAFaultInjector()
    
    try:
        # Inject XID
        fault = api_client.inject_xid_error(node_name="worker-node-1", xid_type=79)
        print(f"{Colors.YELLOW}XID injected{Colors.RESET}")
        
        # Inject CUDA faults
        cuda_injector.build_library()
        cuda_injector.create_configmap_with_library("dynamo-oviya")
        cuda_injector.patch_deployment_for_cuda_fault(
            "vllm-v1-disagg-router", "dynamo-oviya", "worker-node-1", 79
        )
        print(f"{Colors.YELLOW}CUDA faults injected{Colors.RESET}")
        
        # Your test logic here
        time.sleep(60)
        
        print(f"{Colors.GREEN}Test passed{Colors.RESET}")
        
    finally:
        # Always cleanup
        cuda_injector.cleanup_cuda_fault_injection(
            "vllm-v1-disagg-router", "dynamo-oviya", force_delete_pods=True
        )
        api_client.delete_fault(fault.fault_id)

if __name__ == "__main__":
    test_my_gpu_fault()
```

**Run:** `python3 scripts/run_test_incluster.py examples/my_gpu_test.py`

