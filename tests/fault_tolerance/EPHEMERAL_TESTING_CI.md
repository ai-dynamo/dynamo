# Fault Tolerance Testing Framework

Ephemeral deployment testing framework for hardware fault injection with CI integration.

## Overview

This framework provides:
- **ManagedDeployment** - Ephemeral Dynamo deployments for isolated testing
- **HWFaultManager** - Hardware fault injection lifecycle management
- **CI-ready tests** - Pytest-based tests that can run in GitHub Actions

## Architecture

```
+-------------------------------------------------------------------------+
|                           Test Execution                                 |
|                                                                          |
|  +-------------------------------------------------------------------+  |
|  |  ManagedDeployment (async context manager)                         |  |
|  |                                                                    |  |
|  |  - Creates ephemeral namespace                                     |  |
|  |  - Deploys dynamo-platform (NATS + etcd) if needed                |  |
|  |  - Creates DynamoGraphDeployment                                  |  |
|  |  - Waits for pods to be ready                                     |  |
|  |  - Injects HW faults (if enabled)                                 |  |
|  |  - Collects logs/metrics on exit                                  |  |
|  |  - Cleans up everything                                           |  |
|  +-------------------------------------------------------------------+  |
|                              |                                           |
|                              v                                           |
|  +-------------------------------------------------------------------+  |
|  |  HWFaultManager                                                    |  |
|  |                                                                    |  |
|  |  - Builds CUDA shim library                                       |  |
|  |  - Patches deployments with LD_PRELOAD                            |  |
|  |  - Manages port-forward to fault-injection-api                    |  |
|  |  - Injects XID faults via API                                     |  |
|  |  - Toggles CUDA faults without restarts                           |  |
|  |  - Cleans up (uncordons nodes, removes artifacts)                 |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## Quick Start

### Prerequisites for Local Testing

**kubectl context:** The tests use your current kubectl context. Ensure you're connected to a GPU cluster:

```bash
# Check current context
kubectl config current-context

# Switch to GPU cluster context (example)
kubectl config use-context my-gpu-cluster

# Verify GPU nodes are available
kubectl get nodes -l nvidia.com/gpu.present=true

# Verify fault-injection-system is deployed
kubectl get pods -n fault-injection-system
```

**Environment variables (optional):**
```bash
# Override kubeconfig location
export KUBECONFIG=/path/to/kubeconfig

# Override default namespace
export FAULT_TEST_NAMESPACE=my-test-namespace
```

### Run Tests Locally

```bash
# Basic HW fault test
pytest tests/fault_tolerance/deploy/test_hw_faults.py \
  --enable-hw-faults \
  --namespace=hw-fault-test \
  -v -s

# With specific backend
pytest tests/fault_tolerance/deploy/test_hw_faults.py \
  --enable-hw-faults \
  --hw-fault-backend=vllm \
  --namespace=hw-fault-test \
  -v -s

# Skip NATS/etcd restart (faster, if cluster is clean)
pytest tests/fault_tolerance/deploy/test_hw_faults.py \
  --enable-hw-faults \
  --skip-service-restart \
  -v -s
```

### Using ManagedDeployment

```python
from tests.utils.managed_deployment import ManagedDeployment, DeploymentSpec

# Load deployment spec
spec = DeploymentSpec("examples/backends/vllm/deploy/disagg_router.yaml")
spec.name = "my-test"
spec.set_model("Qwen/Qwen3-0.6B")
spec.set_image("nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0")

# Run test with ephemeral deployment
async with ManagedDeployment(
    namespace="test-namespace",
    deployment_spec=spec,
    log_dir="./logs",
    enable_hw_faults=True,
    hw_fault_config={"xid_type": 79}
) as deployment:

    # Setup CUDA passthrough (pods restart once)
    await deployment.setup_cuda_passthrough(xid_type=79)
    await deployment.wait_for_all_pods_ready(timeout=300)

    # Run baseline tests...

    # Enable faults (no restart)
    await deployment.toggle_cuda_faults(enable=True)

    # Inject XID for NVSentinel
    fault_id = await deployment.inject_hw_fault(fault_type="xid", xid_type=79)

    # Wait for NVSentinel to cordon node
    assert deployment.is_node_cordoned(target_node)

    # Cleanup and recover
    await deployment.cleanup_cuda_spec_without_restart()
    await deployment.wait_for_pods_on_healthy_nodes(exclude_node=target_node)
```

## CI Integration

### GitHub Actions Workflow

```yaml
name: Hardware Fault Tolerance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:
    inputs:
      backend:
        description: 'Backend to test'
        default: 'vllm'
        type: choice
        options: [vllm, sglang, trtllm]

jobs:
  hw-fault-tests:
    runs-on: self-hosted-gpu  # Requires GPU cluster access
    timeout-minutes: 60

    steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[test]"
        pip install kubernetes kubernetes-asyncio kr8s

    - name: Setup kubeconfig
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy fault-injection-system
      run: |
        kubectl apply -f tests/fault_tolerance/hardware/fault_injection_service/deploy/
        kubectl wait --for=condition=available deployment/fault-injection-api \
          -n fault-injection-system --timeout=300s

    - name: Run HW fault tests
      run: |
        pytest tests/fault_tolerance/deploy/test_hw_faults.py \
          --enable-hw-faults \
          --hw-fault-backend=${{ github.event.inputs.backend || 'vllm' }} \
          --namespace=ci-hw-fault-${{ github.run_id }} \
          --skip-service-restart \
          -v -s \
          --tb=short \
          --junit-xml=results/hw-fault-tests.xml
      timeout-minutes: 45

    - name: Upload test results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: test-results
        path: results/

    - name: Cleanup namespace
      if: always()
      run: |
        kubectl delete namespace ci-hw-fault-${{ github.run_id }} --ignore-not-found
```

### Prerequisites for CI

1. **Self-hosted runner with:**
   - kubectl configured for GPU cluster
   - At least 2 GPU nodes (for reschedule testing)
   - Network access to container registries

2. **Cluster requirements:**
   - Dynamo operator installed (cluster-wide)
   - NVSentinel deployed with all modules
   - fault-injection-system namespace deployed

3. **Secrets:**
   - `KUBECONFIG` - Base64-encoded kubeconfig

### Pytest Configuration

The HW fault test options (`--enable-hw-faults`, `--hw-fault-backend`, `--namespace`, `--skip-service-restart`) are already configured in [`tests/fault_tolerance/conftest.py`](conftest.py).

## Test Phases

A typical HW fault test follows this flow:

| Phase | Description | Duration |
|-------|-------------|----------|
| 1. Deploy | Create namespace, deploy NATS/etcd, create DGD | ~2-3 min |
| 2. CUDA Setup | Patch deployment, wait for pods | ~5-7 min |
| 3. Baseline | Verify inference works | ~30 sec |
| 4. Inject Fault | XID + CUDA toggle | ~5 sec |
| 5. Verify Fault | Check inference fails | ~30 sec |
| 6. NVSentinel | Wait for node cordon | ~30 sec |
| 7. Cleanup Spec | Remove CUDA artifacts from DGD | ~5 sec |
| 8. Eviction | Wait for node-drainer | ~2 min |
| 9. Recovery | Wait for new pods | ~5-7 min |
| 10. Verify | Check inference recovers | ~30 sec |
| **Total** | | **~20-25 min** |

## Key Classes

### DeploymentSpec

Wrapper for DynamoGraphDeployment YAML:

```python
spec = DeploymentSpec("path/to/deployment.yaml")
spec.name = "test-deployment"
spec.set_model("Qwen/Qwen3-0.6B")
spec.set_image("nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0")
spec.set_tensor_parallel(2)  # TP=2
spec.set_logging(enable_jsonl=True, log_level="debug")
```

### ManagedDeployment

Async context manager for ephemeral deployments:

```python
async with ManagedDeployment(
    namespace="test",
    deployment_spec=spec,
    log_dir="./logs",
    enable_hw_faults=True,
    hw_fault_config={"xid_type": 79},
    skip_service_restart=True,  # Faster if cluster is clean
) as deployment:
    # Deployment is ready here
    pods = deployment.get_pods()
    pf = deployment.port_forward(pods["Frontend"][0], 8000)
    # ...
# Cleanup happens automatically
```

### HWFaultManager

Hardware fault lifecycle management:

```python
# Usually accessed via ManagedDeployment, but can use directly:
from tests.utils.hw_fault_helpers import HWFaultManager, HWFaultConfig

manager = HWFaultManager(
    namespace="test",
    deployment_name="my-deployment",
    config=HWFaultConfig(xid_type=79),
)

await manager.setup()
await manager.setup_cuda_passthrough(xid_type=79)
await manager.toggle_cuda_faults(enable=True)
fault_id = await manager.inject_xid_fault(xid_type=79)
manager.wait_for_node_cordon(timeout=180)
await manager.cleanup()
```

## Troubleshooting

### Local Testing Issues

```bash
# Check kubectl context
kubectl config current-context
kubectl cluster-info

# Verify namespace doesn't exist (or is clean)
kubectl get namespace hw-fault-test

# Check GPU node availability
kubectl get nodes -l nvidia.com/gpu.present=true -o wide

# Verify fault-injection-system
kubectl get pods -n fault-injection-system
kubectl logs -n fault-injection-system -l app=fault-injection-api --tail=50
```

### CI-Specific Issues

**Test timeout:**
- Increase `timeout-minutes` in workflow
- Check if pods are stuck in `Pending` (resource constraints)
- Verify image pull succeeds

**Namespace not cleaned:**
- Add `if: always()` to cleanup step
- Use unique namespace per run: `ci-hw-fault-${{ github.run_id }}`

**NVSentinel not cordoning:**
- Verify NVSentinel pods are running
- Check `syslog-health-monitor` logs
- Ensure test namespace is in `node-drainer.userNamespaces`

**Pods not rescheduling:**
- Need at least 2 GPU nodes
- Check other nodes have available GPU resources
- Verify node affinity was removed from DGD spec

### General Debugging

```bash
# Get test pod logs
kubectl logs -n <test-namespace> -l nvidia.com/dynamo-component-type=worker --tail=100

# Check events in namespace
kubectl get events -n <test-namespace> --sort-by='.lastTimestamp'

# Describe stuck pods
kubectl describe pod -n <test-namespace> <pod-name>

# Check NVSentinel status
kubectl get pods -n nvsentinel
kubectl logs -n nvsentinel -l app.kubernetes.io/name=node-drainer --tail=50
```

## Files Reference

| File | Purpose |
|------|---------|
| [`tests/utils/managed_deployment.py`](../utils/managed_deployment.py) | Core ephemeral deployment framework |
| [`tests/utils/hw_fault_helpers.py`](../utils/hw_fault_helpers.py) | HW fault manager and config |
| [`tests/fault_tolerance/deploy/test_hw_faults.py`](deploy/test_hw_faults.py) | Integration test example |
| [`tests/fault_tolerance/hardware/fault_injection_service/`](hardware/fault_injection_service/) | Fault injection components |
| [`tests/fault_tolerance/hardware/fault_injection_service/FAULT_INJECTION_OVERVIEW.md`](hardware/fault_injection_service/FAULT_INJECTION_OVERVIEW.md) | Fault injection service overview |
| [`tests/fault_tolerance/hardware/fault_injection_service/NETWORK_PARTITIONS.md`](hardware/fault_injection_service/NETWORK_PARTITIONS.md) | Network partition injection docs |
| [`tests/fault_tolerance/hardware/fault_injection_service/agents/GPU_XID_FAULT_INJECTION.md`](hardware/fault_injection_service/agents/GPU_XID_FAULT_INJECTION.md) | XID injection agent docs |
| [`tests/fault_tolerance/hardware/fault_injection_service/cuda_fault_injection/CUDA_SHIM_LIBRARY.md`](hardware/fault_injection_service/cuda_fault_injection/CUDA_SHIM_LIBRARY.md) | CUDA shim library docs |
