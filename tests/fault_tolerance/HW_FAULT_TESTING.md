# Hardware Fault Tolerance Testing

## Quick Start

```python
async def test_xid79(namespace, deployment_spec):
    async with ManagedDeployment(namespace, deployment_spec, enable_hw_faults=True) as d:
        assert await d.inference_request()  # baseline
        
        node = d.get_hw_fault_target_node()
        await d.setup_cuda_passthrough()      # inject LD_PRELOAD into pods
        await d.toggle_cuda_faults(True)      # arm the fault trigger
        await d.inject_hw_fault("xid", 79)    # fire XID via kernel module
        
        while not d.is_node_cordoned(node):   # wait for NVSentinel
            await asyncio.sleep(5)
        
        await d.cleanup_cuda_spec_without_restart()
        await d.wait_for_pods_on_healthy_nodes(exclude_node=node)
        
        assert await d.inference_request()    # recovery verified
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Runner                               │
│  ManagedDeployment → HWFaultManager → Fault Injection API       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ fault-injection- │  │ gpu-fault-       │  │ NVSentinel    │  │
│  │ api (Deployment) │  │ injector-kernel  │  │ (DaemonSet)   │  │
│  │                  │  │ (DaemonSet)      │  │               │  │
│  │ Patches pods     │  │ Injects XID via  │  │ Detects XID,  │  │
│  │ with LD_PRELOAD  │  │ /dev/nvidia-uvm  │  │ cordons node  │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Both CUDA + XID?

| Component | What It Does | Why Needed |
|-----------|--------------|------------|
| **CUDA Passthrough** | Patches pod with `LD_PRELOAD=libcuda_fault.so` | Makes CUDA calls return errors → app sees GPU failure |
| **XID Injection** | Writes to `/dev/nvidia-uvm` via kernel module | Generates syslog entry → NVSentinel detects & acts |

**Both required because:**
- XID alone: NVSentinel cordons node, but app continues using "broken" GPU until pod evicted
- CUDA alone: App fails, but NVSentinel never detects (no syslog entry) → no node remediation
- **Together**: Realistic simulation where app fails AND infrastructure responds

## API Reference

### ManagedDeployment (enable_hw_faults=True)

| Method | Purpose |
|--------|---------|
| `get_hw_fault_target_node()` | Auto-detect node running GPU pods |
| `setup_cuda_passthrough(xid_type=79)` | Inject LD_PRELOAD into pods (no restart) |
| `toggle_cuda_faults(True/False)` | Arm/disarm CUDA error returns |
| `inject_hw_fault("xid", xid_type=79)` | Fire XID error to syslog |
| `is_node_cordoned(node)` | Check if NVSentinel cordoned the node |
| `cleanup_cuda_spec_without_restart()` | Remove LD_PRELOAD patches |
| `wait_for_pods_on_healthy_nodes(exclude_node)` | Wait for rescheduling |

### Supported XID Types

| XID | Name | Severity |
|-----|------|----------|
| 79 | GPU Fell Off Bus | Critical - requires reboot |
| 48 | DBE (Double Bit Error) | Critical |
| 94/95 | Contained ECC Error | Recoverable |
| 43/74 | Other GPU errors | Varies |

## Cluster Prerequisites

1. **NVSentinel** deployed with `node-drainer` and `syslog-health-monitor`
2. **Fault Injection Service** deployed:
   ```bash
   kubectl apply -f tests/fault_tolerance/hardware/fault_injection_service/deploy/
   ```

## Key Files

| File | Purpose |
|------|---------|
| [`tests/utils/managed_deployment.py`](../utils/managed_deployment.py) | Main test harness |
| [`tests/utils/hw_fault_helpers.py`](../utils/hw_fault_helpers.py) | HWFaultManager implementation |
| [`tests/fault_tolerance/deploy/test_hw_faults.py`](deploy/test_hw_faults.py) | Reference test |
| [`tests/fault_tolerance/deploy/conftest.py`](deploy/conftest.py) | Pytest fixtures |
| [`hardware/fault_injection_service/`](hardware/fault_injection_service/) | K8s manifests & Dockerfiles |

## Limitations

- **GPU required**: Tests must run against cluster with real GPUs
- **Privileged access**: Kernel module needs root on nodes
- **Single node at a time**: XID injection targets one node per test
- **Recovery time**: NVSentinel detection ~15s, node drain ~2min (configurable)
- **Not reversible**: XID 79 triggers node cordon; manual uncordon or reboot needed

## Running Tests

```bash
# On cluster with fault-injection-service + NVSentinel
pytest tests/fault_tolerance/deploy/test_hw_faults.py \
    --enable-hw-faults \
    --namespace=<your-namespace> \
    -v -s
```

