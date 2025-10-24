# Hardware Fault Tolerance Tests

End-to-end tests validating automatic fault detection and recovery for GPU failures and network partitions.

## What These Tests Do

**`test_gpu_health_check.py`**
- Validates NVSentinel's automated GPU fault recovery pipeline:
  - GPU XID error occurs → DCGM detects → NVSentinel cordons node → Pods evicted → Rescheduled to healthy nodes
- Falls back to manual simulation on hardware that doesn't support GPU reset (e.g., Azure A100)

**`test_network_partition_recovery.py`**
- Injects network partition between frontend and worker → Validates request recovery and rescheduling

## Prerequisites

- Kubernetes cluster with GPU nodes
- Dynamo deployment with worker pods running
- **NVSentinel** installed (GPU fault detection and remediation)
- **Standalone DCGM service** (required for NVSentinel to detect GPU XID errors)

### Quick Setup: Use Pre-configured Environment

The `dynamo-oviya` namespace on Azure AKS cluster `dynamo-dev` already has:
- NVSentinel installed and configured
- DCGM service running and connected
- Dynamo vLLM workers deployed

```bash
# Switch to pre-configured environment
kubectl config use-context dynamo-dev
kubectl config set-context --current --namespace=dynamo-oviya

# Run tests
cd dynamo
python3 -m pytest tests/fault_tolerance/hardware/ -v -s
```

### Manual Setup: Deploy DCGM for NVSentinel

If setting up a new environment, NVSentinel needs standalone DCGM (GPU Operator only provides DCGM Exporter):

```bash
kubectl apply -f tests/fault_tolerance/hardware/dcgm-daemonset.yaml
```

## Run Tests

```bash
cd dynamo

# Run all hardware fault tolerance tests
python3 -m pytest tests/fault_tolerance/hardware/ -v -s

# Run specific test
python3 -m pytest tests/fault_tolerance/hardware/test_gpu_health_check.py -v -s
```

## Troubleshooting

**Test skipped: "No GPU nodes with running worker pods found"**
```bash
# Deploy worker pods
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n <namespace>
kubectl get pods -l app=vllm-worker  # Wait for Running status
```

**Test skipped: "Frontend not reachable"**
```bash
# Port-forward in a separate terminal
kubectl port-forward svc/vllm-agg-frontend 8000:8000 -n <namespace>
```

**Check NVSentinel is detecting GPUs:**
```bash
kubectl logs -n nvsentinel -l app.kubernetes.io/name=gpu-health-monitor --tail=20
# Should see "DCGM_HEALTH_WATCH" messages with status PASS
```

