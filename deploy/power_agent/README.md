# Power Agent Deployment

## Overview

This directory contains Kubernetes manifests for deploying the Power Agent DaemonSet, which enforces GPU power limits on each node based on pod annotations.

## Prerequisites

1. **Kubernetes Cluster** with GPU nodes
2. **NVIDIA GPU Operator** installed (provides DCGM Exporter with pod labels)
3. **kube-state-metrics** deployed
4. **Prometheus** deployed and configured

## Quick Start

### 1. Build the Power Agent Image

```bash
cd components/power_agent
docker build -t dynamo/power-agent:v1.0.0 .
docker push dynamo/power-agent:v1.0.0  # Push to your registry
```

### 2. Deploy the DaemonSet

```bash
kubectl apply -f deploy/power_agent/daemonset.yaml
```

### 3. Verify Deployment

```bash
# Check that DaemonSet is running on all GPU nodes
kubectl get ds -n dynamo-system

# Check pod status
kubectl get pods -n dynamo-system -l app=power-agent

# View logs
kubectl logs -n dynamo-system -l app=power-agent --tail=50
```

## Deployment Architecture

The DaemonSet creates:
- **Namespace**: `dynamo-system`
- **ServiceAccount**: `power-agent-sa`
- **ClusterRole**: `power-agent-role` (pods/get, pods/list, pods/watch)
- **ClusterRoleBinding**: `power-agent-binding`
- **DaemonSet**: `power-agent` (one pod per GPU node)

## Configuration

### Environment Variables

- `NODE_NAME`: Automatically injected by Kubernetes (spec.nodeName)

### Resource Limits

- CPU: 100m (request), 200m (limit)
- Memory: 128Mi (request), 256Mi (limit)

### Security Context

- **privileged: true**: Required for NVML to change hardware power limits
- **hostPID: true**: Required to read /proc/{pid}/cgroup for process-to-pod mapping

## How It Works

1. **Planner** sets annotations on worker pods:
   ```yaml
   metadata:
     annotations:
       dynamo.nvidia.com/gpu-power-limit: "250"
   ```

2. **Power Agent** (running on each node):
   - Queries K8s API for pods on its node with power limit annotations
   - For each GPU, gets running processes via NVML
   - Maps PIDs to pod UIDs via /proc/{pid}/cgroup
   - Applies power limits via NVML if pod has annotation

3. **NVML** applies the power limit in hardware

## Troubleshooting

### DaemonSet not starting

```bash
# Check events
kubectl describe ds -n dynamo-system power-agent

# Common issues:
# - Image pull failures (check image name/tag)
# - Node selector mismatch (nodes must have nvidia.com/gpu.present=true label)
# - RBAC issues (check ServiceAccount and ClusterRole)
```

### Power limits not being applied

```bash
# Check agent logs
kubectl logs -n dynamo-system -l app=power-agent | grep "Setting power limit"

# Verify pods have annotations
kubectl get pods -o yaml | grep "dynamo.nvidia.com/gpu-power-limit"

# Check if pods are scheduled to GPU nodes
kubectl get pods -o wide | grep gpu-node

# Verify GPU processes
kubectl exec -it <worker-pod> -- nvidia-smi
```

### Cgroup mapping failures

The agent uses regex to parse `/proc/{pid}/cgroup`. If you see errors:

```bash
# Check cgroup format on your nodes
kubectl exec -it <agent-pod> -n dynamo-system -- cat /proc/1/cgroup

# Expected patterns:
# cgroupfs: /kubepods/burstable/pod<uuid>/...
# systemd: /kubepods.slice/kubepods-burstable-pod<uuid>.slice/...
```

## Integration with SLA Planner

The Power Agent works with the SLA Planner's power-aware autoscaling feature:

### Enable Power Awareness in Planner

```yaml
# In planner deployment
args:
  - --enable-power-awareness
  - --total-gpu-power-limit=2000
  - --prefill-engine-gpu-power-limit=250
  - --decode-engine-gpu-power-limit=250
```

### Verify End-to-End

```bash
# 1. Check planner is setting annotations
kubectl logs <planner-pod> | grep "Applied power limits"

# 2. Check agent is applying limits
kubectl logs -n dynamo-system -l app=power-agent | grep "Setting power limit"

# 3. Verify GPU power limits
kubectl exec <worker-pod> -- nvidia-smi -q -d POWER | grep "Power Limit"
```

## Monitoring

### Key Metrics to Track

- **DCGM_FI_DEV_POWER_USAGE**: Current GPU power consumption
- **planner:predicted_num_p**: Planned prefill replicas
- **planner:predicted_num_d**: Planned decode replicas

### Prometheus Queries

```promql
# Total cluster GPU power
sum(DCGM_FI_DEV_POWER_USAGE)

# Power by component
avg(DCGM_FI_DEV_POWER_USAGE) by (label_nvidia_com_dynamo_component)

# Power budget utilization
sum(DCGM_FI_DEV_POWER_USAGE) / <total_power_limit> * 100
```

## Security Considerations

### Why Privileged Mode?

The Power Agent runs in privileged mode because:
- NVML requires privileged access to change hardware power limits
- This is the only way to apply power limits to GPUs
- Alternative approaches (nvidia-smi in pods) have worse security posture

### RBAC Permissions

The agent only needs:
- `pods/get`, `pods/list`, `pods/watch` on all namespaces
- No `pods/exec` or `pods/patch` permissions required

### Risk Mitigation

- Agent only runs on GPU nodes (nodeSelector)
- RBAC limits blast radius to pod queries
- Logs all power limit changes for audit trail
- Reconciles every 15s (limits impact of misconfiguration)

## Uninstalling

```bash
kubectl delete -f deploy/power_agent/daemonset.yaml
```

This removes:
- DaemonSet and all pods
- ServiceAccount and RBAC
- Namespace (if empty)

Note: GPU power limits will remain at last-set values until GPUs are reset or driver is reloaded.

## References

- Design Document: `MR3_onefile.md`
- Power Agent Code: `components/power_agent/`
- SLA Planner Integration: `components/src/dynamo/planner/`

