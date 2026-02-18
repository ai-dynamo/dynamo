# Power Agent DaemonSet

## Overview

The Power Agent is a node-local service that enforces GPU power limits based on Kubernetes pod annotations. It runs as a DaemonSet on each GPU node and watches for pods with power limit annotations.

## How It Works

1. **Watches Pod Annotations**: The agent queries the Kubernetes API for pods running on its node that have the `dynamo.nvidia.com/gpu-power-limit` annotation.

2. **Maps Processes to Pods**: For each GPU, it:
   - Gets running processes via NVML
   - Maps each process PID to its pod UID by reading `/proc/{pid}/cgroup`
   - Applies the power limit if the pod has an annotation

3. **Applies Power Limits**: Uses NVML (`nvmlDeviceSetPowerManagementLimit`) to set the GPU power limit in hardware.

## Key Features

- **No kubectl exec**: Uses standard cgroup inspection (same pattern as cadvisor)
- **Kubernetes-native**: Deployed as a DaemonSet, managed by K8s
- **Secure**: Uses RBAC for pod queries, privileged mode only for NVML access
- **Automatic**: Reconciles every 15 seconds

## Deployment

The Power Agent is deployed via DaemonSet to all GPU nodes. See `deploy/power_agent/daemonset.yaml` for the manifest.

### Prerequisites

- Kubernetes 1.25+
- NVIDIA GPU Operator (or DCGM Exporter)
- RBAC permissions for `pods/get`, `pods/list`, `pods/watch`

### Building

```bash
cd components/power_agent
docker build -t dynamo/power-agent:v1.0.0 .
```

### Deploying

```bash
kubectl apply -f ../../deploy/power_agent/daemonset.yaml
```

## Environment Variables

- `NODE_NAME`: Required. The name of the node this agent is running on (injected by K8s).

## Architecture

This component is part of the power-aware autoscaling feature documented in `MR3_onefile.md`. It works together with:

- **SLA Planner**: Sets power limit annotations on worker pods
- **Prometheus**: Monitors GPU power consumption via DCGM
- **DCGM Exporter**: Exposes GPU metrics to Prometheus

## Security Considerations

- **Privileged Mode**: Required for NVML to change hardware power limits (unavoidable)
- **hostPID**: Required to read `/proc/{pid}/cgroup` for process-to-pod mapping
- **RBAC**: Minimal permissions (pods/get, pods/list, pods/watch) on all namespaces

## Troubleshooting

### Agent not applying limits

1. Check if DaemonSet is running: `kubectl get ds -n dynamo-system power-agent`
2. Check logs: `kubectl logs -n dynamo-system -l app=power-agent`
3. Verify pods have annotations: `kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}{"\n"}{end}'`

### Cgroup mapping issues

The regex pattern handles both cgroupfs and systemd drivers. If you see mapping failures, check:
- `/proc/{pid}/cgroup` format on your nodes
- Kubernetes version and cgroup version (v1 vs v2)

## References

- Design Document: `MR3_onefile.md` (Part 4: Refactored Architecture)
- Kubernetes DaemonSet Best Practices: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
- NVML Documentation: https://docs.nvidia.com/deploy/nvml-api/

