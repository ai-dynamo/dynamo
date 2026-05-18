# Power Agent

Privileged DaemonSet that enforces per-GPU NVML power caps on Dynamo worker pods.

## How it works

1. Every 15 seconds, the agent lists pods on the current node via the K8s API.
2. For each physical GPU, it calls `nvmlDeviceGetComputeRunningProcesses()` to get the host PIDs.
3. For each PID it reads `/proc/{pid}/cgroup` to extract the pod UID.
4. It looks up the pod's `dynamo.nvidia.com/gpu-power-limit` annotation.
5. It calls `nvmlDeviceSetPowerManagementLimit(handle, watts × 1000)`.

## Deployment

```bash
kubectl apply -f deploy/power_agent/rbac.yaml
kubectl apply -f deploy/power_agent/daemonset.yaml
```

## Troubleshooting

Verify annotations are being set on pods:

```bash
kubectl get pods -l nvidia.com/dynamo-graph-deployment=<dgd-name> \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}{"\n"}{end}'
```

Check Power Agent logs:

```bash
kubectl logs -l app=dynamo-power-agent -n <namespace> --tail=100
```

Key Prometheus metrics:
- `dynamo_power_agent_applied_limit_watts{gpu="N"}` — cap currently applied per GPU
- `dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"}` — multi-pod-per-GPU conflicts (should be 0)
- `dynamo_power_agent_safe_default_applied_total` — times the safe default was used
- `dynamo_power_agent_apply_failures_total` — NVML apply failures

## Graceful shutdown

The DaemonSet sets `terminationGracePeriodSeconds: 30`. On SIGTERM, the agent
restores default TGP on all managed GPUs before exiting. If the agent is
SIGKILL'd, the GPU caps persist until the next reconcile cycle. On startup,
the agent restores default TGP on idle GPUs it previously managed (UUID-gated,
see `managed_gpus.json`).
