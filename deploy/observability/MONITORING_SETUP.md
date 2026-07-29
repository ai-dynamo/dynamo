# Monitoring Setup for Dynamo Disaggregated Deployment

## Prerequisites

The k8s cluster must be created with GPU operator configured to enable DCGM ServiceMonitor for Prometheus metrics collection.

```bash
helm upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set operator.defaultRuntime=containerd \
  --set gdrcopy.enabled=true \
  --set dcgmExporter.serviceMonitor.enabled=true \
  --set dcgmExporter.serviceMonitor.additionalLabels.release=prometheus \
  --wait --timeout=600s
```

The key settings for monitoring are:
- `dcgmExporter.serviceMonitor.enabled=true` - Enables ServiceMonitor creation
- `dcgmExporter.serviceMonitor.additionalLabels.release=prometheus` - Adds label for Prometheus discovery

## Installation

Once the cluster is properly configured, run:

```bash
./setup-monitoring.sh
```

This script will:
1. Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
2. Configure Prometheus to discover pod monitors across all namespaces
3. Update Dynamo operator with Prometheus endpoint
4. Configure DCGM custom metrics for NVLink profiling
5. Verify DCGM ServiceMonitor exists (created by GPU operator)
6. Deploy Grafana disaggregated dashboard ConfigMap (auto-imported by Grafana sidecar)
7. Provide Grafana credentials

### Remote write (optional)

To push scraped metrics to an external Prometheus (or any remote-write–compatible endpoint), set these environment variables before running the script:

| Variable                      | Description                                  | Default     |
|-------------------------------|----------------------------------------------|-------------|
| `REMOTE_WRITE_HOST`           | Host of the remote write endpoint            | — (disabled if unset) |
| `REMOTE_WRITE_PORT`           | Port of the remote write endpoint           | `9091`      |
| `REMOTE_WRITE_METRIC_REGEX`   | Prometheus regex for metric names to send   | `^agent_.*` |

Metrics are sent only when `REMOTE_WRITE_HOST` is set. The script configures Prometheus with a single `remote_write` target: `http://<host>:<port>/api/v1/write`. Only metrics whose name matches `REMOTE_WRITE_METRIC_REGEX` are sent; all others are filtered out. You can use any valid Prometheus regex (e.g. `^agent_.*`, `^nixl_.*`, `^(agent_|nixl_).*`).

Examples:

```bash
# Default filter: agent_.*
REMOTE_WRITE_HOST=prom.example.com REMOTE_WRITE_PORT=9091 ./setup-monitoring.sh

# Custom filter: only nixl_ metrics
REMOTE_WRITE_HOST=prom.example.com REMOTE_WRITE_METRIC_REGEX='^nixl_.*' ./setup-monitoring.sh

# Multiple patterns: agent_ or nixl_
REMOTE_WRITE_HOST=prom.example.com REMOTE_WRITE_METRIC_REGEX='^(agent_|nixl_).*' ./setup-monitoring.sh
```

## Verification

Check that GPU metrics are flowing:

```bash
# Verify DCGM ServiceMonitor exists and is owned by ClusterPolicy
kubectl get servicemonitor -n gpu-operator nvidia-dcgm-exporter -o yaml | grep -A 5 ownerReferences

# Query Prometheus for GPU metrics
kubectl exec -n monitoring prometheus-prometheus-kube-prometheus-prometheus-0 -- \
  wget -q -O- 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL'
```

Expected: ServiceMonitor should show `ownerReferences` pointing to ClusterPolicy, and query should return 8+ GPU series.
