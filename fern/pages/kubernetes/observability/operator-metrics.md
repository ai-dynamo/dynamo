---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Dynamo Operator Metrics

## Overview

The Dynamo Operator exposes Prometheus metrics for monitoring its own health and performance. These metrics are separate from application metrics (frontend/worker) and provide visibility into:

- **Controller Reconciliation**: How efficiently controllers process DynamoGraphDeployments, DynamoComponentDeployments, and DynamoModels
- **Webhook Validation**: Performance and outcomes of admission webhook requests
- **Resource Inventory**: Current count of managed resources by state and namespace

## Prerequisites

The operator metrics feature requires the same monitoring infrastructure as application metrics. For detailed setup instructions, see the [Kubernetes Metrics Guide](./metrics.md#prerequisites).

**Quick checklist:**
- kube-prometheus-stack installed (for ServiceMonitor support)
- Prometheus and Grafana running
- Dynamo Operator installed via Helm

## Metrics Collection

### ServiceMonitor

Operator metrics are automatically collected via a ServiceMonitor, which is created by the Helm chart when `metricsService.enabled: true` (default).

**Unlike application metrics** (which use PodMonitor), the operator uses ServiceMonitor and requires no manual RBAC configuration. The operator's kube-rbac-proxy sidecar is configured with `--ignore-paths=/metrics` to allow Prometheus access.

To verify the ServiceMonitor is created:

```bash
kubectl get servicemonitor -n dynamo-system
```

### Disabling Metrics Collection

To disable operator metrics collection:

```bash
helm upgrade dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  --set dynamo-operator.metricsService.enabled=false
```

## Available Metrics

All metrics use the `dynamo_operator` namespace prefix.

### Reconciliation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_operator_reconcile_duration_seconds` | Histogram | `resource_type`, `namespace`, `result` | Duration of reconciliation loops |
| `dynamo_operator_reconcile_total` | Counter | `resource_type`, `namespace`, `result` | Total number of reconciliations |
| `dynamo_operator_reconcile_errors_total` | Counter | `resource_type`, `namespace`, `error_type` | Total reconciliation errors by type |

**Labels:**
- `resource_type`: `DynamoGraphDeployment`, `DynamoComponentDeployment`, `DynamoModel`, `DynamoGraphDeploymentRequest`, `DynamoGraphDeploymentScalingAdapter`
- `namespace`: Target namespace of the resource
- `result`: `success`, `error`, `requeue`
- `error_type`: `not_found`, `already_exists`, `conflict`, `validation`, `bad_request`, `unauthorized`, `forbidden`, `timeout`, `server_timeout`, `unavailable`, `rate_limited`, `internal`

### Webhook Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_operator_webhook_duration_seconds` | Histogram | `resource_type`, `operation` | Duration of webhook validation requests |
| `dynamo_operator_webhook_requests_total` | Counter | `resource_type`, `operation`, `result` | Total webhook admission requests |
| `dynamo_operator_webhook_denials_total` | Counter | `resource_type`, `operation`, `reason` | Total webhook denials with reasons |

**Labels:**
- `resource_type`: Same as reconciliation metrics
- `operation`: `CREATE`, `UPDATE`, `DELETE`
- `result`: `allowed`, `denied`
- `reason`: Validation failure reason (e.g., `immutable_field_changed`, `invalid_config`)

### Resource Inventory Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_operator_resources_total` | Gauge | `resource_type`, `namespace`, `status` | Current count of resources by state |

**Labels:**
- `resource_type`: `DynamoGraphDeployment`, `DynamoComponentDeployment`, `DynamoModel`, `DynamoGraphDeploymentRequest`, `DynamoGraphDeploymentScalingAdapter`
- `namespace`: Resource namespace
- `status`: Resource state derived from each CRD's status. Common values:
  - `"ready"` - Resource is healthy and operational (DCD, DM, DGDSA)
  - `"not_ready"` - Resource exists but is not operational (DCD, DM, DGDSA)
  - `"unknown"` - State cannot be determined (default for empty status)
  - DGD uses: `"pending"`, `"successful"`, `"failed"` from `.status.state`
  - DGDR uses: `"Pending"`, `"Profiling"`, `"Deploying"`, `"Ready"`, `"DeploymentDeleted"`, `"Failed"` from `.status.state`

## Example Queries

### Reconciliation Performance

```promql
# P95 reconciliation duration by resource type
histogram_quantile(0.95,
  sum by (resource_type, le) (
    rate(dynamo_operator_reconcile_duration_seconds_bucket[5m])
  )
)

# Reconciliation rate by result
sum by (resource_type, result) (
  rate(dynamo_operator_reconcile_total[5m])
)

# Error rate by type
sum by (resource_type, error_type) (
  rate(dynamo_operator_reconcile_errors_total[5m])
)
```

### Webhook Performance

```promql
# Webhook P95 latency
histogram_quantile(0.95,
  sum by (resource_type, le) (
    rate(dynamo_operator_webhook_duration_seconds_bucket[5m])
  )
)

# Webhook denial rate
sum by (resource_type, operation, reason) (
  rate(dynamo_operator_webhook_denials_total[5m])
)
```

### Resource Inventory

```promql
# Total resources by type and state
sum by (resource_type, status) (
  dynamo_operator_resources_total
)

# DynamoGraphDeployments by state
sum by (status) (
  dynamo_operator_resources_total{resource_type="DynamoGraphDeployment"}
)

# All resources by namespace and state
sum by (resource_type, namespace, status) (
  dynamo_operator_resources_total
)
```

## Grafana Dashboard

A pre-built Grafana dashboard is available for visualizing operator metrics.

### Dashboard Sections

1. **Reconciliation Metrics** (3 panels)
   - Reconciliation rate by resource type and result
   - P95 reconciliation duration
   - Reconciliation errors by type

2. **Webhook Metrics** (3 panels)
   - Webhook request rate by operation
   - P95 webhook duration
   - Webhook denials by reason

3. **Resource Inventory** (2 panels)
   - Resource inventory timeline by state and namespace (filterable by resource type)
   - Current resource count by state (filterable by resource type)

4. **Operational Health** (2 panels)
   - Reconciliation success rate gauges
   - Webhook admission success rate gauges

### Deploying the Dashboard

```bash
kubectl apply -f deploy/observability/k8s/grafana-operator-dashboard-configmap.yaml
```

The dashboard will automatically appear in Grafana (assuming you have the Grafana dashboard sidecar configured, which is included in kube-prometheus-stack).

### Finding the Dashboard

1. Port-forward to Grafana (if needed):
   ```bash
   kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
   ```

2. Log in to Grafana at http://localhost:3000

3. Navigate to **Dashboards** → Search for **"Dynamo Operator"**

### Dashboard Filters

The dashboard includes two filter variables:

- **Namespace**: View metrics across all namespaces or filter by specific ones (multi-select)
- **Resource Type**: Filter all panels by resource type or select "All" to see aggregated metrics across all CRDs (single select)

When "All" is selected for Resource Type, all panels will show data for all five managed CRDs with resource_type labels for differentiation.

## Accessing Metrics Directly

For instructions on accessing Prometheus and Grafana, see the [Kubernetes Metrics Guide](./metrics.md#viewing-the-metrics).

Once you have access to Prometheus, you can query operator metrics directly:

```bash
# Port-forward to Prometheus
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring

# Visit http://localhost:9090 and try queries like:
# - dynamo_operator_reconcile_total
# - dynamo_operator_webhook_requests_total
# - dynamo_operator_resources_total
```

## Troubleshooting

### Metrics Not Appearing in Prometheus

1. **Check ServiceMonitor exists:**
   ```bash
   kubectl get servicemonitor -n dynamo-system | grep operator
   ```

2. **Check ServiceMonitor is discovered by Prometheus:**
   - Go to Prometheus UI → Status → Targets
   - Look for `serviceMonitor/dynamo-system/dynamo-platform-dynamo-operator-operator`
   - Should show state: `UP`

3. **Check Prometheus selector configuration:**
   ```bash
   kubectl get prometheus -o yaml | grep serviceMonitorSelector
   ```
   Ensure `serviceMonitorSelectorNilUsesHelmValues: false` was set during kube-prometheus-stack installation.

### Dashboard Not Appearing in Grafana

1. **Check ConfigMap is created:**
   ```bash
   kubectl get configmap -n monitoring grafana-operator-dashboard
   ```

2. **Check ConfigMap has the label:**
   ```bash
   kubectl get configmap -n monitoring grafana-operator-dashboard -o jsonpath='{.metadata.labels.grafana_dashboard}'
   ```
   Should return `"1"`

3. **Check Grafana dashboard sidecar configuration:**
   ```bash
   kubectl get deployment -n monitoring prometheus-grafana -o yaml | grep -A 5 sidecar
   ```
   The sidecar should be configured to watch for `grafana_dashboard: "1"` label.

4. **Restart Grafana pod** to force dashboard refresh:
   ```bash
   kubectl rollout restart deployment/prometheus-grafana -n monitoring
   ```

## Related Documentation

- [Kubernetes Metrics Guide](./metrics.md) - Application metrics for frontends and workers
- [Dynamo Operator Guide](../dynamo-operator.md) - Operator architecture and deployment modes
- [Operator Webhooks](../webhooks.md) - Webhook validation details
