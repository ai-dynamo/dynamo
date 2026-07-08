---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Observability
subtitle: Enable metrics, logging, and tracing on a Kubernetes deployment, load the Grafana dashboards, and query the signals
---

This page covers the day-to-day observability actions for a running Dynamo deployment: turning each signal on or off per deployment, loading the Grafana dashboards, and querying the results. It assumes the monitoring stack (kube-prometheus-stack, DCGM exporter, and the Loki + Alloy logging stack) is already installed — see [Observability](../../observability/README.md) in the Installation section for those one-time steps.

For what each metric, label, and variable *means*, see the reference: [Metrics Catalog](../../reference/observability/metrics-catalog.mdx), [Metric Labels](../../reference/observability/metric-labels.mdx), [Environment Variables](../../reference/observability/environment-variables.mdx), and [Operator Metrics](../../reference/observability/operator-metrics.mdx).

## Enable signals per deployment

### Metrics

Metrics are **on by default**. The operator adds a `PodMonitor` and labels every pod `nvidia.com/metrics-enabled: "true"`, so Prometheus discovers and scrapes frontend and worker `/metrics` endpoints with no deployment-level field required. Both components expose OpenMetrics-format metrics: the frontend on its HTTP port, workers on their system port.

To opt a deployment **out** of metrics collection, set the `nvidia.com/enable-metrics: "false"` annotation:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
  annotations:
    nvidia.com/enable-metrics: "false"
spec:
  # …
```

### NIXL telemetry

To add NIXL transfer metrics (populated only during disaggregated serving or multimodal embedding transfers), set `NIXL_TELEMETRY_ENABLE: "y"` on the worker component. NIXL exposes its metrics on a separate port (`NIXL_TELEMETRY_PROMETHEUS_PORT`, default `19090`). See [Environment Variables](../../reference/observability/environment-variables.mdx#system-and-metrics) for the full NIXL variable set.

```yaml
spec:
  services:
    VllmDecodeWorker:
      componentType: worker
      extraPodSpec:
        mainContainer:
          env:
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
```

### Logging

To emit structured JSONL logs (required for the Loki logging stack), set the logging variables on the DGD. Graph-level `spec.envs` applies to every component; a per-component `env` entry overrides it. See [Environment Variables](../../reference/observability/environment-variables.mdx#logging) for the full set and their meanings.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-logging
spec:
  envs:                                # applied to every component
    - name: DYN_LOGGING_JSONL
      value: "true"
    - name: DYN_LOG
      value: "info"
  services:
    Frontend:
      componentType: frontend
      extraPodSpec:
        mainContainer:
          env:                         # applied to this component only
            - name: DYN_LOG
              value: "info,dynamo_runtime::system_status_server:trace"
```

### Traces and logs export

To export traces to Tempo and logs to Loki via OTLP, set the OpenTelemetry variables on the DGD. `OTEL_EXPORT_ENABLED` is the master switch — without it, traces and logs never leave the process. See [Environment Variables](../../reference/observability/environment-variables.mdx#opentelemetry-traces-and-logs).

```yaml
spec:
  envs:
    - name: OTEL_EXPORT_ENABLED
      value: "true"
    - name: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
      value: "http://tempo.observability.svc.cluster.local:4317"
```

## View dashboards

Load each dashboard by applying its ConfigMap. Each is labeled `grafana_dashboard: "1"`, so the Grafana sidecar (included in kube-prometheus-stack) discovers and imports it automatically.

```bash
# Application dashboard (frontend, KV-router, worker metrics)
kubectl apply -n monitoring -f deploy/observability/grafana-dynamo-dashboard-configmap.yaml

# Operator dashboard (reconciliation, webhooks, resource inventory)
kubectl apply -f deploy/observability/grafana-operator-dashboard-configmap.yaml

# Logging dashboard + Loki datasource
kubectl apply -n monitoring -f deploy/observability/logging/grafana/loki-datasource.yaml
kubectl apply -n monitoring -f deploy/observability/logging/grafana/logging-dashboard.yaml
```

The application dashboard includes panels for frontend request rates, time to first token, inter-token latency, request duration, input/output sequence lengths, GPU utilization (via DCGM), node CPU and system load, and per-pod CPU and memory.

### Open Grafana and Prometheus

```bash
# Grafana credentials
export GRAFANA_USER=$(kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-user}" | base64 --decode)
export GRAFANA_PASSWORD=$(kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode)
echo "Grafana user: $GRAFANA_USER / password: $GRAFANA_PASSWORD"

# Port-forward Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

Visit `http://localhost:3000` and find the Dynamo dashboards under **Dashboards**. To query Prometheus directly:

```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring
```

Visit `http://localhost:9090` and use the queries below.

To view logs, open **Home > Dashboards > Dynamo Logs** in Grafana. The dashboard filters by DynamoGraphDeployment, namespace, and component type (frontend, worker, and so on).

## Useful queries

### Application metrics

```promql
# Total frontend requests
dynamo_frontend_requests_total

# Time-to-first-token distribution
dynamo_frontend_time_to_first_token_seconds_bucket
```

### Derived signals

These cluster-wide totals combine per-stage frontend gauges into the signals operators most often want. `dynamo_frontend_stage_requests` has no `model` label, so these cannot be split by model; add `by (pod)` or `by (instance)` for per-pod visibility. See [Stage values](../../reference/observability/metric-labels.mdx#stage-values) for what each stage covers.

```promql
# Requests waiting for a worker to start generating (the old "queued" semantic)
sum(dynamo_frontend_stage_requests{stage=~"preprocess|route|dispatch"})

# Requests currently being processed by a backend worker (authoritative worker-side gauge)
sum(dynamo_component_inflight_requests{dynamo_component="backend",dynamo_endpoint="generate"})

# Router saturation — spikes when workers can't be selected fast enough
sum(dynamo_frontend_stage_requests{stage="route"})

# Backend prefill saturation — spikes when the backend is slow to produce first tokens
sum(dynamo_frontend_stage_requests{stage="dispatch"})
```

### Operator metrics

Operator metrics use the `dynamo_operator_*` prefix and are scraped via a ServiceMonitor (created by the Helm chart), separate from the application PodMonitor. See [Operator Metrics](../../reference/observability/operator-metrics.mdx) for the full catalog and label sets.

```promql
# P95 reconciliation duration by resource type
histogram_quantile(0.95,
  sum by (resource_type, le) (
    rate(dynamo_operator_reconcile_duration_seconds_bucket[5m])
  )
)

# Reconciliation errors by type
sum by (resource_type, error_type) (
  rate(dynamo_operator_reconcile_errors_total[5m])
)

# Webhook denial rate
sum by (resource_type, operation, reason) (
  rate(dynamo_operator_webhook_denials_total[5m])
)

# Managed resources by type and state
sum by (resource_type, status) (
  dynamo_operator_resources_total
)
```

## Troubleshooting

### Metrics not appearing in Prometheus

1. Check the monitor exists — `PodMonitor` for application metrics, `ServiceMonitor` for operator metrics:

   ```bash
   kubectl get podmonitor,servicemonitor -n dynamo-system
   ```

2. Confirm Prometheus discovered the target: in the Prometheus UI, go to **Status → Targets** and look for the Dynamo job in state `UP`.

3. Check the Prometheus selector configuration. The discovery of monitors outside the Prometheus release requires `podMonitorSelectorNilUsesHelmValues=false` (and the equivalent for ServiceMonitors), set during kube-prometheus-stack installation:

   ```bash
   kubectl get prometheus -o yaml | grep -iE "podMonitorSelector|serviceMonitorSelector"
   ```

> [!NOTE]
> Prometheus metric families are registered lazily: each label set is created the first time it fires, so a freshly-started process shows empty metric families until the first relevant request. An idle cluster does not mean scraping is broken.

### Dashboard not appearing in Grafana

1. Confirm the ConfigMap was created and carries the discovery label:

   ```bash
   kubectl get configmap -n monitoring <dashboard-configmap-name> \
     -o jsonpath='{.metadata.labels.grafana_dashboard}'   # should return "1"
   ```

2. Confirm the Grafana dashboard sidecar is watching for that label:

   ```bash
   kubectl get deployment -n monitoring prometheus-grafana -o yaml | grep -A5 sidecar
   ```

3. Restart Grafana to force a refresh:

   ```bash
   kubectl rollout restart deployment/prometheus-grafana -n monitoring
   ```
