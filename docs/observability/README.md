---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Observability
subtitle: Install the monitoring stack and enable metrics, logging, and tracing for Dynamo deployments
---

Dynamo emits three signals — **metrics** (Prometheus), **logs** (structured text or JSONL, exportable to Loki), and **traces** (OpenTelemetry, exportable to Tempo). This page installs the backends that collect them. Enabling each signal is a matter of setting a few environment variables per process or deployment.

## Signals at a glance

| Signal | Turn it on with | Collected by |
|--------|-----------------|--------------|
| Metrics | `DYN_SYSTEM_PORT` (workers/router); the frontend serves metrics on its HTTP port | Prometheus |
| Traces | `OTEL_EXPORT_ENABLED=true` + `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Tempo |
| Logs | `DYN_LOGGING_JSONL=true` (+ `OTEL_EXPORT_ENABLED` to export) | Loki |

`OTEL_EXPORT_ENABLED` is the master switch for both traces and logs — without it, neither leaves the process even when Tempo and Loki are healthy. OTLP endpoints must be gRPC listeners (Dynamo's exporter does not speak OTLP/HTTP). For the full variable catalog — defaults, per-signal grouping, and the Kubernetes operator presets — see [Environment Variables](../reference/observability/environment-variables.mdx).

> [!NOTE]
> Prometheus metric families are registered lazily: each label set is created the first time it fires, so a freshly-started process shows empty metric families until the first relevant request. An idle cluster does not mean scraping is broken.

## Kubernetes stack

On Kubernetes, install the monitoring backends once, then enable signals per deployment — see [Observability](../kubernetes/observability/metrics.md) under Operations for the per-deployment steps. The Dynamo operator wires metrics scraping automatically (it adds a `PodMonitor` to every managed pod), so the only one-time work is installing Prometheus, the exporters, and the logging stack.

### kube-prometheus-stack

Install Prometheus, Grafana, and the Prometheus Operator. The selector flags let Prometheus discover `PodMonitor` and `ServiceMonitor` resources created outside its own Helm release (such as the ones the Dynamo operator creates):

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set-json 'prometheus.prometheusSpec.podMonitorNamespaceSelector={}' \
  --set-json 'prometheus.prometheusSpec.probeNamespaceSelector={}'
```

> [!NOTE]
> The Dynamo install command in the [Installation Guide](../kubernetes/installation-guide.md#kube-prometheus-stack) sets `dynamo-operator.dynamo.metrics.prometheusEndpoint` to point the operator at this Prometheus. Set it there, in one pass, rather than here.
>
> An older form of the discovery flags used `--set ...podMonitorNamespaceSelector.matchLabels=null` (and the same for `probeNamespaceSelector`). The `--set-json '...={}'` form above is equivalent and preferred; use one form or the other consistently, not both.

kube-prometheus-stack bundles **node-exporter** by default, which supplies the node CPU, system load, and container resource panels on the Dynamo dashboard. Verify it is running:

```bash
kubectl get daemonset -A | grep node-exporter
```

If you run a custom Prometheus instead of kube-prometheus-stack, deploy node-exporter separately as a DaemonSet.

### DCGM exporter (GPU metrics)

GPU-utilization panels are populated by [dcgm-exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html). Check whether it is already running (the NVIDIA GPU Operator installs it as part of its monitoring components):

```bash
kubectl get daemonset -A | grep dcgm-exporter
```

If the output is empty, install it via the GPU Operator or the standalone dcgm-exporter chart.

### Loki + Alloy (logging stack)

To collect pod logs into Grafana Loki, install Loki and the Grafana Alloy collector. This reference setup suits development and testing clusters (including Minikube and MicroK8s); use a high-availability configuration for production.

Set the namespaces once:

```bash
export MONITORING_NAMESPACE=monitoring    # where Loki is installed
export DYN_NAMESPACE=dynamo-system        # where Dynamo is installed
```

Install Loki in single-binary mode (local MinIO storage), then install Alloy pointed at it:

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Loki
helm install --values deploy/observability/logging/values/loki-values.yaml \
  loki grafana/loki -n $MONITORING_NAMESPACE

# Alloy collector (k8s-monitoring chart), templated with the namespaces above
envsubst < deploy/observability/logging/values/alloy-values.yaml > alloy-custom-values.yaml
helm install --values alloy-custom-values.yaml \
  alloy grafana/k8s-monitoring -n $MONITORING_NAMESPACE
```

The Alloy values file forwards logs to Loki, restricts collection to `$DYN_NAMESPACE`, and maps the `nvidia.com/dynamo-component-type` and `nvidia.com/dynamo-graph-deployment-name` pod labels into Loki labels so the logging dashboard can filter by deployment and component. Applying the Grafana datasource and logging dashboard, and enabling JSONL logging on a deployment, are covered in [Observability](../kubernetes/observability/metrics.md#logging) under Operations.

## Local stack (single machine)

Bring up the full stack on one machine with Docker Compose for local development and demos.

Install these on your machine:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Starting the Observability Stack

Dynamo provides a Docker Compose-based observability stack that includes Prometheus, Grafana, Tempo, Loki, an OpenTelemetry Collector, and various exporters for metrics, tracing, logging, and visualization.

From the Dynamo root directory:

```bash
# Start infrastructure (NATS, etcd)
docker compose -f dev/docker-compose.yml up -d

# Start observability stack (Prometheus, Grafana, Tempo, DCGM GPU exporter, NATS exporter)
docker compose -f dev/docker-observability.yml up -d
```

For detailed setup instructions and configuration, see [Prometheus + Grafana Setup](prometheus-grafana.md).

### Signal how-tos

Once the stack is running, each signal has a how-to guide:

- [Prometheus + Grafana](prometheus-grafana.md) — visualize metrics on the local stack
- [Tracing](tracing.md) — distributed tracing with OpenTelemetry and Tempo
- [Logging](logging.md) — structured logging and OTLP log export to Loki
- [Health Checks](health-checks.md) — component health and readiness endpoints

For the metric, label, and variable catalogs, see the [Observability reference](../reference/observability/metrics-catalog.mdx). To create your own metrics, see the [Metrics Developer Guide](metrics-developer-guide.md).

---

## Topology

This provides:
- **Prometheus** on `http://localhost:9090` - metrics collection and querying
- **Grafana** on `http://localhost:3000` - visualization dashboards (username: `dynamo`, password: `dynamo`)
- **Tempo** on `http://localhost:3200` - distributed tracing backend
- **Loki** on `http://localhost:3100` - log aggregation backend
- **OpenTelemetry Collector** on `http://localhost:4317` (gRPC) / `http://localhost:4318` (HTTP) - receives OTLP signals and routes traces to Tempo and logs to Loki
- **DCGM Exporter** on `http://localhost:9401/metrics` - GPU metrics
- **NATS Exporter** on `http://localhost:7777/metrics` - NATS messaging metrics

### Service Relationship Diagram
```mermaid
graph TD
    BROWSER[Browser] -->|:3000| GRAFANA[Grafana :3000]
    subgraph DockerComposeNetwork [Network inside Docker Compose]
        NATS_PROM_EXP[nats-prom-exp :7777 /metrics] -->|:8222/varz| NATS_SERVER[nats-server :4222, :6222, :8222]
        PROMETHEUS[Prometheus server :9090] -->|:2379/metrics| ETCD_SERVER[etcd-server :2379, :2380]
        PROMETHEUS -->|:9401/metrics| DCGM_EXPORTER[dcgm-exporter :9401]
        PROMETHEUS -->|:7777/metrics| NATS_PROM_EXP
        PROMETHEUS -->|:8000/metrics| DYNAMOFE[Dynamo HTTP FE :8000]
        PROMETHEUS -->|:8081/metrics| DYNAMOBACKEND[Dynamo backend :8081]
        DYNAMOFE --> DYNAMOBACKEND
        DYNAMOFE -->|OTLP :4317| OTEL_COLLECTOR[OTel Collector :4317/:4318]
        DYNAMOBACKEND -->|OTLP :4317| OTEL_COLLECTOR
        OTEL_COLLECTOR -->|traces| TEMPO[Tempo :3200]
        OTEL_COLLECTOR -->|logs| LOKI[Loki :3100]
        GRAFANA -->|:9090/query API| PROMETHEUS
        GRAFANA -->|:3200/query API| TEMPO
        GRAFANA -->|:3100/query API| LOKI
    end
```

The dcgm-exporter service in the Docker Compose network is configured to use port 9401 instead of the default port 9400. This adjustment is made to avoid port conflicts with other dcgm-exporter instances that may be running simultaneously. Such a configuration is typical in distributed systems like SLURM.

### Configuration Files

The following configuration files are located in the `dev/observability/` directory:
- [docker-compose.yml](../../dev/docker-compose.yml): Defines NATS and etcd services
- [docker-observability.yml](../../dev/docker-observability.yml): Defines Prometheus, Grafana, Tempo, and exporters
- [prometheus.yml](../../dev/observability/prometheus.yml): Contains Prometheus scraping configuration
- [grafana-datasources.yml](../../dev/observability/grafana-datasources.yml): Contains Grafana datasource configuration
- [otel-collector.yaml](../../dev/observability/otel-collector.yaml): OpenTelemetry Collector configuration (routes traces to Tempo, logs to Loki)
- [loki.yaml](../../dev/observability/loki.yaml): Loki log aggregation configuration
- [loki-datasource.yml](../../dev/observability/loki-datasource.yml): Grafana Loki datasource with trace ID linking to Tempo
- [grafana_dashboards/dashboard-providers.yml](../../dev/observability/grafana_dashboards/dashboard-providers.yml): Contains Grafana dashboard provider configuration
- [grafana_dashboards/dynamo.json](../../dev/observability/grafana_dashboards/dynamo.json): Engine-agnostic per-model dashboard covering frontend, KV-router, and worker metrics. Filterable by `model`. See the [per-model dashboard guide](prometheus-grafana.md#per-model-dynamo-dashboard) for details.
- [grafana_dashboards/dcgm-metrics.json](../../dev/observability/grafana_dashboards/dcgm-metrics.json): Contains Grafana dashboard configuration for DCGM GPU metrics
- [grafana_dashboards/kvbm.json](../../dev/observability/grafana_dashboards/kvbm.json): Contains Grafana dashboard configuration for KVBM metrics
