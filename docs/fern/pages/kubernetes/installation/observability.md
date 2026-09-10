---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Observability
subtitle: Install Prometheus, Grafana, DCGM exporter, Loki, and Alloy for Dynamo deployments
---

Dynamo emits three signals: **metrics** for Prometheus, **logs** as structured text or JSONL, and
**traces** over OpenTelemetry Protocol (OTLP). This page installs the metrics and logging backends.
To collect traces, provide an OTLP collector and trace backend that the Dynamo pods can reach; this
reference stack does not install them.

## Signals at a glance

| Signal | Turn it on with | Collected by |
|--------|-----------------|--------------|
| Metrics | `DYN_SYSTEM_PORT` (workers/router); the frontend serves metrics on its HTTP port | Prometheus |
| Traces | `OTEL_EXPORT_ENABLED=true` + an OTLP endpoint | User-provided collector and trace backend |
| Logs | `DYN_LOGGING_CONSOLE_FORMAT=jsonl` for console JSONL; `OTEL_EXPORT_ENABLED=true` for OTLP export | Alloy and Loki for stdout; user-provided collector and backend for OTLP |
| FPM traces | `DYN_FPM_TRACE=1` or `--fpm-trace` | Rotating gzip JSONL files |
| Health status | `/live` and `/health` endpoints | Supervisors or Kubernetes probes |
| Request traces | `DYN_REQUEST_TRACE=1` or `DYN_REQUEST_TRACE_RECORDS` | Files, NATS, stderr, or OTLP logs |

`OTEL_EXPORT_ENABLED` is the master switch for OTLP-exported traces and log records. It does not
control console output: the reference logging stack collects JSONL from pod stdout with Alloy and
sends it to Loki. Dynamo supports OTLP over `grpc` and `http/protobuf`. Use
`OTEL_EXPORTER_OTLP_ENDPOINT` for a shared collector or signal-specific endpoint variables for
separate destinations. For the full variable catalog, including defaults and Kubernetes operator
presets, see
[Environment Variables](../../reference/observability/environment-variables.mdx).

For scheduler telemetry, see [Observe a Local Deployment](../../cli/operations/observability.mdx#capture-forward-pass-metrics). For replay metadata or request payload capture, see [Observe a Local Deployment](../../cli/operations/observability.mdx#capture-and-replay-requests).

> [!NOTE]
> Prometheus metric families are registered lazily: each label set is created the first time it fires, so a freshly-started process shows empty metric families until the first relevant request. An idle cluster does not mean scraping is broken.

## Kubernetes stack

On Kubernetes, install the monitoring backends once, then enable signals per deployment. See
[Observability](../operations/observability.mdx) under Operations for the per-deployment steps. The
Dynamo platform chart creates `PodMonitor` resources that select operator-managed pods, so the
one-time work is installing Prometheus, the exporters, and the logging stack.

If your platform already provides an OTLP collector and trace backend, use the
[distributed trace export workflow](../operations/observability.mdx#export-distributed-traces-and-logs)
to connect Dynamo to them.

### kube-prometheus-stack

Install Prometheus, Grafana, and the Prometheus Operator. The selector flags let Prometheus discover
`PodMonitor` and `ServiceMonitor` resources created outside its own Helm release, including the
ones created by the Dynamo platform chart:

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
> Install Prometheus before Dynamo when possible. The Dynamo install command in the
> [Installation Guide](install-dynamo.md#kube-prometheus-stack) sets
> `dynamo-operator.dynamo.metrics.prometheusEndpoint`, and the platform chart detects the Prometheus
> Operator CRDs and creates its `PodMonitor` resources in that same install.
>
> An older form of the discovery flags used `--set ...podMonitorNamespaceSelector.matchLabels=null` (and the same for `probeNamespaceSelector`). The `--set-json '...={}'` form above is equivalent and preferred; use one form or the other consistently, not both.

If Dynamo is already installed, installing Prometheus does not retroactively create the monitors or
configure the endpoint. Upgrade the existing platform release with the same chart version and any
other values from its original install, adding these settings:

```bash
helm upgrade dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --reuse-values \
  --set "dynamo-operator.dynamo.metrics.prometheusEndpoint=http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090" \
  --set "dynamo-operator.dynamo.metrics.podMonitors.enabled=true"
```

This command uses `RELEASE_VERSION` and `NAMESPACE` from the
[Installation Guide](install-dynamo.md#install-the-dynamo-platform). Setting `podMonitors.enabled`
to `true` makes the upgrade explicit instead of relying on chart auto-detection.

kube-prometheus-stack bundles **node-exporter** by default, which supplies the node CPU, system load, and container resource panels on the Dynamo dashboard. Verify it is running:

```bash
kubectl get daemonset -A | grep node-exporter
```

If you run a custom Prometheus instead of kube-prometheus-stack, deploy node-exporter separately as a DaemonSet.

### PodMonitors created by the Dynamo chart

Whether the platform chart creates PodMonitors is controlled by `dynamo-operator.dynamo.metrics.podMonitors.enabled`, which is tri-state:

| Value | Behavior |
|---|---|
| `null` (default) | Auto-detect. Creates PodMonitors only when the prometheus-operator CRDs exist in the cluster, which requires a live cluster lookup and so works with `helm install` and `helm upgrade`. |
| `true` | Always create. Use with `helm template` and GitOps workflows, where CRDs cannot be detected and auto-detect would silently create nothing. |
| `false` | Never create, even when prometheus-operator is installed. |

When enabled, the chart creates these PodMonitors:

| PodMonitor | Selects | Scrapes |
|---|---|---|
| `dynamo-frontend` | `component-type: frontend` | port `http` (`frontendPath`, default `/metrics`) |
| `dynamo-worker` | `worker` / `decode` / `prefill` | port `system` (worker metrics), ports `system-0` / `system-1` when present (GMS shadow-failover), port `http` when present (frontend sidecar, also `frontendPath`), and port `nixl` when `dynamo-operator.dynamo.metrics.podMonitors.nixlTelemetry=true` |
| `dynamo-planner` | `component-type: planner` | port `metrics` |
| `dynamo-router` | `component-type: default` | port 9090 (`DYN_SYSTEM_PORT`), addressed by relabeling rather than a named container port |

Each endpoint matches on a named container port, so a pod that does not declare that port produces no scrape target for it rather than a failing one.

A frontend can move its own Prometheus endpoint by setting `DYN_HTTP_SVC_METRICS_PATH`. When it does, set `dynamo-operator.dynamo.metrics.podMonitors.frontendPath` to the same value. It defaults to `/metrics` and applies only to the two endpoints that scrape a frontend HTTP service: the `dynamo-frontend` monitor and the `dynamo-worker` sidecar endpoint on port `http`. The remaining endpoints scrape the system status server, which always serves `/metrics`.

> [!WARNING]
> A mismatch fails silently. Prometheus scrapes the old path, receives a 404, and every `dynamo_frontend_*` series disappears while the pods stay healthy and the targets still exist. The SLA planner reads TTFT and ITL from those series, so it reads zero and never scales up.

### DCGM exporter (GPU metrics)

GPU-utilization panels are populated by [dcgm-exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html). Check whether it is already running (the NVIDIA GPU Operator installs it as part of its monitoring components):

```bash
kubectl get daemonset -A | grep dcgm-exporter
```

If the output is empty, install it via the GPU Operator or the standalone dcgm-exporter chart.

### Loki + Alloy (logging stack)

To collect pod logs into Grafana Loki, install Loki and the Grafana Alloy collector. This reference setup suits development and testing clusters (including Minikube and MicroK8s); use a high-availability configuration for production.

Set the namespaces once. The supplied Grafana datasource uses the in-namespace service name
`loki-gateway`, so this reference setup keeps Grafana and Loki together in `monitoring`:

```bash
export MONITORING_NAMESPACE=monitoring    # where Prometheus, Grafana, and Loki are installed
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

The Alloy values file forwards logs to Loki, restricts collection to `$DYN_NAMESPACE`, and maps the `nvidia.com/dynamo-component-type` and `nvidia.com/dynamo-graph-deployment-name` pod labels into Loki labels so the logging dashboard can filter by deployment and component. Applying the Grafana datasource and logging dashboard, and enabling JSONL logging on a deployment, are covered in [Observability](../operations/observability.mdx#configure-structured-logs) under Operations.
