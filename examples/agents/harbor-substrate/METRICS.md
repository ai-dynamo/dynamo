<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Collect Agent Substrate's native platform metrics through an OpenTelemetry
Collector and display them in an existing Prometheus/Grafana installation.
This configuration uses the public Substrate binaries. It adds no session
labels, changes no worker capacity, and enables no suspension policy.

## Prerequisites

- Complete the [Substrate integration setup](INSTALL.md).
- Install Prometheus Operator and Grafana. This example uses the
  `kube-prometheus-stack` release in namespace `monitoring`.
- Set `KUBE_CONTEXT` to the cluster context. Run commands from this directory.
- Stop active Harbor jobs before changing the telemetry configuration.

> [!WARNING]
> Restarting `ate-controller` propagates the telemetry settings to its worker
> Deployments and rolls their pods. Make this change while actors are idle.
> The worker count and resource requests remain unchanged.

## Deploy the metrics bridge

[metrics-bridge.yaml](metrics-bridge.yaml) creates a metrics-only Collector,
Service, ServiceMonitor, and ingress NetworkPolicy in `ate-system`. The
policy permits OTLP from Substrate components and workers, and scraping from
`monitoring`. No cluster-scoped permissions are required by the Collector.

```bash
kubectl --context "$KUBE_CONTEXT" apply -f metrics-bridge.yaml
kubectl --context "$KUBE_CONTEXT" -n ate-system rollout status \
  deployment/substrate-metrics-bridge --timeout=120s
```

The ServiceMonitor uses `release: kube-prometheus-stack`. Match that label to
your Prometheus selector if it differs. Collection and scraping run every five
seconds. Native CPU/memory sampling still runs approximately once per minute.

For a dedicated tutorial installation, point Substrate at this Collector.
These commands replace its telemetry destination and disable OTLP traces and
logs. Save the current configuration for cleanup. On a shared installation,
instead add a metrics export to the existing Collector and preserve its other
pipelines. The rc1 atelet relay requires a common destination for all signals,
so a separate metrics endpoint override cannot preserve the other destinations.

```bash
kubectl --context "$KUBE_CONTEXT" -n ate-system get configmap ate-otel-config \
  -o yaml > .local/ate-otel-config.before.yaml
kubectl --context "$KUBE_CONTEXT" -n ate-system patch configmap ate-otel-config \
  --type=merge -p '{"data":{
    "OTEL_EXPORTER_OTLP_ENDPOINT":"http://substrate-metrics-bridge.ate-system.svc.cluster.local:4317",
    "OTEL_METRIC_EXPORT_INTERVAL":"5000",
    "OTEL_METRIC_EXPORT_TIMEOUT":"5000",
    "OTEL_METRICS_EXPORTER":"otlp",
    "OTEL_TRACES_EXPORTER":"none",
    "OTEL_LOGS_EXPORTER":"none",
    "ATE_WORKER_OTEL_METRICS_EXPORTER":"otlp",
    "ATE_WORKER_OTEL_TRACES_EXPORTER":"none",
    "ATE_WORKER_OTEL_LOGS_EXPORTER":"none"
  }}'
kubectl --context "$KUBE_CONTEXT" -n ate-system rollout restart \
  daemonset/atelet deployment/ate-api-server deployment/atenet-router \
  deployment/atenet-egress deployment/ate-controller
```

This example disables Substrate OTLP traces/logs because the Collector has only
a metrics pipeline. It does not affect Dynamo's request/tool trace capture.
Workers normally send metrics through the node's atelet relay. The controller
also propagates the endpoint for their direct fallback.

Wait for the affected workloads and worker pools to become ready. Confirm both
`substrate-metrics-bridge` targets are **UP** in Prometheus before starting a
trial. A healthy Collector readiness probe alone does not prove metrics are
arriving. This query should return several emitting services:

```promql
count by (service_name) ({substrate_telemetry="otlp"})
```

## Import the dashboard

Import [substrate-dashboard.json](substrate-dashboard.json) through Grafana's
**Dashboards → New → Import** and select the Prometheus data source. The default
data-source UID is `prometheus`; the dashboard has a selector for other UIDs.

For kube-prometheus-stack installations with the dashboard sidecar enabled,
provision the same dashboard from a labeled ConfigMap:

```bash
kubectl --context "$KUBE_CONTEXT" -n monitoring create configmap \
  substrate-native-metrics-dashboard \
  --from-file=substrate-native-metrics.json=substrate-dashboard.json \
  --dry-run=client -o yaml > .local/substrate-dashboard-configmap.yaml
kubectl label --local -f .local/substrate-dashboard-configmap.yaml \
  grafana_dashboard=1 -o yaml > .local/substrate-dashboard-labeled.yaml
kubectl --context "$KUBE_CONTEXT" apply -f .local/substrate-dashboard-labeled.yaml
kubectl --context "$KUBE_CONTEXT" -n monitoring port-forward \
  service/kube-prometheus-stack-grafana 3007:80
```

Open `http://localhost:3007/d/substrate-native-metrics`. The dashboard contains
22 panels for collection health, worker state, lifecycle operations, scheduling,
routing, resource usage, image-cache lookups, checkpoints, restores, and parking.

## Validate with one task

Run one prepared-image Harbor trial using the existing model endpoint and
concurrency one. Select its time range in Grafana. Expect:

1. A worker changes from idle to partial or at capacity during execution.
2. Lifecycle metrics record create/resume, followed by cleanup operations.
3. Scheduler assignment and image-cache lookup counters advance.
4. A sufficiently long trial produces CPU/memory samples.
5. The worker returns to idle after cleanup.

Interpret the panels according to their native semantics:

- **Workers are pods, not actor slots.** Fleet gauges from API replicas are
  deduplicated with `max` before summing.
- **Peak sampled actors measures coverage.** It is not an authoritative actor
  inventory. Short activations can miss resource-sampling intervals.
- **Operation totals and means are cumulative since process start.** A newly
  emitted counter's first event is visible here, even when `increase()` cannot
  reconstruct it. The values are not restricted to the selected time range.
- **Restore and checkpoint phases overlap.** Do not add their durations.
- **Empty panels can be expected.** A cold-boot-only task need not restore a
  snapshot or park a request. No observed errors is also an empty error series.
- **Router latency excludes execution.** It ends when the worker endpoint is
  found; it is not model-inference or guest-tool duration.
- **Parking is aggregate.** The dashboard provides no per-session attribution;
  a parking interval can also include restoration after capacity becomes free.

Prometheus retains the history. The Collector's exporter expires series that
stop reporting after 30 seconds; native sampling adds its own delay. Check
exporter logs and scrape targets if all panels go empty together.

## Restore the previous telemetry configuration

After saving results and stopping actors, restore the complete ConfigMap data.
Replacing the data field also removes settings added by this walkthrough:

```bash
python3 -m venv .local/metrics-tools
.local/metrics-tools/bin/pip install pyyaml
.local/metrics-tools/bin/python - <<'PY'
import json
from pathlib import Path
import yaml
before = yaml.safe_load(Path('.local/ate-otel-config.before.yaml').read_text())
Path('.local/restore-otel.json').write_text(json.dumps([
    {'op': 'replace', 'path': '/data', 'value': before.get('data', {})}
]))
PY
kubectl --context "$KUBE_CONTEXT" -n ate-system patch configmap ate-otel-config \
  --type=json --patch-file .local/restore-otel.json
kubectl --context "$KUBE_CONTEXT" -n ate-system rollout restart \
  daemonset/atelet deployment/ate-api-server deployment/atenet-router \
  deployment/atenet-egress deployment/ate-controller
```

Wait for those components and worker pools to be ready. Confirm that the
previous telemetry destination receives data before deleting the bridge:

```bash
kubectl --context "$KUBE_CONTEXT" delete -f metrics-bridge.yaml
kubectl --context "$KUBE_CONTEXT" -n monitoring delete configmap \
  substrate-native-metrics-dashboard --ignore-not-found
```

Only use the saved configuration if no other operator changed telemetry during
the walkthrough. Otherwise, merge the restore with those changes first.
