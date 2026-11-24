# Example Grafana Dashboards

This directory contains example Grafana dashboards for Dynamo observability. These are starter files that you can use as references for building your own custom dashboards.

- `dynamo.json` - Per-model view across the Dynamo stack: frontend KPIs (RPS, TTFT/ITL/E2E percentiles, ISL/OSL distributions, outcome breakdown), KV-router internals (per-worker active blocks, hit rate, routing-overhead breakdown), worker-side request rate/duration, and an Admission Pressure row. Filterable by `model`.
- `sglang.json` - SGLang engine metrics (request latency, throughput, cache) and HiCache KV cache metrics (GPU/CPU tier usage, eviction/load-back, PIN count)
- `disagg-dashboard.json` - Dashboard for disaggregated serving - See [DASHBOARD_METRICS.md](DASHBOARD_METRICS.md) for detailed documentation on all metrics and panels
- `dcgm-metrics.json` - GPU metrics dashboard using DCGM exporter data
- `temp-loki.json` - Logging dashboard for Loki integration
- `dashboard-providers.yml` - Configuration file for dashboard provisioning

## Admission Pressure

The `dynamo.json` Admission Pressure row separates router-policy queues and queue-limit rejections, TCP work queues and worker-pool occupancy, permit-wait p95, and engine waiting requests. It uses existing metrics available in Dynamo v1.6.2; it does not enable admission controls or add new metrics.

The **Frontend Targets** and **Worker Targets** selectors filter only this row. They list scrape instances with router queue metrics and generate request metrics, respectively, for the selected model. TCP metrics cover the entire selected worker process, including any other models or endpoints it serves; they are not per-model engine metrics. Keep independent frontend queue views separate, and do not add values across queue layers to count unique requests or infer a root cause.

Missing series remain **No data**, not zero. Router panels require KV-router queue metrics; a disabled or idle queue can still export zero. Permit-wait percentiles require completed permit acquisitions in the rate interval. The engine panel supports vLLM and SGLang; SGLang requires `--enable-metrics`. Other engines do not populate that panel.

For setup instructions and usage, see [Observability Documentation](../../../docs/fern/pages/kubernetes/installation/observability.md).

For Kubernetes deployment setup, see [deploy/observability/MONITORING_SETUP.md](../../../deploy/observability/MONITORING_SETUP.md).
