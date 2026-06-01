# Example Grafana Dashboards

This directory contains example Grafana dashboards for Dynamo observability. These are starter files that you can use as references for building your own custom dashboards.

- `dynamo.json` - Engine-agnostic per-model view across the Dynamo stack: frontend KPIs (RPS, TTFT/ITL/E2E percentiles, ISL/OSL distributions, outcome breakdown), KV-router internals (per-worker active blocks, hit rate, routing-overhead breakdown), and worker-side request rate/duration. Filterable by `model`.
- `sglang.json` - SGLang engine metrics (request latency, throughput, cache) and HiCache KV cache metrics (GPU/CPU tier usage, eviction/load-back, PIN count)
- `disagg-dashboard.json` - Dashboard for disaggregated serving - See [DASHBOARD_METRICS.md](DASHBOARD_METRICS.md) for detailed documentation on all metrics and panels
- `dcgm-metrics.json` - GPU metrics dashboard using DCGM exporter data
- `kvbm.json` - KV Block Manager metrics dashboard
- `temp-loki.json` - Logging dashboard for Loki integration
- `dashboard-providers.yml` - Configuration file for dashboard provisioning

For setup instructions and usage, see [Observability Documentation](../../../docs/observability/).

For Kubernetes deployment setup, see [deploy/observability/MONITORING_SETUP.md](../../../deploy/observability/MONITORING_SETUP.md).
