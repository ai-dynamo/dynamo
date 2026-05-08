# Example Grafana Dashboards

This directory contains example Grafana dashboards for Dynamo observability. These are starter files that you can use as references for building your own custom dashboards.

- `dynamo.json` - General Dynamo dashboard showing software and hardware metrics
- `sglang.json` - SGLang engine metrics (request latency, throughput, cache) and HiCache KV cache metrics (GPU/CPU tier usage, eviction/load-back, PIN count)
- `disagg-dashboard.json` - Dashboard for disaggregated serving - See [DASHBOARD_METRICS.md](DASHBOARD_METRICS.md) for detailed documentation on all metrics and panels
- `dcgm-metrics.json` - GPU metrics dashboard using DCGM exporter data
- `kvbm.json` - KV Block Manager metrics dashboard
- `dynamo-planner.json` - Planner observability covering all 6 `dynamo_planner_*` metric families (DEP-XXXX). Recommended dashboard during the `scheduling.use_orchestrator` flag-flip canary — the EXECUTE outcomes row catches connector silent-skip regressions, plugin circuit-state row catches external plugin failures. See [docs/components/planner/observability.md](../../../docs/components/planner/observability.md) for the metric reference.
- `temp-loki.json` - Logging dashboard for Loki integration
- `dashboard-providers.yml` - Configuration file for dashboard provisioning

For setup instructions and usage, see [Observability Documentation](../../../docs/observability/).

For Kubernetes deployment setup, see [../k8s/MONITORING_SETUP.md](../k8s/MONITORING_SETUP.md).
