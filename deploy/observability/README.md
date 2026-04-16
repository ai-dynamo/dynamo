# Dynamo Observability

For detailed documentation on Observability (Prometheus metrics, tracing, and logging), please refer to [docs/observability/](../../docs/observability/).

## Dynamo Local Resource Monitor (Prometheus Exporter)

[`dynamo_local_resource_monitor.py`](dynamo_local_resource_monitor.py) is a Dynamo-specific Prometheus exporter that tracks per-process resource usage (VRAM, GPU utilization, PCIe bandwidth, CPU, disk I/O, network I/O) for Dynamo inference processes — labeled by model name, process identity, and PID.

**Run on the host machine** (not inside a container):

```bash
pip install psutil nvidia-ml-py prometheus-client
python3 dynamo_local_resource_monitor.py --host 0.0.0.0 --port 8051
```

Then verify metrics at `http://<host>:8051/metrics`.

If any dependencies are missing, the script prints the exact `pip install` command needed and exits.

> **Firewall note:** If your host runs UFW (or similar), you must allow port 8051 from the Docker Compose bridge network for `dynamo-resource-monitor` to scrape the exporter. See the [full documentation](../../docs/observability/local-resource-monitor.md#firewall-configuration) for details.

See the [full documentation](../../docs/observability/local-resource-monitor.md) for architecture details, Prometheus integration, and available metrics.
