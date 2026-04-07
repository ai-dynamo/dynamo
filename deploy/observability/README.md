# Dynamo Observability

For detailed documentation on Observability (Prometheus metrics, tracing, and logging), please refer to [docs/observability/](../../docs/observability/).

## GPU System Monitor

[`gpu_monitor.py`](gpu_monitor.py) is a self-contained, real-time system monitor served as a web dashboard. It tracks per-process GPU memory, utilization, PCIe bandwidth, CPU usage, disk I/O, and network I/O — useful for profiling Dynamo deployments during inference.

**Run on the host machine** (not inside a container):

```bash
pip install flask flask-socketio psutil pynvml
python3 gpu_monitor.py --host 0.0.0.0 --port 9999
```

If any dependencies are missing, the script prints the exact `pip install` command needed and exits.

Then open `http://<host>:9999` in a browser. See the [full documentation](../../docs/observability/gpu-system-monitor.md) for architecture details, chart layout, and feature descriptions.
