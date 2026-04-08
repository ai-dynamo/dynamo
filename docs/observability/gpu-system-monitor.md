---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: GPU System Monitor
---

# GPU System Monitor

A self-contained, real-time system monitor that runs as a web dashboard. It tracks GPU memory, utilization, PCIe bandwidth, CPU usage, disk I/O, and network I/O with per-process breakdowns — useful for profiling Dynamo deployments and understanding resource consumption during inference.

No build step is required. The server embeds the full HTML/JS client and serves it directly. The client loads [Plotly.js](https://cdn.plot.ly/) and [Socket.IO](https://cdn.socket.io/) from public CDNs, so the browser needs internet access (or you can vendor these files locally for air-gapped environments).

> [!IMPORTANT]
> This tool is designed to run on the **host machine**, not inside a container. It needs direct access to GPU devices and host-level process information.

## Quick Start

```bash
pip install flask flask-socketio psutil pynvml
python3 deploy/observability/gpu_monitor.py --host 0.0.0.0 --port 9999
```

Then open `http://<host>:9999` in a browser.

If any dependencies are missing, the script prints the exact `pip install` command needed and exits.

> [!NOTE]
> If the dashboard is not reachable from your browser, check that the port (default 8051) is not blocked by a firewall on the host. For example: `sudo ufw allow 9999/tcp` or the equivalent for your firewall.

## Architecture

The monitor uses a multiprocess design to bypass the Python GIL:

| Process | Sample Rate | What It Collects |
|---------|-------------|------------------|
| 1 per GPU | ~10/s | PCIe TX/RX throughput via pynvml |
| 1 shared | ~5/s | CPU %, GPU memory/utilization/temperature, network I/O via psutil + pynvml |
| 1 shared | ~1/s | Aggregate disk I/O via psutil |
| Main | — | Polls pipes from above, pushes deltas to browser via WebSocket |

The browser client batches all pending deltas into a single `Plotly.extendTraces()` call per animation frame (`requestAnimationFrame`), eliminating redundant redraws.

## Dashboard Layout

**Per GPU** (repeated for each GPU):

| Row | Left Y-Axis | Right Y-Axis |
|-----|------------|--------------|
| 1 | GPU Memory by Process (GiB, stacked area) | PCIe TX/RX (GB/s) |
| 2 | GPU Utilization (%, filled) | Temperature (°C) |

**System-wide:**

| Row | Left Y-Axis | Right Y-Axis |
|-----|------------|--------------|
| 3 | CPU Usage by Process Group (%, stacked area, top-5 + "Other") | — |
| 4 | Disk I/O (MB/s, aggregate read/write) | Network I/O (MB/s) |

## Features

- **Per-process GPU memory tracking** with automatic identification of vLLM, SGLang, and TensorRT-LLM processes (walks the parent process chain for `--model`/`--model-path` arguments)
- **Top-N pruning** (default 12): excess processes are grouped into a gray "Other" bucket
- **Stable per-PID color assignment** with slot recycling when processes exit
- **GPU memory ceiling** shown as a solid red line
- **Time range buttons** (1m, 2m, 5m, 10m, 15m) with synced zoom/pan across all x-axes
- **Progressive data coarsening**: old data is auto-downsampled to keep the client responsive at long time ranges (MAX\_POINTS=3000)
- **Adaptive push rate** based on zoom level (75ms at 1m, 1000ms at 15m)
- **State persistence**: saves/restores metrics to `~/.cache/gpu_monitor/metrics.json` on SIGTERM/SIGINT so data survives restarts
- **Pause/resume** and PNG snapshot export
- **Responsive layout**: charts auto-reflow on browser window resize
- **Graceful degradation**: works with 0–8 GPUs; shows CPU + disk only when no NVIDIA GPUs are present

## CLI Options

```text
usage: gpu_monitor.py [-h] [--port PORT] [--host HOST]

options:
  --port PORT   Server port (default: 8051)
  --host HOST   Bind address (default: 127.0.0.1)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | HTTP server |
| `flask-socketio` | WebSocket push |
| `psutil` | CPU, disk, network metrics |
| `pynvml` (`nvidia-ml-py`) | GPU metrics via NVML |

## Relationship to Other Observability Tools

This monitor is designed for **interactive, real-time debugging** on a single node — complementing the Prometheus + Grafana stack described in [Prometheus + Grafana Setup](prometheus-grafana.md), which is better suited for long-term historical monitoring across a cluster.

| | GPU System Monitor | Prometheus + Grafana |
|---|---|---|
| Deployment | Single `python3` command | Docker Compose stack |
| Scope | One node | Multi-node cluster |
| Data retention | In-memory + optional JSON cache | Prometheus TSDB |
| Per-process GPU memory | Yes | No (DCGM reports per-GPU totals) |
| PCIe bandwidth | Yes | Via DCGM exporter |
| Best for | Live debugging, profiling | Dashboards, alerting, historical queries |
