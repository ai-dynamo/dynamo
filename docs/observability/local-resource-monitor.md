---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Dynamo Local Resource Monitor
---

# Dynamo Local Resource Monitor

A Dynamo-specific Prometheus exporter that tracks per-process resource usage (VRAM, GPU utilization, PCIe bandwidth, CPU, disk I/O, network I/O) for Dynamo inference processes — labeled by model name, process identity (vLLM, SGLang, TRT-LLM, dynamo modules), and PID. Designed for profiling Dynamo deployments and understanding resource consumption during inference.

> [!IMPORTANT]
> This tool is designed to run on the **host machine**, not inside a container. It needs direct access to GPU devices and host-level process information.

## Quick Start

```bash
pip install psutil nvidia-ml-py prometheus-client
python3 deploy/observability/dynamo_local_resource_monitor.py --host 0.0.0.0 --port 8051
```

Then verify metrics at `http://<host>:8051/metrics`.

If any dependencies are missing, the script prints the exact `pip install` command needed and exits.

## Architecture

The exporter uses a multiprocess design to bypass the Python GIL:

| Process | Sample Rate | What It Collects |
|---------|-------------|------------------|
| 1 per GPU | ~10/s | PCIe TX/RX throughput via pynvml |
| 1 shared | ~5/s | CPU %, GPU memory/utilization/temperature, network I/O via psutil + pynvml |
| 1 shared | ~1/s | Aggregate disk I/O via psutil |
| Main | — | Polls pipes from above, updates Prometheus gauges |

## Prometheus Metrics

All metrics use the `dynamo_` prefix.

### GPU Metrics (labeled by `gpu`, and `pid`/`process_name` where applicable)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_gpu_memory_used_gib` | Gauge | `gpu`, `pid`, `process_name` | GPU memory used by process (GiB) |
| `dynamo_gpu_memory_total_gib` | Gauge | `gpu` | Total GPU memory (GiB) |
| `dynamo_gpu_utilization_percent` | Gauge | `gpu` | GPU utilization (%) |
| `dynamo_gpu_temperature_celsius` | Gauge | `gpu` | GPU temperature (Celsius) |
| `dynamo_gpu_pcie_tx_gbps` | Gauge | `gpu` | PCIe TX throughput (GB/s) |
| `dynamo_gpu_pcie_rx_gbps` | Gauge | `gpu` | PCIe RX throughput (GB/s) |

### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_cpu_utilization_percent` | Gauge | — | Overall CPU utilization (%) |
| `dynamo_cpu_process_percent` | Gauge | `process_name` | CPU usage by process name group (%) |
| `dynamo_network_sent_mbps` | Gauge | — | Network bytes sent (MB/s) |
| `dynamo_network_recv_mbps` | Gauge | — | Network bytes received (MB/s) |
| `dynamo_disk_read_mbps` | Gauge | — | Disk read throughput (MB/s) |
| `dynamo_disk_write_mbps` | Gauge | — | Disk write throughput (MB/s) |

### Process Identification

The exporter automatically identifies Dynamo inference processes and labels them with human-readable names:

- **vLLM**: `VLLM::EngineCore, <model>, PID=<pid>`
- **SGLang**: `<process_name>, <model>, PID=<pid>`
- **TensorRT-LLM**: `TRT-LLM/MPI, <model>, PID=<pid>` (via mpi4py/orted ancestry)
- **Python modules**: `dynamo.frontend, <model>, PID=<pid>`

## Prometheus Integration

### Why a Dedicated Prometheus Instance?

The GPU exporter is scraped at **100ms intervals** (10 samples/sec) to capture short-lived GPU memory spikes and PCIe bursts. Mixing these into the main Prometheus instance (which scrapes at 1-10s intervals) would:

1. **Inflate TSDB storage** for all metrics — retention is global, not per-job
2. **Increase resource pressure** on main Prometheus, risking query latency for Grafana dashboards and alerting
3. **Make short retention impossible** without also truncating long-lived NATS/etcd/DCGM/backend history

The solution is a second Prometheus instance with `--storage.tsdb.retention.time=15m`, keeping high-frequency samples without impacting the primary observability stack. Note: Prometheus TSDB stores data in 2-hour minimum blocks, so actual data may persist longer than 15 minutes until a block boundary is reached.

### Setup

The `docker-observability.yml` compose file includes a `dynamo-resource-monitor` service pre-configured for this. Start the full stack:

```bash
# Start infrastructure (NATS, etcd)
docker compose -f deploy/docker-compose.yml up -d

# Start observability stack (includes both Prometheus instances)
docker compose -f deploy/docker-observability.yml up -d

# Start the resource monitor on the host
python3 deploy/observability/dynamo_local_resource_monitor.py --host 0.0.0.0 --port 8051
```

The dedicated instance is available at `http://localhost:9091`. Grafana is pre-configured with a `dynamo-resource-monitor` datasource pointing to it.

### Firewall Configuration

The `dynamo-resource-monitor` container scrapes the host-side exporter via `host.docker.internal:8051`. If your host has a firewall (e.g., UFW), you must allow inbound TCP on port 8051 from the Docker Compose bridge network — otherwise Prometheus will report `context deadline exceeded` and Grafana will show "No data".

```bash
# Find the bridge interface for the compose network
BRIDGE_IF=$(docker network inspect deploy_server --format '{{.Id}}' | cut -c1-12)
BRIDGE_IF="br-${BRIDGE_IF}"

# Allow the exporter port from that bridge
sudo ufw allow in on "$BRIDGE_IF" to any port 8051 proto tcp comment "dynamo_local_resource_monitor from docker compose network"
```

To verify the scrape target is healthy:

```bash
curl -s http://localhost:9091/api/v1/targets | python3 -m json.tool | grep -A2 '"health"'
```

## CLI Options

```text
usage: dynamo_local_resource_monitor.py [-h] [--port PORT] [--host HOST]
                                  [--main-interval MAIN_INTERVAL]
                                  [--disk-interval DISK_INTERVAL]

options:
  --port PORT                    Prometheus metrics port (default: 8051)
  --host HOST                    Bind address (default: 0.0.0.0)
  --main-interval MAIN_INTERVAL  Main collection interval ms (default: 200)
  --disk-interval DISK_INTERVAL  Disk I/O collection interval ms (default: 1000)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `psutil` | CPU, disk, network metrics |
| `nvidia-ml-py` (`pynvml`) | GPU metrics via NVML |
| `prometheus-client` | Prometheus exposition format HTTP server |

## Relationship to Other Observability Tools

This exporter is designed for **high-frequency, per-process resource profiling** on a single node — complementing the Prometheus + Grafana stack described in [Prometheus + Grafana Setup](prometheus-grafana.md) and the DCGM exporter which provides per-GPU (not per-process) metrics.

| | Dynamo Local Resource Monitor | DCGM Exporter |
|---|---|---|
| Per-process GPU memory | Yes | No (per-GPU totals only) |
| PCIe bandwidth | Yes (10/s, dedicated subprocesses) | Yes |
| Sample rate | 100ms (PCIe), 200ms (GPU/CPU), 1s (disk) | 5s |
| CPU/disk/network | Yes | No |
| Process identification | Yes (vLLM, SGLang, TRT-LLM aware) | No |
