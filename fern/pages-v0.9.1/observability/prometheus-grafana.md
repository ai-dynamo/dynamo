---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Metrics Visualization with Prometheus and Grafana

## Overview

This guide shows how to set up Prometheus and Grafana for visualizing Dynamo metrics on a single machine for demo purposes.

![Grafana Dynamo Dashboard](/assets/img/grafana-dynamo-composite.png)

**Components:**
- **Prometheus Server** - Collects and stores metrics from Dynamo services
- **Grafana** - Provides dashboards by querying the Prometheus Server

**For metrics reference**, see [Metrics Documentation](metrics.md).

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_SYSTEM_PORT` | System metrics/health port | `-1` (disabled) | `8081` |

## Getting Started Quickly

This is a single machine example.

### Start the Observability Stack

Start the observability stack (Prometheus, Grafana, Tempo, exporters). See [Observability Getting Started](README.md#getting-started-quickly) for instructions and prerequisites.

### Start Dynamo Components

Start frontend and worker (a simple single GPU example):

```bash
# Start frontend (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
python -m dynamo.frontend &

# Start vLLM worker with metrics enabled on port 8081
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager
```

After the workers are running, send a few test requests to populate metrics in the system:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 100
  }'
```

After sending a few requests, the Prometheus Exposition Format text metrics are available at:
- Frontend: `http://localhost:8000/metrics`
- Backend worker: `http://localhost:8081/metrics`

### Access Web Interfaces

Once Dynamo components are running:

1. Open **Grafana** at `http://localhost:3000` (username: `dynamo`, password: `dynamo`)
2. Click on **Dashboards** in the left sidebar
3. Select **Dynamo Dashboard** to view metrics and traces

Other interfaces:
- **Prometheus**: `http://localhost:9090`
- **Tempo** (tracing): Accessible through Grafana's Explore view. See [Tracing Guide](tracing.md) for details.

**Note:** If accessing from another machine, replace `localhost` with the machine's hostname or IP address, and ensure firewall rules allow access to these ports (3000, 9090).

---

## Configuration

### Prometheus

The Prometheus configuration is specified in [prometheus.yml](https://github.com/ai-dynamo/dynamo/tree/main/deploy/observability/prometheus.yml). This file is set up to collect metrics from the metrics aggregation service endpoint.

Please be aware that you might need to modify the target settings to align with your specific host configuration and network environment.

After making changes to prometheus.yml, restart the Prometheus service. See [Observability Getting Started](README.md#getting-started-quickly) for Docker Compose commands.

### Grafana

Grafana is pre-configured with:
- Prometheus datasource
- Sample dashboard for visualizing service metrics

### Troubleshooting

1. Verify services are running using `docker compose ps`

2. Check logs using `docker compose logs`

3. Check Prometheus targets at `http://localhost:9090/targets` to verify metric collection.

4. If you encounter issues with stale data or configuration, stop services and wipe volumes using `docker compose down -v` then restart.

  **Note:** The `-v` flag removes named volumes (grafana-data, tempo-data), which will reset dashboards and stored metrics.

For specific Docker Compose commands, see [Observability Getting Started](README.md#getting-started-quickly).

## Developer Guide

For detailed information on creating custom metrics in Dynamo components, see:

- [Metrics Developer Guide](metrics-developer-guide.md)
