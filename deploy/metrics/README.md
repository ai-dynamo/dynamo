# Metrics Visualization with Prometheus and Grafana

This directory contains configuration for visualizing metrics from the metrics aggregation service using Prometheus and Grafana.

## Components

- **Prometheus**: Collects and stores metrics from the service
- **Grafana**: Provides visualization dashboards for the metrics

## Getting Started

1. Make sure Docker and Docker Compose are installed on your system

2. Start the `components/metrics` application to begin monitoring for metric events from dynamo workers
   and aggregating them on a prometheus metrics endpoint: `http://localhost:9091/metrics`.

3. Start worker(s) that publishes KV Cache metrics.
  - For quick testing, `examples/rust/service_metrics/bin/server.rs` can populate dummy KV Cache metrics.
  - For a real workflow with real data, see the KV Routing example in `examples/python_rs/llm/vllm`.

4. Start the visualization stack:

  ```bash
  docker compose --profile metrics up -d
  ```

5. Web servers started:
   - Grafana: `http://localhost:3001` (default login: admin/admin) (started by docker compose)
   - Prometheus Server: `http://localhost:9090` (started by docker compose)
   - Prometheus Metrics Endpoint: `http://localhost:9091/metrics` (started by `components/metrics` application)

## Configuration

### Prometheus

The Prometheus configuration is defined in `prometheus.yml`. It is configured to scrape metrics from the metrics aggregation service endpoint.

Note: You may need to adjust the target based on your host configuration and network setup.

### Grafana

Grafana is pre-configured with:
- Prometheus datasource
- Sample dashboard for visualizing service metrics

## Required Files

The following configuration files should be present in this directory:
- `docker-compose.yml`: Defines the Prometheus and Grafana services
- `prometheus.yml`: Contains Prometheus scraping configuration
- `grafana.json`: Contains Grafana dashboard configuration
- `grafana-datasources.yml`: Contains Grafana datasource configuration
- `grafana-dashboard-providers.yml`: Contains Grafana dashboard provider configuration

## Metrics

The prometheus metrics endpoint exposes the following metrics:
- `llm_requests_active_slots`: Number of currently active request slots per worker
- `llm_requests_total_slots`: Total available request slots per worker
- `llm_kv_blocks_active`: Number of active KV blocks per worker
- `llm_kv_blocks_total`: Total KV blocks available per worker
- `llm_kv_hit_rate_percent`: Cumulative KV Cache hit percent per worker
- `llm_load_avg`: Average load across workers
- `llm_load_std`: Load standard deviation across workers

## Troubleshooting

1. Verify services are running:
  ```bash
  docker compose ps
  ```

2. Check logs:
  ```bash
  docker compose logs prometheus
  docker compose logs grafana
  ```

