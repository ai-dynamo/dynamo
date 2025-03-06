# Metrics Visualization with Prometheus and Grafana

This directory contains configuration for visualizing metrics from the metrics aggregation service using Prometheus and Grafana.

## Components

- **Prometheus**: Collects and stores metrics from the service
- **Grafana**: Provides visualization dashboards for the metrics

## Getting Started

1. Make sure Docker and Docker Compose are installed on your system

2. Start `metrics` and the corresponding `examples/rust/service_metrics/bin/server.rs` that populates dummy KV Cache metrics.

3. Start the visualization stack:

  ```bash
  docker compose --profile metrics up -d
  ```

4. Web servers started:
   - Grafana: http://localhost:3000 (default login: admin/admin)
   - Prometheus: http://localhost:9090

## Configuration

### Prometheus

The Prometheus configuration is defined in `dashboard/prometheus.yml`. It is configured to scrape metrics from the metrics aggregation service endpoint.

Note: You may need to adjust the target based on your host configuration and network setup.

### Grafana

Grafana is pre-configured with:
- Prometheus datasource
- Sample dashboard for visualizing service metrics

## Required Files

The following configuration files should be present in this directory:
- `..\docker-compose.yml`: Defines the Prometheus and Grafana services
- `prometheus.yml`: Contains Prometheus scraping configuration
- `grafana.json`: Contains Grafana dashboard configuration
- `grafana-datasources.yml`: Contains Grafana datasource configuration
- `grafana-dashboard-providers.yml`: Contains Grafana dashboard provider configuration

## Metrics

The prometheus service exposes the following metrics:
- `llm_load_avg`: Average load across workers
- `llm_load_std`: Load standard deviation across workers
- `llm_requests_active_slots`: Number of currently active request slots
- `llm_requests_total_slots`: Total available request slots
- `llm_kv_blocks_active`: Number of active KV blocks
- `llm_kv_blocks_total`: Total KV blocks available

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

