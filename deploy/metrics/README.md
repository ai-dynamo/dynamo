# Metrics Visualization with Prometheus and Grafana

This directory contains configuration for visualizing metrics from the metrics aggregation service using Prometheus and Grafana.

## Components

- **Prometheus**: Collects and stores metrics from the service
- **Grafana**: Provides visualization dashboards for the metrics

## Topology

Default Service Relationship Diagram:
```text
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │ nats-server │    │ etcd-server │    │dcgm-exporter│
     │   :4222     │    │   :2379     │    │   :9400     │
     │   :6222     │    │   :2380     │    │             │
     │   :8222     │    │             │    │             │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            │                  │                  │
            │ :8222/varz       │ :2379/metrics    │ :9400/metrics
            │                  │                  │
            ▼                  │                  │
     ┌─────────────┐           │                  │
     │nats-prom-exp│           │                  │
     │   :7777     │           │                  │
     │             │           │                  │
     │  /metrics   │           │                  │
     └──────┬──────┘           │                  │
            │                  │                  │
            │ :7777/metrics    │                  │
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────────────────────────────────────────┐
     │                prometheus                       │
     │                  :9090                          │
     │                                                 │
     │  scrapes: nats-prom-exp:7777/metrics            │
     │           etcd-server:2379/metrics              │
     │           dcgm-exporter:9400/metrics            │
     └──────────────────┬──────────────────────────────┘
                        │
                        │ :9090/query API
                        │
                        ▼
                ┌─────────────┐
                │   grafana   │
                │    :3001    │
                │             │
                └─────────────┘
```

Networks:
- monitoring: nats-prom-exp, etcd-server, dcgm-exporter, prometheus, grafana
- default: nats-server (accessible via host network)

## Getting Started

1. Make sure Docker and Docker Compose are installed on your system

2. Start Dynamo dependencies. Assume you're at the root dynamo path:

   ```bash
   docker compose -f deploy/metrics/docker-compose.yml up -d  # Minimum components for Dynamo: etcd/nats/dcgm-exporter
   # or
   docker compose -f deploy/metrics/docker-compose.yml --profile metrics up -d  # In addition to the above, start Prometheus & Grafana
   ```

   If you have particular GPU(s) to use, set the variable below before docker compose:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,2
   ```

3. Web servers started. The ones that end in /metrics are in Prometheus format:
   - Grafana: `http://localhost:3001` (default login: dynamo/dynamo)
   - Prometheus Server: `http://localhost:9090`
   - NATS Server: `http://localhost:8222` (monitoring endpoints: /varz, /healthz, etc.)
   - NATS Prometheus Exporter: `http://localhost:7777/metrics`
   - etcd Server: `http://localhost:2379/metrics`
   - DCGM Exporter: `http://localhost:9401/metrics`

4. Optionally, if you want to experiment further, look through components/metrics/README.md for more details on launching a metrics server (subscribes to nats), mock_worker (publishes to nats), and real workers.

   - Start the [components/metrics](../../components/metrics/README.md) application to begin monitoring for metric events from dynamo workers and aggregating them on a Prometheus metrics endpoint: `http://localhost:9091/metrics`.
   - Uncomment the appropriate lines in prometheus.yml to poll port 9091.
   - Start worker(s) that publishes KV Cache metrics: [examples/rust/service_metrics/bin/server](../../lib/runtime/examples/service_metrics/README.md)` can populate dummy KV Cache metrics.
   - For a real workflow with real data, see the KV Routing example in [examples/llm/utils/vllm.py](../../examples/llm/utils/vllm.py).


## Configuration

### Prometheus

The Prometheus configuration is defined in [prometheus.yml](./prometheus.yml). It is configured to scrape metrics from the metrics aggregation service endpoint.

Note: You may need to adjust the target based on your host configuration and network setup.

### Grafana

Grafana is pre-configured with:
- Prometheus datasource
- Sample dashboard for visualizing service metrics
![grafana image](./grafana1.png)

## Required Files

The following configuration files should be present in this directory:
- [docker-compose.yml](./docker-compose.yml): Defines the Prometheus and Grafana services
- [prometheus.yml](./prometheus.yml): Contains Prometheus scraping configuration
- [grafana.json](./grafana.json): Contains Grafana dashboard configuration
- [grafana-datasources.yml](./grafana-datasources.yml): Contains Grafana datasource configuration
- [grafana-dashboard-providers.yml](./grafana-dashboard-providers.yml): Contains Grafana dashboard provider configuration

## Running the example `metrics` component

When you run the example [components/metrics](../../components/metrics/README.md) component, it exposes a Prometheus /metrics endpoint with the followings (defined in [../../components/metrics/src/lib.rs](../../components/metrics/src/lib.rs)):
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
