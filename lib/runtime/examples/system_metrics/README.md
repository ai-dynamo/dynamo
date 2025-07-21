# System Metrics Example

Demonstrates custom metrics and monitoring in Dynamo Runtime using Prometheus.

## Overview

- Automatic hierarchical labeling: Runtime automatically adds `namespace` → `component` → `endpoint` labels
- Uses existing Prometheus implementations
- HTTP metrics endpoint automatically added

## Quick Start

### Build
```bash
cd lib/runtime/examples/system_metrics
cargo build
```

### Run Server
```bash
export DYN_LOG=1 DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8000
cargo run --bin system_server
```

### Run Client
```bash
cargo run --bin system_client
```

Note: Running the client will increment `service_requests_total`.

### View Metrics
```bash
curl http://localhost:8000/metrics
```

Example output:
```
# HELP service_request_duration_seconds Time spent processing requests
# TYPE service_request_duration_seconds histogram
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.005"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.01"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.025"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.05"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.1"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.25"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="1"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="2.5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="10"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="+Inf"} 2
service_request_duration_seconds_sum{component="component",endpoint="endpoint",namespace="system",service="backend"} 0.000022239000000000002
service_request_duration_seconds_count{component="component",endpoint="endpoint",namespace="system",service="backend"} 2
# HELP service_requests_total Total number of requests processed
# TYPE service_requests_total counter
service_requests_total{component="component",endpoint="endpoint",namespace="system",service="backend"} 2
# HELP uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE uptime_seconds gauge
uptime_seconds{namespace="http_server"} 725.997013676
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_LOG` | Enable logging | `0` |
| `DYN_SYSTEM_ENABLED` | Enable system metrics | `false` |
| `DYN_SYSTEM_PORT` | HTTP server port | `8000` |

## Metrics

- `service_requests_total`: Request counter
- `service_request_duration_seconds`: Request duration histogram
- `uptime_seconds`: Server uptime gauge

This provides automatic context and grouping for all metrics without manual configuration.

## Troubleshooting

- **Port in use**: Change `DYN_SYSTEM_PORT`
- **Connection refused**: Ensure server is running first
- **No metrics**: Verify `DYN_SYSTEM_ENABLED=true`