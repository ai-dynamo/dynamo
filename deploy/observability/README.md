# Dynamo Observability

For detailed documentation on Observability (Prometheus metrics, tracing, and logging), please refer to [docs/pages/observability/](../../docs/pages/observability/).

## Quick Start

### Prerequisites

The observability stack requires the main `deploy/docker-compose.yml` to be running first (for the `deploy_server` Docker network with NATS and etcd).

```bash
# Start core services first
docker compose -f deploy/docker-compose.yml up -d

# Start the full observability stack
docker compose -f deploy/docker-observability.yml up -d
```

### Services

| Service | Port | Description |
|---|---|---|
| Grafana | 3000 | Dashboards (login: dynamo/dynamo) |
| Prometheus | 9090 | Metrics collection |
| Tempo | 3200 | Distributed tracing backend |
| Pyroscope | 4040 | Continuous profiling backend |
| OTEL Collector | 4317 | OTLP gRPC receiver |
| OTEL Collector | 4318 | OTLP HTTP receiver |
| OTEL Collector | 8889 | Prometheus metrics (span metrics) |
| OTEL Collector | 13133 | Health check |
| DCGM Exporter | 9401 | GPU metrics |
| NATS Exporter | 7777 | NATS metrics |

## Enabling OTEL Trace Export from Dynamo

Set these environment variables on the Dynamo process:

```bash
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=dynamo
export DYN_LOGGING_JSONL=1
```

For the Python KVBM connector, the same `OTEL_EXPORT_ENABLED=1` flag enables
trace export. Requires `opentelemetry-api` and `opentelemetry-sdk` Python packages.

## Continuous Profiling with Pyroscope

Pyroscope provides always-on continuous profiling with minimal overhead (1–3% CPU).
Applications push profiles directly to Pyroscope via language-specific SDKs —
profiles are **not** routed through the OTEL Collector (the OTel profiling signal
is still experimental).

### Python (vLLM / KVBM connector)

Install the SDK:

```bash
pip install pyroscope-io
```

Add to your application entry point (e.g. top of `dynamo_connector.py` or a
`sitecustomize.py` that runs at import time):

```python
import pyroscope

pyroscope.configure(
    application_name="dynamo.vllm",          # appears in Pyroscope/Grafana
    server_address="http://host.docker.internal:4040",
    sample_rate=100,           # Hz (default)
    detect_subprocesses=True,  # profile vLLM worker sub-processes
    oncpu=True,                # CPU time only (excludes I/O wait)
    gil_only=True,             # only GIL-holding threads
    tags={
        "service": "dynamo-kvbm",
        "component": "vllm",
    },
)
```

Environment variables for the `dynamo-kvbm` container:

```bash
PYROSCOPE_SERVER_ADDRESS=http://host.docker.internal:4040
PYROSCOPE_APPLICATION_NAME=dynamo.vllm
```

You can toggle profiling on/off by wrapping the `pyroscope.configure()` call
behind an environment variable check:

```python
import os
if os.environ.get("PYROSCOPE_ENABLED", "0") == "1":
    import pyroscope
    pyroscope.configure(
        application_name=os.environ.get("PYROSCOPE_APPLICATION_NAME", "dynamo.vllm"),
        server_address=os.environ.get("PYROSCOPE_SERVER_ADDRESS", "http://host.docker.internal:4040"),
        detect_subprocesses=True,
        tags={"service": "dynamo-kvbm"},
    )
```

### Rust (KVBM block manager)

Add the `pyroscope` and `pyroscope_pprofrs` crates to your `Cargo.toml`:

```toml
[dependencies]
pyroscope = "0.5"
pyroscope_pprofrs = "0.2"
```

Initialize the agent early in `main()`:

```rust
use pyroscope::PyroscopeAgent;
use pyroscope_pprofrs::{pprof_backend, PprofConfig};

let agent = PyroscopeAgent::builder(
        "http://host.docker.internal:4040",
        "dynamo.kvbm.rust",
    )
    .backend(pprof_backend(PprofConfig::new().sample_rate(100)))
    .tags([("service", "dynamo-kvbm"), ("component", "kvbm")].to_vec())
    .build()
    .expect("failed to build pyroscope agent");

let agent_running = agent.start().expect("failed to start pyroscope agent");
// ... application code ...
// agent_running.stop() on shutdown
```

Environment variables for the Rust process:

```bash
PYROSCOPE_SERVER_ADDRESS=http://host.docker.internal:4040
PYROSCOPE_APPLICATION_NAME=dynamo.kvbm.rust
```

### Viewing Profiles in Grafana

1. Open Grafana at http://localhost:3000 (login: dynamo/dynamo)
2. Go to **Explore** and select the **Pyroscope** datasource
3. Choose an application name (e.g. `dynamo.vllm`) and profile type
4. Browse the flame graph to identify hot code paths

The Pyroscope standalone UI is also available at http://localhost:4040.

## Architecture

Dynamo services emit OTLP traces to the OTEL Collector on port 4317. The
collector runs a `spanmetrics` processor that generates Prometheus-compatible
latency histograms and span counts, then forwards the raw traces to Tempo.
Prometheus scrapes the collector's metrics endpoint on port 8889. Application
code pushes continuous profiling data directly to Pyroscope on port 4040 via
the language-specific SDKs (pyroscope-io for Python, pyroscope-rs for Rust).
Grafana reads from Prometheus (metrics), Tempo (traces), and Pyroscope
(profiles) with cross-signal correlation enabled.

## Trace Correlation

All KVBM operations are correlated by W3C trace ID across:

- **Rust side**: `#[tracing::instrument]` spans with `otel.name` fields like
  `kvbm.remote_transfer`, `kvbm.offload`, `kvbm.onboard_local`
- **Python side**: OpenTelemetry spans in `dynamo_connector.py` like
  `kvbm.get_matched_tokens`, `kvbm.build_connector_meta`
- **W3C Trace Context**: Propagated via `traceparent` headers across NATS,
  HTTP, and TCP transports

### Finding Traces in Grafana

1. Open Grafana at http://localhost:3000 (login: dynamo/dynamo)
2. Go to Explore, select the Tempo datasource
3. Use TraceQL to search:
   - All dynamo traces: `{ resource.service.name="dynamo" }`
   - By request ID: `{ span.kvbm.request_id="<id>" }`
   - Remote transfers only: `{ name=~"kvbm.remote_transfer.*" }`
   - Slow transfers: `{ name=~"kvbm.*" && duration > 100ms }`
4. The KVBM Tracing dashboard has pre-built panels for span rates and latency.

## Grafana Datasources

| Datasource | Type | URL |
|---|---|---|
| Prometheus | prometheus | http://prometheus:9090 |
| Tempo | tempo | http://tempo:3200 |
| Pyroscope | grafana-pyroscope-datasource | http://pyroscope:4040 |

## Grafana Dashboards

| Dashboard | Description |
|---|---|
| Dynamo | Overall system metrics |
| KVBM | Block manager metrics (cache hits, offload/onboard) |
| KVBM Tracing | Trace explorer and span rate/latency metrics |
| Disagg | Disaggregated serving metrics |
| DCGM Metrics | GPU utilization and memory |
