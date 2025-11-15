# Dynamo TUI

Terminal dashboard for monitoring a running Dynamo deployment. The TUI discovers namespaces, components, and endpoints via ETCD, surfaces NATS client statistics, and optionally scrapes Prometheus metrics to highlight serving latency and throughput.

## Usage

```bash
cargo run -p dynamo-tui -- \
  --metrics-url http://localhost:9100/metrics \
  --metrics-interval 3s \
  --nats-interval 2s
```

The binary uses the same environment variables as other Dynamo components to locate ETCD and NATS. Override ETCD/NATS endpoints via the standard configuration (see the repository README for details).

### Command-line options

| Flag | Description |
| ---- | ----------- |
| `--metrics-url <url>` | Optional Prometheus `/metrics` endpoint. When provided the TUI computes TTFT, TPOT, request rate, and token throughput. |
| `--metrics-interval <duration>` | Scrape interval for Prometheus metrics (default: `3s`). |
| `--nats-interval <duration>` | Poll interval for NATS connection statistics (default: `2s`). |

Durations accept `humantime` formats such as `500ms`, `2s`, or `1m`.

## Key bindings

| Key | Action |
| --- | ------ |
| `↑` / `k` | Move selection up within the focused column |
| `↓` / `j` | Move selection down within the focused column |
| `←` / `h` | Move focus left (e.g. from endpoints to components) |
| `→` / `l` | Move focus right (e.g. from components to endpoints) |
| `r` | Trigger an immediate refresh of metrics and NATS stats |
| `q`, `Esc`, or `Ctrl+C` | Exit the TUI |

The focus indicator highlights which column (Namespace, Component, Endpoint) responds to up/down navigation.

## Panels

* **Namespaces / Components / Endpoints** – hierarchical view sourced from ETCD discovery. Endpoint tiles display health state (`Ready`, `Provisioning`, `Offline`), instance counts, and last activity timestamps.
* **Endpoint details** – contextual information about the selected endpoint, including NATS subjects for each instance.
* **NATS** – byte and message counters plus connection status gathered from the async NATS client.
* **Metrics** – derived Prometheus metrics (TTFT, TPOT, request rate, output token throughput, inflight/queued request gauges) when a metrics endpoint is configured.
* **Status bar** – latest status or error message together with time since the last update.

## Limitations & Future Work

- The dashboard marks endpoints as `Offline` when they lose all instances but retains them in the hierarchy for visibility. Additional reconciliation logic may be required to age out stale endpoints.
- Metrics scraping currently expects Dynamo frontend metric names (e.g. `dynamo_frontend_*`). Other engines may expose different labels.
- Building the crate pulls in `dynamo-runtime` and its native dependencies (e.g. ZeroMQ). Ensure system packages for C++ compilation are available.
