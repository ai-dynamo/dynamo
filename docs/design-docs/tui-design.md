<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo TUI — Design Document

**Issue:** [#1453](https://github.com/ai-dynamo/dynamo/issues/1453), [#1455](https://github.com/ai-dynamo/dynamo/issues/1455)
**Status:** Draft
**Author:** Kaustubh Karthik

## 1. Motivation

Dynamo uses ETCD for service discovery and NATS for inter-component messaging. Today, operators must inspect raw `etcdctl` output or NATS CLI tooling to understand deployment state. This is cumbersome and error-prone for debugging and monitoring.

We propose a **K9s-style Terminal UI** (`dynamo-tui`) that provides real-time visibility into a running Dynamo deployment directly from the terminal.

## 2. Goals

| Priority | Goal |
|----------|------|
| P0 | Watch ETCD and display discovered namespaces, components, and endpoints in a hierarchical view |
| P0 | Show health status (Ready, Provisioning, Offline) per component/endpoint |
| P0 | Vim-style keyboard navigation (hjkl, arrows, tab) |
| P1 | Monitor NATS connection status and message statistics (bytes in/out, msg count) |
| P1 | Display NATS JetStream stream and consumer information |
| P2 | Scrape Prometheus metrics endpoint (TTFT, TPOT, throughput, queue depth) |
| P2 | Model card display (show registered models per component) |

## 3. Non-Goals

- Mutating state (no restart/scale/deploy actions in MVP)
- Kubernetes-native discovery (ETCD-only for MVP; K8s can be added later)
- Log tailing or request tracing
- Multi-cluster support

## 4. Technology Choice

**Rust + Ratatui** — selected for the following reasons:

1. **Native integration**: Dynamo's core is Rust. The TUI can directly depend on `dynamo-runtime` for ETCD/NATS client types, environment variable conventions, and discovery data models.
2. **Issue direction**: The issue author confirmed "Currently being prototyped in Ratatui."
3. **Performance**: Zero-overhead terminal rendering for high-frequency updates.
4. **Single binary**: `cargo install` or `cargo run -p dynamo-tui` — no Python environment needed.

## 5. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      dynamo-tui                         │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐            │
│  │  ETCD    │  │  NATS    │  │ Metrics   │  Sources   │
│  │  Source  │  │  Source  │  │ Source    │  (async)   │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘            │
│       │              │              │                   │
│       ▼              ▼              ▼                   │
│  ┌─────────────────────────────────────────┐           │
│  │              App State                   │           │
│  │  - namespaces: Vec<Namespace>            │           │
│  │  - nats_stats: NatsStats                 │           │
│  │  - metrics: Option<PrometheusMetrics>    │           │
│  │  - focus: PaneFocus                      │           │
│  │  - selected: (ns_idx, comp_idx, ep_idx)  │           │
│  └──────────────────┬──────────────────────┘           │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────┐           │
│  │              UI Renderer                 │           │
│  │  ┌────────┬──────────┬──────────┐       │           │
│  │  │Ns List │Comp List │EP List   │       │           │
│  │  │        │          │          │       │           │
│  │  │ dynamo │ backend  │ generate │       │           │
│  │  │ > test │ frontend │ health   │       │           │
│  │  │        │          │          │       │           │
│  │  ├────────┴──────────┴──────────┤       │           │
│  │  │  NATS: Connected │ Metrics   │       │           │
│  │  │  ▲12.3K ▼8.1K    │ TTFT: 45 │       │           │
│  │  └──────────────────────────────┘       │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Data Sources

Each source runs as a Tokio task and sends updates through `tokio::sync::mpsc` channels:

| Source | Connection | Update Mechanism | Interval |
|--------|-----------|-----------------|----------|
| ETCD | `etcd-client` | Watch API (push-based) | Real-time |
| NATS | `async-nats` | Poll `server_info()` + JetStream API | 2s default |
| Metrics | `reqwest` | HTTP GET + parse Prometheus text format | 3s default |

### 5.2 ETCD Key Hierarchy

The TUI watches three ETCD prefixes:

```
v1/instances/{namespace}/{component}/{endpoint}/{instance_id}    → Endpoints
v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}          → Model Cards
v1/event_channels/{namespace}/{component}/{topic}/{instance_id}  → Event Channels
```

Watch events (`Put` → Added, `Delete` → Removed) are translated into the hierarchical tree model.

### 5.3 App State Model

```rust
struct Namespace {
    name: String,
    components: Vec<Component>,
}

struct Component {
    name: String,
    status: HealthStatus,       // Ready | Provisioning | Offline
    endpoints: Vec<Endpoint>,
    instance_count: usize,
    models: Vec<ModelInfo>,     // From v1/mdc/ keys
}

struct Endpoint {
    name: String,
    status: HealthStatus,
    instance_count: usize,
    last_seen: Instant,
}

enum HealthStatus {
    Ready,
    Provisioning,
    Offline,
}
```

### 5.4 UI Layout

```
╔═══════════════╦═══════════════════╦═══════════════════════╗
║  Namespaces   ║   Components      ║    Endpoints          ║
║               ║                   ║                       ║
║ > dynamo      ║ > backend [Ready] ║   generate  [Ready]   ║
║   test-ns     ║   frontend        ║   clear_kv  [Ready]   ║
║               ║   router          ║   health    [Ready]   ║
║               ║                   ║                       ║
╠═══════════════╩═══════════════════╩═══════════════════════╣
║  NATS: Connected │ Msgs ▲1.2K ▼890 │ Bytes ▲45MB ▼12MB   ║
╠═══════════════════════════════════════════════════════════╣
║  Streams: audit-events (3 consumers) │ kv-events (1)      ║
╠═══════════════════════════════════════════════════════════╣
║  Metrics: TTFT p50=42ms p99=180ms │ TPOT p50=8ms          ║
╚═══════════════════════════════════════════════════════════╝
  [q]uit  [r]efresh  [←→] focus  [↑↓/jk] navigate
```

## 6. Configuration

The TUI reuses Dynamo's standard environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ETCD_ENDPOINTS` | `http://localhost:2379` | ETCD connection |
| `NATS_SERVER` | `nats://localhost:4222` | NATS connection |
| `ETCD_AUTH_*` | — | ETCD TLS/auth |
| `NATS_AUTH_*` | — | NATS auth |

CLI arguments (via `clap`):

```
dynamo-tui [OPTIONS]

Options:
  --etcd-endpoints <URL>       Override ETCD_ENDPOINTS
  --nats-server <URL>          Override NATS_SERVER
  --metrics-url <URL>          Prometheus metrics endpoint to scrape
  --nats-interval <DURATION>   NATS poll interval [default: 2s]
  --metrics-interval <DURATION> Metrics scrape interval [default: 3s]
  --watch-prefix <PREFIX>      ETCD prefix to watch [default: v1/]
```

## 7. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑`/`k` | Move selection up |
| `↓`/`j` | Move selection down |
| `←`/`h` | Focus previous pane |
| `→`/`l` | Focus next pane |
| `Tab` | Cycle focus forward |
| `Enter` | Expand/collapse detail view |
| `r` | Force refresh all sources |
| `q`/`Esc`/`Ctrl+C` | Quit |

## 8. Error Handling

- **ETCD unreachable**: Show "Disconnected" banner, retry with exponential backoff (1s, 2s, 4s, max 30s)
- **NATS unreachable**: Show "NATS: Disconnected" in status bar, continue ETCD monitoring
- **Metrics endpoint down**: Show "Metrics: Unavailable", skip until next interval
- **Terminal resize**: Ratatui handles resize events natively; layout is responsive

## 9. Future Extensions

1. **Kubernetes discovery backend** — swap ETCD source for kube-rs reflectors
2. **Log viewer pane** — tail NATS JetStream audit events
3. **Request tracing** — follow a request through the pipeline
4. **Actions** — restart components, scale instances (requires operator integration)
5. **Theme support** — configurable color schemes

## 10. Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ratatui` | latest | Terminal UI framework |
| `crossterm` | latest | Terminal backend |
| `etcd-client` | 0.17.0 | ETCD client (workspace) |
| `async-nats` | 0.45.0 | NATS client (workspace) |
| `tokio` | 1.48.0 | Async runtime (workspace) |
| `clap` | 4.5+ | CLI argument parsing |
| `reqwest` | 0.12 | HTTP client for metrics |
| `serde` / `serde_json` | 1 | Serialization (workspace) |
| `anyhow` | 1 | Error handling (workspace) |
| `chrono` | 0.4 | Timestamps (workspace) |
| `humantime` | 2.2 | Duration parsing (workspace) |
