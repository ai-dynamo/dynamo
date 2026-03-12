<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo TUI

A K9s-like Terminal User Interface for monitoring [Dynamo](https://github.com/ai-dynamo/dynamo) deployments in real-time.

## Features

- **ETCD Discovery**: Real-time watching of ETCD to detect namespaces, components, and endpoints
- **Hierarchical View**: Three-column layout showing Namespaces → Components → Endpoints
- **Health Monitoring**: Visual status indicators (Ready, Provisioning, Offline)
- **NATS Monitoring**: Connection status, message statistics, JetStream stream info
- **Prometheus Metrics**: Optional scraping of TTFT, TPOT, throughput, and queue depth
- **Keyboard Navigation**: Vim-style (hjkl) and arrow-key navigation

## Usage

### Build & Run

```bash
# Basic usage (ETCD + NATS monitoring)
cargo run -p dynamo-tui

# With Prometheus metrics endpoint
cargo run -p dynamo-tui -- \
  --metrics-url http://localhost:9100/metrics \
  --metrics-interval 3s \
  --nats-interval 2s

# With custom ETCD/NATS endpoints
cargo run -p dynamo-tui -- \
  --etcd-endpoints http://etcd1:2379,http://etcd2:2379 \
  --nats-server nats://nats-server:4222
```

### Environment Variables

The TUI reuses Dynamo's standard environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ETCD_ENDPOINTS` | `http://localhost:2379` | ETCD connection endpoints |
| `NATS_SERVER` | `nats://localhost:4222` | NATS server URL |
| `ETCD_AUTH_USERNAME` | — | ETCD authentication |
| `ETCD_AUTH_PASSWORD` | — | ETCD authentication |

CLI arguments take precedence over environment variables.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑`/`k` | Move selection up |
| `↓`/`j` | Move selection down |
| `←`/`h` | Focus previous pane |
| `→`/`l` | Focus next pane |
| `Tab` | Cycle focus forward |
| `r` | Clear errors / refresh |
| `q`/`Esc`/`Ctrl+C` | Quit |

## Architecture

```
┌──────────────────────────────────────────────┐
│  Sources (async Tokio tasks)                 │
│  ┌──────┐  ┌──────┐  ┌─────────┐           │
│  │ ETCD │  │ NATS │  │ Metrics │           │
│  └──┬───┘  └──┬───┘  └────┬────┘           │
│     └─────────┴────────────┘                 │
│                │ mpsc channel                │
│                ▼                              │
│  ┌─────────────────────┐                     │
│  │     App (state)     │ ← Input events      │
│  └──────────┬──────────┘                     │
│             │                                │
│             ▼                                │
│  ┌─────────────────────┐                     │
│  │   UI (Ratatui)      │                     │
│  └─────────────────────┘                     │
└──────────────────────────────────────────────┘
```

Each data source implements the `Source` trait and can be mocked for testing:

```rust
#[async_trait]
pub trait Source: Send + 'static {
    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<AppEvent>,
        cancel: CancellationToken,
    );
}
```

## Testing

```bash
# Run all unit tests
cargo test -p dynamo-tui

# Run with output
cargo test -p dynamo-tui -- --nocapture
```

Tests cover:
- Data model (tree building, key parsing)
- App state (navigation, focus cycling, event handling)
- Input mapping (vim keys, arrows, quit keys)
- Prometheus text parsing
- Config resolution (CLI vs env vars)
- UI helpers (formatting, colors)

## Prerequisites

- **Rust 1.90+** (as specified in `rust-toolchain.toml`)
- **protobuf compiler** (`protoc`) — required by `etcd-client`
  - macOS: `brew install protobuf`
  - Ubuntu: `apt install protobuf-compiler`

## Related

- [Design Document](../../docs/design-docs/tui-design.md)
- [Implementation Plan](../../docs/design-docs/tui-implementation-plan.md)
- [Issue #1453](https://github.com/ai-dynamo/dynamo/issues/1453) — Feature request
- [Issue #1455](https://github.com/ai-dynamo/dynamo/issues/1455) — NATS monitoring
