<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamo TUI — Development Guide

## Overview

Terminal UI for monitoring Dynamo deployments. Watches ETCD for service discovery, monitors NATS message flows, and optionally scrapes Prometheus metrics.

## Build & Test

```bash
cargo check -p dynamo-tui
cargo test -p dynamo-tui
cargo run -p dynamo-tui
```

## Architecture

- `src/main.rs` — Entry point, CLI parsing (clap), terminal setup, event loop
- `src/app.rs` — App state, event handling, navigation logic
- `src/model.rs` — Data types (Namespace, Component, Endpoint, NatsStats, etc.) and tree building
- `src/ui.rs` — Ratatui rendering (3-column layout, status bars)
- `src/input.rs` — Key → Action mapping
- `src/sources/mod.rs` — `Source` trait and `AppEvent` enum
- `src/sources/etcd.rs` — ETCD watcher (snapshot + watch)
- `src/sources/nats.rs` — NATS stats poller
- `src/sources/metrics.rs` — Prometheus text format scraper

## Key Design Decisions

1. **Trait-based sources**: All data sources implement `Source` trait for testability. Each source has a `mock` module under `#[cfg(test)]`.
2. **Channel-based architecture**: Sources send `AppEvent`s via `tokio::sync::mpsc`. The main loop uses `tokio::select!` to multiplex.
3. **Pure state updates**: `App::handle_event()` is a pure function that takes events and updates state. This makes it fully unit-testable without a terminal.
4. **ETCD key parsing**: `parse_endpoint_key()` and `parse_model_key()` in `model.rs` handle the `v1/instances/...` and `v1/mdc/...` key hierarchies.

## ETCD Key Hierarchy

```
v1/instances/{namespace}/{component}/{endpoint}/{instance_id_hex}
v1/mdc/{namespace}/{component}/{endpoint}/{instance_id_hex}[/{suffix}]
v1/event_channels/{namespace}/{component}/{topic}/{instance_id_hex}
```

## Adding a New Data Source

1. Create `src/sources/new_source.rs`
2. Implement the `Source` trait
3. Add a mock variant under `#[cfg(test)]`
4. Add a new `AppEvent` variant
5. Handle the event in `App::handle_event()`
6. Render it in `ui.rs`

## Common Pitfalls

- `etcd-client` requires `protoc` (protobuf compiler) at build time
- Rust 2024 edition: `std::env::remove_var()` is unsafe — use CLI overrides in tests instead
- Tracing goes to stderr (not stdout) since we own the terminal for the TUI
