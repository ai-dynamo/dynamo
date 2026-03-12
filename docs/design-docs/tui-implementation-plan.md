<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo TUI — Implementation Plan

**Design Doc:** [tui-design.md](tui-design.md)
**Issue:** [#1453](https://github.com/ai-dynamo/dynamo/issues/1453)

## Phase 1: Scaffold & Core (Day 1-2)

### 1.1 Create crate `lib/tui/`

- Add `lib/tui` to workspace members in root `Cargo.toml`
- Create `Cargo.toml` with dependencies (ratatui, crossterm, clap, etcd-client, async-nats, tokio, etc.)
- Directory structure:

```
lib/tui/
├── Cargo.toml
├── CLAUDE.md           # Development guidance
├── src/
│   ├── main.rs         # Entry point, CLI parsing, app loop
│   ├── app.rs          # App state, event loop, update logic
│   ├── ui.rs           # Ratatui rendering
│   ├── input.rs        # Keyboard input handling
│   ├── sources/
│   │   ├── mod.rs      # Source trait, AppEvent enum
│   │   ├── etcd.rs     # ETCD discovery watcher
│   │   ├── nats.rs     # NATS stats poller
│   │   └── metrics.rs  # Prometheus scraper
│   └── model.rs        # Data model (Namespace, Component, Endpoint, etc.)
```

### 1.2 CLI entry point (`main.rs`)

- Parse arguments with `clap` (derive API)
- Initialize crossterm terminal
- Create `App` and run event loop
- Restore terminal on exit (panic hook + normal exit)

### 1.3 Data model (`model.rs`)

- Define `Namespace`, `Component`, `Endpoint`, `HealthStatus`
- Define `NatsStats` (connected, msgs_in, msgs_out, bytes_in, bytes_out, streams)
- Define `PrometheusMetrics` (ttft, tpot, throughput, inflight, queued)

## Phase 2: ETCD Discovery Source (Day 2-3)

### 2.1 ETCD connection (`sources/etcd.rs`)

- Connect using `etcd-client` with env-var config (reuse `ETCD_ENDPOINTS`, auth vars)
- Initial snapshot: `get_all` with prefix `v1/instances/` + `v1/mdc/`
- Parse keys to extract namespace/component/endpoint/instance_id hierarchy
- Parse values as JSON to get `Instance` data

### 2.2 ETCD watch

- Watch prefix `v1/` for real-time updates
- On `Put` event: add/update in tree model
- On `Delete` event: remove from tree model
- Send `AppEvent::DiscoveryUpdate(Vec<Namespace>)` via channel

### 2.3 Tree building

- Aggregate flat key-value entries into hierarchical `Namespace → Component → Endpoint` tree
- Derive health status:
  - `Ready`: at least one instance registered
  - `Provisioning`: key exists but value indicates initializing
  - `Offline`: no instances for a previously-seen component

## Phase 3: TUI Rendering (Day 3-4)

### 3.1 Layout (`ui.rs`)

- Top section: 3-column layout (Namespaces | Components | Endpoints) using `ratatui::layout::Layout`
- Bottom section: Status bars for NATS, Streams, Metrics
- Footer: Keyboard shortcut hints

### 3.2 Rendering

- Use `ratatui::widgets::List` for each column with highlight state
- Color-coded health indicators: green (Ready), yellow (Provisioning), red (Offline)
- Focus ring: bold border on active pane
- Instance count badges next to each item

### 3.3 Input handling (`input.rs`)

- Map crossterm `KeyEvent` to `AppAction` enum
- hjkl/arrows for navigation
- Tab for focus cycling
- q/Esc/Ctrl+C for quit

### 3.4 App event loop (`app.rs`)

- `tokio::select!` over:
  - Crossterm event stream (keyboard input)
  - ETCD source channel
  - NATS source channel (Phase 4)
  - Metrics source channel (Phase 5)
  - Tick timer (250ms for UI refresh)
- Update state, then render

## Phase 4: NATS Monitoring (Day 4-5)

### 4.1 NATS connection (`sources/nats.rs`)

- Connect using `async-nats` with env-var config
- Poll `client.server_info()` for connection stats
- Query JetStream for stream names and consumer counts

### 4.2 Display

- Bottom status bar: connection state, message counts, byte counts
- Stream list with consumer counts

## Phase 5: Metrics & Polish (Day 5-6)

### 5.1 Prometheus scraping (`sources/metrics.rs`)

- HTTP GET to configurable metrics endpoint
- Parse Prometheus text exposition format
- Extract `dynamo_frontend_*` metrics (TTFT, TPOT, throughput, queue depth)

### 5.2 Polish

- Graceful error handling (show errors in UI, don't crash)
- Reconnection logic for ETCD and NATS
- Responsive layout on terminal resize
- Loading spinner during initial connection

## Phase 6: Testing & Documentation (Day 6-7)

### 6.1 Unit tests

- Tree building from flat ETCD entries
- Key parsing (namespace/component/endpoint extraction)
- Prometheus text format parsing
- Input mapping

### 6.2 Integration tests

- Mock ETCD data → verify rendered state
- NATS stats parsing

### 6.3 Documentation

- `lib/tui/README.md` — usage, screenshots, configuration
- `lib/tui/CLAUDE.md` — development guide
- Update design doc with final screenshots

## File-Level Dependency Graph

```
main.rs
  ├── app.rs (App state + event loop)
  │    ├── model.rs (data types)
  │    ├── sources/mod.rs (AppEvent enum)
  │    │    ├── sources/etcd.rs
  │    │    ├── sources/nats.rs
  │    │    └── sources/metrics.rs
  │    └── input.rs (key → action mapping)
  └── ui.rs (rendering)
       └── model.rs (reads state for display)
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| `etcd-client` requires protobuf compiler | Document in README; add `protobuf-compiler` to CI |
| macOS lacks `inotify` (etcd-client notify) | Document Docker-based dev workflow as fallback |
| ETCD not running locally | Graceful "Connecting..." UI state; don't crash |
| Large deployments (100s of endpoints) | Virtualized list rendering; paginate if needed |
