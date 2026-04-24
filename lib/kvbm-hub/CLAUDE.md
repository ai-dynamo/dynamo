# CLAUDE.md

## Commands

```bash
# check
cargo check -p kvbm-hub

# All tests (always use timeout — async tests can deadlock)
timeout 60 cargo test -p kvbm-hub

# Single test
timeout 60 cargo test -p kvbm-hub -- <test_name>

# Lint
cargo clippy -p kvbm-hub
cargo fmt
cargo machete
```

## Architecture

`kvbm-hub` is a library crate targeting clients and a server executable (kvbm_hub). It provides a central HTTP coordination service for KVBM velo clients and is consumed by other crates in the workspace.

**Dual-port design:**
- **Discovery port** (default `1337`) — read-only `GET` endpoints implementing the `velo::discovery::PeerDiscovery` trait over HTTP. Clients hit this for peer lookups.
- **Control port** (default `8337`) — write endpoints (register, unregister, heartbeat) plus mirrored discovery endpoints. Also the entry point for velo active messaging.

**Key types and their roles:**

| Type | Module | Role |
|------|--------|------|
| `HubServerState` | `server` | `Arc<RwLock<Registry>>` — in-memory `InstanceId`↔`PeerInfo` + `WorkerId`↔`InstanceId` maps |
| `HubServer` | `server` | Running server with two axum listeners; `CancellationToken` shutdown |
| `HubClient` | `client` | `reqwest`-based HTTP client; implements `PeerDiscovery`; holds an `OnceLock<HubRegistrationGuard>` |
| `HubRegistrationGuard` | `client` | RAII guard — issues HTTP `DELETE` on drop (best-effort via `handle.spawn`) |
| `protocol` | `protocol` | Shared JSON request/response types and URL path constants |
| `handlers` | `handlers` | Velo active-message handlers installed on the client side (currently only heartbeat) |

**Two messaging planes:**
1. **HTTP** (bootstrap + discovery) — client registers via `POST /v1/instances`, looks up peers via `GET /v1/peers/...`, heartbeats via `POST /v1/instances/{id}/heartbeat`.
2. **Velo active messaging** (post-discovery control plane) — hub pushes heartbeats to registered clients via velo handlers (`_kvbm_hub_heartbeat`). Registered by calling `HubClient::register_handlers` before `register_instance`.

**Integration test pattern:** tests bind both ports to `0` (OS-assigned) and extract the actual addresses from the returned `HubServer`. This avoids port conflicts across parallel test runs.

## Configuration (figment merge strategy)

Priority order (lowest → highest):

1. `HubConfig::default()` — compiled-in defaults (`bind_addr = 0.0.0.0`, `discovery_port = 1337`, `control_port = 8337`)
2. TOML or JSON file — path from `--config` CLI arg or `KVBM_HUB_CONFIG` env var; format auto-detected by `.toml`/`.json` extension
3. `KVBM_HUB_*` env vars — `KVBM_HUB_BIND_ADDR`, `KVBM_HUB_DISCOVERY_PORT`, `KVBM_HUB_CONTROL_PORT`
4. CLI args — `--bind-addr`, `--discovery-port`, `--control-port` (highest priority; only override when explicitly passed)

`KVBM_HUB_CONFIG` is consumed by the CLI's `--config` arg before `HubConfig::figment()` is called, and is explicitly ignored in the env layer to avoid double-processing.

**Why `Option<T>` CLI fields (not clap defaults):** CLI fields in the binary are `Option<T>` with no defaults. Only `Some` values are merged into Figment as a final layer. This ensures CLI args only win when explicitly passed — clap defaults would otherwise shadow env vars for fields the user never specified.
