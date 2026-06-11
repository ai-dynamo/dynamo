# Runtime

The `KvbmRuntime` is the composed shared infrastructure for KVBM operations. It bundles
the minimal set of components that all downstream managers and services need:

- **Tokio runtime** -- async execution context (owned or borrowed handle)
- **Messenger (Velo)** -- distributed RPC for leader/worker communication and peer discovery
- **NixlAgent** -- RDMA/UCX data transfers (optional, disabled when NixL config is absent)
- **EventManager** -- worker coordination and transfer completion notifications (accessed via Messenger)

## Construction

Two quick constructors cover the common case:

```rust,ignore
// Leader role (reads KVBM_* env vars + TOML files)
let runtime = KvbmRuntime::from_env_leader().await?;

// Worker role
let runtime = KvbmRuntime::from_env_worker().await?;
```

For tests or custom setups, use the builder:

```rust,ignore
let config = KvbmConfig::from_env()?;
let runtime = KvbmRuntime::builder(config)
    .with_runtime_handle(Handle::current())   // inject existing tokio runtime
    .with_messenger(messenger)                // inject pre-built Messenger
    .with_nixl_agent(agent)                   // inject pre-built NixlAgent
    .build_leader()
    .await?;
```

`KvbmRuntimeBuilder::from_json(json)` is the primary entrypoint for vLLM's
`kv_connector_extra_config` dict -- JSON values have highest priority, overriding
env vars, TOML files, and defaults.

## Component access

| Method              | Returns                      | Notes                                 |
|---------------------|------------------------------|---------------------------------------|
| `handle()` / `tokio()` | `tokio::runtime::Handle`  | Borrowed or owned runtime handle      |
| `messenger()`       | `&Arc<Messenger>`            | Velo RPC                              |
| `nixl_agent()`      | `Option<&NixlAgent>`        | `None` when NixL disabled in config   |
| `event_system()`    | `Arc<velo::EventManager>`   | From Messenger, used for transfer notifications |
| `config()`          | `&KvbmConfig`               | Full configuration snapshot            |

## RuntimeHandle

`RuntimeHandle` is an enum that abstracts over owned (`Arc<Runtime>`) and borrowed
(`Handle`) tokio runtimes. The builder creates an owned runtime from config when none
is injected.
