# lib/router/kv-router

`dynamo-kv-router` is the composition crate: it assembles the kernel and the stages into a running router, puts every transport (HTTP, ZMQ, the replica mesh, Prometheus) behind a feature, and is the one crate plugins name: `plugins::worker_selection` or `plugins::request_classifier` for the plugin API, `plugins::experimental` for APIs with no compatibility promise, and `plugins::testing` for their tests and benches. `plugins::host` is for routing hosts: resolving plugins is `RouterPluginRegistryHostExt` there, not a method on the registry a plugin's `register` receives. Two subdirectories have their own rules: [`src/services/indexer/AGENTS.md`](src/services/indexer/AGENTS.md) and [`src/services/slot_tracker/AGENTS.md`](src/services/slot_tracker/AGENTS.md).

## Guardrails

- **Adding a dependency or an import.** Don't depend on `dynamo-runtime`, directly or behind a feature, and don't assume a Dynamo request, discovery, or event plane exists: processes that run no other Dynamo component use this crate.
- **Adding to `src/services/`.** Use brokerless transports (TCP, HTTP, ZMQ) and keep each service's protocol usable by a process that runs no other Dynamo component. Keep reusable routing, indexing, and tracking logic outside `src/services/`.
- **Adding a transport or an optional dependency.** Put it behind a feature. `default = []` builds the configuration, the plugin registry, and the selection service with no transport crate. CI checks every feature alone (`cargo hack check --each-feature`), so gate code that needs two features with `cfg(all(...))`. The README's Features section lists what each feature turns on.
- **Adding a type a plugin needs.** Add it by name (no glob of another crate) to `src/plugins/worker_selection.rs` or `src/plugins/request_classifier.rs`, or once to `mod shared` in `src/plugins.rs` when a policy and a classifier both name it, as they do the worker rows; the root `plugins` module and both kind modules re-export `shared` whole. Use it in `tests/plugin_api.rs`, which is a policy and a classifier written against those two modules alone. An API that may still change or be removed (the dispatch gate, hint policies, #18443) goes in its own submodule of `src/plugins/experimental.rs`, and its names move into a kind's module when it is ready. A type only a routing host names goes in `src/plugins/host.rs` when it is defined in this crate, and otherwise stays in its owning crate's `host` module; the two kind modules and the root `plugins` module hold no host-only name. Never add a kernel or host type to a kind's module so that a plugin's test can build a request: extend `src/plugins/testing.rs`.
- **Adding a re-export to `src/compat.rs`.** Don't. It holds only the pre-split paths, all deprecated and removed in 1.10. New items are imported from the crate that owns them. When you delete an item, delete its alias there too (#18798, #18799, #18800, #18802).
- **Enabling `testing` or `bench`.** Only from `[dev-dependencies]`. They expose test-only hooks in the stage crates.

## Test and bench

A plain `cargo test -p dynamo-kv-router` skips the indexer and slot-tracker services and every adapter, because they are feature-gated. Run the second command when you touch `src/services/`.

```bash
cargo test -p dynamo-kv-router --lib --tests
cargo test -p dynamo-kv-router --features http,ingest-zmq,replica-sync-zmq,peer-recovery,metrics --lib --tests
cargo hack check --locked -p dynamo-kv-router --each-feature --all-targets   # needs cargo-hack
cargo bench -p dynamo-kv-router --bench policy_queue
cargo bench -p dynamo-kv-router --bench tracking_hash
cargo bench -p dynamo-kv-router --features bench --bench worker_selection
cargo bench -p dynamo-kv-router --features bench --bench admission_core
cargo bench -p dynamo-kv-router --features ingest-zmq,replica-sync-zmq,peer-recovery,bench --bench selection_core
```

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was applied.

- **Adding a field to a request type that carries a prompt.** Add it to the flat wire struct (`inline_prompt_wire!` in `lib/router/scheduler/src/selection/types.rs`) and parse from bytes as `PromptJson` does. `#[serde(flatten)]` or an intermediate `serde_json::Value` buffers the whole body, which holds every prompt token (#17238).
- **Opening a ZMQ socket.** Use the pooled helpers in `src/services/common/zmq.rs` (`create_sub_socket_topics`, `connect_sub_socket`, `connect_dealer_socket`); a context per socket costs two threads per socket. `create_bound_pub_socket` is the exception: it takes its own context so its port is free as soon as its owner shuts down, so don't open one per worker. Open a per-worker socket that is rarely used on demand, as `replay_gap` dials its DEALER (#17253).
