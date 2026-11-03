# lib/router/protocol

`dynamo-router-protocol` is data: the identities and the worker-facing wire vocabulary every router crate names (`attempt`, `columns`, `identity`, `session`, `worker`, `worker_type`). A type enters only if it crosses a process boundary to an engine, a worker, or a sidecar, or is an identity every crate names; behavior beyond serde derives and conversions belongs to a consumer crate, and the gate greps `src/` for `dyn`, `async`, `std::fs`, `anyhow`, `tracing`, and `derive_builder`. `AttemptIdMinter::mint` is the only source of attempt ids; the chosen-value constructors stay behind `testing` and `bench`, which no normal dependency may enable.

- Depends on (Dynamo crates): none; `[dependencies]` stays `serde`, `serde_json`, `thiserror`.
- Must not depend on: the runtime, tokio, any transport, any router crate.
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
