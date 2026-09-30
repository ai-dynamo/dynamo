# Request Trace Rust Modules

This folder owns `DYN_REQUEST_TRACE` capture in `dynamo-llm`: replay hashing,
request-end emission, optional `agent_context` enrichment, local trace sinks,
and optional harness tool-event ingestion.

## Scope

- Keep `mod.rs` as the module wiring, public re-export surface, and request-trace
  runtime entry point.
- Keep request handling lean. Do not add file, network, or heavy serialization
  work inline on the hot path; publish bounded records to the trace bus and let
  sinks do the I/O.
- Treat `agent_context` as additive metadata on eligible request-trace rows. It
  should not create a second tracing subsystem or bypass replay-shape checks.
- Keep the ZMQ tool-event bridge optional and behind the request-trace tool-event
  env vars.

## Validation

For changes under `request_trace/`, run:

```bash
cargo fmt --check
cargo check -p dynamo-llm --lib --no-default-features
cargo test -p dynamo-llm --lib request_trace --no-default-features
```
