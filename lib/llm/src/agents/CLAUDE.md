# Agent Runtime Rust Modules

This folder contains Rust support for agent-facing request metadata in
`dynamo-llm`. Request trace owns emission, replay hashing, tool-event relay,
and trace sinks.

## Scope

- Keep `mod.rs` files as module wiring and public re-export surfaces when
  possible. Put implementation-heavy logic in named modules.
- Keep hot-path request handling lean. Agent metadata should remain cheap to
  clone and safe to carry through the OpenAI preprocessing path.
- Put request-trace enrichment under `request_trace/`; this folder should not
  grow a second tracing subsystem.

## Validation

For changes under `agents/`, run:

```bash
cargo check -p dynamo-llm
cargo test -p dynamo-llm request_trace --lib
```
