# Agent Runtime Rust Modules

This folder contains Rust support for agent-facing request metadata and trace
emission in `dynamo-llm`.

## Scope

- Keep `mod.rs` files as module wiring and public re-export surfaces when
  possible. Put implementation-heavy logic in named modules.
- Keep hot-path request handling lean. Agent trace work should stay gated by the
  trace policy and should publish bounded records to the async trace bus instead
  of doing file or network I/O inline.
- Keep agent trace schemas in `trace/types.rs`, sink behavior in `trace/sink.rs`,
  event-plane/ZMQ relay behavior in `trace/relay.rs`, and record construction or
  validation in `trace/record.rs`.
- Do not expose Python bindings for agent trace internals unless a concrete
  Python caller needs them in the same change.

## Validation

For changes under `trace/`, run:

```bash
cargo check -p dynamo-llm
cargo test -p dynamo-llm agents::trace --lib
```
