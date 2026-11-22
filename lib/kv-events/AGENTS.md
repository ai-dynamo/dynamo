# lib/kv-events

`dynamo-kv-events` is the KV cache event wire between engines and the router: the event types and the `RouterEvent` envelope, the engine ZMQ msgpack codec (`zmq_wire`), the replay cursor, and the residency vocabulary. Workers and sidecars publish; the router's indexer ingests. Every KV event passes through this crate.

## Rules

- **Changing the engine codec (`src/zmq_wire/`).** vLLM, SGLang, and TensorRT-LLM publish the shapes listed in `src/zmq_wire/README.md`, and released engines keep publishing them. Add a field as a named map key or at the end of the optional tail. Don't reorder or remove a positional slot. Add a case to `src/zmq_wire/tests.rs` for the new shape and update that README.
- **Dropping an event in `ZmqEventNormalizer`.** Give the drop a `ZmqEventFilterReason`. `as_label` is a metric label, so don't rename an existing one. A dropped store also drops every later block of that chain, so drop only what is truly ambiguous, and add a test for the case you keep (#18060).
- **Changing `RouterEvent` or `KvCacheEvent` (`src/event.rs`).** Both ends serialize them. Give a new field `#[serde(default)]`, and `skip_serializing_if` when it is optional, as the existing fields do.
- **Changing `CursorState::observe` (`src/cursor.rs`).** Event ids rise by one. `observe` reports `Initial`, `Contiguous`, `Gap`, or `Stale`; the listener in `lib/router/kv-router/src/services/indexer/listener.rs` replays the missing range on a gap (`replay_gap`) and skips a stale id.
- **Adding a socket, task, or server.** Not here. Those are `dynamo-kv-router` adapters behind its ZMQ features.

## Test

```bash
cargo test -p dynamo-kv-events --lib
```
