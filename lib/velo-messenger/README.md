# velo-messenger

Active messaging layer for Velo distributed systems. Sits between `velo-backend` (transport abstraction) and higher-level session/service crates, providing request-response and fire-and-forget messaging patterns over pluggable transports.

## Architecture

```
velo-common  →  velo-backend  →  velo-messenger  →  (higher-level crates)
  (types)       (transports)     (messaging)
```

`Messenger` is the central type. It owns a `velo-backend::VeloBackend`, wires up inbound message dispatch, and exposes a builder-based API for both registering handlers and sending messages.

### Modules

```
src/
  messenger.rs       Messenger + MessengerBuilder
  discovery.rs       PeerDiscovery trait
  client/            ActiveMessageClient, send/unary/typed-unary builders
  handlers/          Handler definitions, builder API, dispatch adapters
  server/            Inbound message dispatch, system handlers (_hello, _list_handlers)
  common/            Wire format (MessageId, encoding, ResponseManager)
```

## Messaging Patterns

**Fire-and-forget** -- send a message with no response expected:
```rust
messenger.am_send("notify")?.payload(&data)?.instance(peer).send().await?;
```

All four patterns are async. For fire-and-forget, the future completes once the message has been issued to the transport. Because most transports are reliable (TCP, gRPC, NATS), this provides strong delivery guarantees even without a response. For the remaining three patterns, the future completes only after the remote handler has finished executing.

**Synchronous (ack/nack)** -- send and wait for the remote handler to complete:
```rust
messenger.am_sync("process")?.payload(&data)?.instance(peer).send().await?;
```

**Unary (request-response)** -- send and receive raw bytes:
```rust
let response: Bytes = messenger.unary("ping")?.raw_payload(Bytes::new()).instance(peer).send().await?;
```

**Typed unary** -- automatic serde serialization:
```rust
let resp: MyResponse = messenger.typed_unary::<MyResponse>("rpc")?.payload(&request)?.instance(peer).send().await?;
```

## Registering Handlers

Handlers are registered on the `Messenger` and dispatched to incoming messages by name. Each handler type has sync and async variants, plus a dispatch mode (`.spawn()` for task isolation, `.inline()` for minimal latency).

```rust
// Sync unary handler
let handler = Handler::unary_handler("ping", |_ctx| Ok(Some(Bytes::new()))).build();
messenger.register_handler(handler)?;

// Async typed handler with automatic deserialization/serialization
let handler = Handler::typed_unary_async("add", |ctx: TypedContext<AddRequest>| async move {
    Ok(AddResponse { sum: ctx.input.a + ctx.input.b })
}).build();
messenger.register_handler(handler)?;
```

Handler context objects (`Context`, `TypedContext`) include a reference to the `Messenger` via `ctx.msg`, allowing handlers to send messages, query peers, or register new handlers.

## Peer Discovery

Discovery is abstracted behind the `PeerDiscovery` trait. Higher-level crates provide implementations (etcd, consul, etc.) without pulling those dependencies into this layer.

```rust
let messenger = Messenger::builder()
    .add_transport(tcp_transport)
    .discovery(my_discovery_impl)
    .build()
    .await?;
```

When no discovery backend is configured, peers must be registered manually via `messenger.register_peer(peer_info)`.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `http`  | yes     | HTTP transport (via `velo-backend`) |
| `grpc`  | yes     | gRPC transport |
| `nats`  | yes     | NATS transport |
| `ucx`   | no      | UCX transport (requires system libs) |

## Examples

See [`examples/ping_pong.rs`](examples/ping_pong.rs) for an end-to-end benchmark that creates two `Messenger` instances on separate runtimes and measures unary RTT over TCP.

```sh
cargo run -p velo-messenger --example ping_pong -- --rounds 5000
```

## Tests

Unit tests live alongside the source in each module. Run them with:

```sh
cargo test -p velo-messenger
```
