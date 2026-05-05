# Bidi Streaming Example

End-to-end demo of bidirectional streaming over the velo request plane.

The user-facing surface is intentionally narrow:

- **Server**: implement `AsyncEngine<ManyIn<T>, ManyOut<U>, Error>` and
  register with `EndpointConfigBuilder::bidi_engine::<T, U, I>(engine)`.
- **Client**: call `PushRouter::bidi_generate(many_in)` (or
  `bidi_generate_with::<I>(many_in, init)` to ship a typed init payload
  with the handshake). You get back a regular `ManyOut<U>` — no protocol
  details leak through.

Under the hood this opens two velo SPSC streams (one per direction),
exchanges anchor handles via a single velo unary kick-off, and runs a
half-close protocol so each side can independently say "no more `Data`"
without prematurely tearing down the channel.

## Layout

```
bidi_streaming/
├── Cargo.toml
└── src/
    ├── lib.rs                   # shared constants
    └── bin/
        ├── server.rs            # registers the bidi handler
        └── client.rs            # opens a session and prints responses
```

## What the example does

1. **Client** builds a `Stream<String>` of 4 input items and opens a bidi
   session, passing `init = "demo"` with the handshake.
2. **Server** reads `init` from the request's `Context` registry under
   `BIDI_INIT_KEY`, uppercases each input item, prefixes it with the init
   payload, and emits the result.
3. After the client's input ends (peer sent `BidiFrame::Done`), the server
   continues to emit a single trailing line with a count — demonstrating
   the half-close.
4. When the server's response stream ends, both sides finalize their velo
   senders and the client's `ManyOut<U>` returns `None`.

## Running

The bidi path requires the velo request plane.

```bash
# Terminal 1 (server)
DYN_REQUEST_PLANE=velo cargo run -p bidi_streaming --bin server

# Terminal 2 (client)
DYN_REQUEST_PLANE=velo cargo run -p bidi_streaming --bin client
```

Expected client output:

```text
client: streaming 4 input items...
server -> [demo] HELLO
server -> [demo] BIDI
server -> [demo] STREAMING
server -> [demo] WORLD
server -> (handler saw 4 item(s))
client: response stream ended cleanly
```

## Notes

- The handler uses `request.get::<String>(BIDI_INIT_KEY)` to recover the
  init payload — the framework stuffs it there during the handshake. Use
  the same key on the server side that you parameterized on the client
  side (`bidi_generate_with::<String>(...)` here).
- `T` and `U` only need `Serialize + DeserializeOwned`; for `U` we pick
  `Annotated<String>` because `PushRouter` requires `U: MaybeError`.
- Cancellation is automatic. Drop the `ManyOut<U>` on the client and the
  server's handler observes `ctx.killed()` within bounded time; both
  velo streams tear down via the cross-network `_stream_cancel` AM.
