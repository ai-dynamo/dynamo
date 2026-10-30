# lib/zmq

- This crate is the ZMQ socket helper and nothing else: socket construction, async multipart IO, the context pool, endpoint validation.
- Depend only on zmq, tokio, tracing, anyhow. Never add a router crate, `dynamo-runtime`, axum, reqwest, or prometheus; the gate rejects them.
- Frame encoding and decoding belong in `dynamo-kv-events`; listeners and meshes that drive these sockets belong in `dynamo-kv-router` adapters.
- `create_bound_pub_socket` is deliberately not pooled: dropping it must free the port when its owner shuts down.
- Run `scripts/router_layering_check.sh` before committing.
