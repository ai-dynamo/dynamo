# lib/zmq

`dynamo-zmq` is the ZMQ socket helper and nothing else: socket construction, async multipart IO, the context pool, endpoint validation. Frame codecs belong in `dynamo-kv-events`; the listeners and meshes that drive these sockets belong in `dynamo-kv-router` adapters. `create_bound_pub_socket` is deliberately not pooled: dropping it must free the port when its owner shuts down.

- Depends on (Dynamo crates): none (zmq, tokio, tracing, anyhow only).
- Must not depend on: the runtime, any router crate, axum, hyper, reqwest, prometheus.
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
