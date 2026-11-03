# lib/kv-events

`dynamo-kv-events` owns the KV cache event types, the `RouterEvent` envelope, the engine ZMQ codec, the replay cursor, the publisher placement helpers, and the residency vocabulary engines publish (`residency`). Changing a wire shape changes what engines and sidecars publish; keep serde compatibility tests next to the type.

- Depends on (Dynamo crates): `dynamo-kv-hashing`, `dynamo-multimodal`, `dynamo-router-protocol`.
- Must not depend on: the runtime, tokio, any transport (zmq, axum, hyper, reqwest, prometheus), any router crate beyond those three.
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
