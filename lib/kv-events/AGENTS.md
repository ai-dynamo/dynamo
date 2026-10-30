# lib/kv-events

- This crate owns KV cache event types, the `RouterEvent` envelope, the engine ZMQ codec, the replay cursor, and the publisher placement helpers. Nothing else belongs here.
- Depend only on `dynamo-router-protocol`, `dynamo-kv-hashing`, serde, serde_json, rmp-serde, rustc-hash, thiserror, tracing. Never add `dynamo-runtime`, `dynamo-kv-router`, tokio, zmq, axum, reqwest, or prometheus; the gate rejects them.
- Identities and residency vocabulary live in `dynamo-router-protocol`; block hash functions live in `dynamo-kv-hashing::routing`. Import them, do not redefine them.
- Changing a wire shape here changes what engines and sidecars publish. Keep serde compatibility tests next to the type.
- Run `scripts/router_layering_check.sh` before committing.
