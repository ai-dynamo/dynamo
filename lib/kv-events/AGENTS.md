# lib/kv-events

- This crate owns KV cache event types, the `RouterEvent` envelope, the engine ZMQ codec, the replay cursor, the publisher placement helpers, and the residency vocabulary engines publish (`residency`: storage tiers, residency domains and owners, the reset scope). Nothing else belongs here.
- Depend only on `dynamo-router-protocol`, `dynamo-kv-hashing`, serde, serde_json, rmp-serde, rustc-hash, thiserror, tracing. Never add `dynamo-runtime`, `dynamo-kv-router`, tokio, zmq, axum, reqwest, or prometheus; the gate rejects them.
- Identities live in `dynamo-router-protocol`; block hash functions live in `dynamo-kv-hashing::routing`. Import them, do not redefine them. The residency vocabulary is owned here: consumers (kv-hints, the indexer, the kernel, lib/llm) import `dynamo_kv_events::residency` directly.
- Changing a wire shape here changes what engines and sidecars publish. Keep serde compatibility tests next to the type.
- Run `scripts/router_layering_check.sh` before committing.
