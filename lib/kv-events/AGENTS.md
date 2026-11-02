# lib/kv-events

- This crate owns KV cache event types, the `RouterEvent` envelope, the engine ZMQ codec, the replay cursor, the publisher placement helpers, and the residency vocabulary engines publish (`residency`: storage tiers, residency domains and owners, the reset scope). Nothing else belongs here.
- Depend only on `dynamo-router-protocol`, `dynamo-kv-hashing`, serde, serde_json, rmp-serde, rustc-hash, thiserror, tracing. Never add `dynamo-runtime`, `dynamo-kv-router`, tokio, zmq, axum, reqwest, or prometheus; the gate rejects them.
- Worker and owner identities live in `dynamo-router-protocol`; the block hash identities (`LocalBlockHash`, `ExternalSequenceBlockHash`) and the functions that compute them live in `dynamo-kv-hashing`. Import them, do not redefine them. The residency vocabulary is owned here: consumers (the indexer, the kernel, lib/llm) import `dynamo_kv_events::residency` directly. The compact `ResidencyOwnerKey` is not: the indexer computes it, so it lives in `dynamo_kv_indexer::residency`.
- Changing a wire shape here changes what engines and sidecars publish. Keep serde compatibility tests next to the type.
- Run `scripts/router_layering_check.sh` before committing.
