# lib/router/kv-hints

`dynamo-kv-hints` owns the KV transfer hint wire (`KvHint`, `KvHintAction`, `KvSourceLocationsPayload`, the `KV_HINT_TRANSFER_*` keys), the transfer source choice, and the worker transfer keys; the transfer candidate vocabulary is re-exported from `dynamo-kv-indexer`, which produces it. Changing a wire shape changes what engines and sidecars receive; keep serde compatibility tests next to the type.

- Depends on (Dynamo crates): `dynamo-kv-hashing`, `dynamo-kv-indexer`, `dynamo-router-protocol`.
- Must not depend on: the runtime, tokio, any transport, any other stage, the kernel, kv-router; the edge points one way (a feature crate depends on the stage that produces its types; the indexer never names this crate).
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
