# lib/router/kv-hints

`dynamo-kv-hints` is what the router hands an engine so the selected worker can fetch a KV prefix from another worker: the hint wire, the hashed worker identity, and the source choice. Engines and sidecars consume it; the scheduler kernel and hosts produce with it.

## Rules

- **Adding a new kind of hint.** Add it as an action with its own `action_type` and `action_version`, not as a field on `KvHint`. `KvHintAction::payload` stays an opaque map so an engine forwards an action it does not know (`deserializes_unknown_action_as_opaque_transport_data`). The `kv.fetch` payload is `KvSourceLocationsPayload` (#17359, #17360).
- **Setting an action id.** Don't. `KvHint::append_numbered` numbers the actions `a1`, `a2`, and so on, and rejects a whole batch that would pass `MAX_KV_HINT_ACTIONS` (`appended_actions_are_numbered_and_bounded`).
- **Renaming a `KV_HINT_TRANSFER_*` key in `src/wire.rs`.** Don't. Workers advertise the capability under these strings in their runtime config, and `components/src/dynamo/common/constants.py` carries the same strings.
- **Changing the source choice (`src/transfer.rs`).** A source shares the target's hint worker type, has a control endpoint, and is neither the target nor a cache owner attached to it. When the target sets a `kv_transfer_domain`, the source is on the same island, unless enforcement is `preferred` and no same-island source qualifies. Worker selection and the hint read one `KvTransferSourceSummary`, so a change applies to both.
- **Hashing a label.** Use `label_key` and `topology_keys` from `dynamo-kv-indexer`, so worker keys and cache-owner keys hash the same way.
- **Adding a dependency.** Among router crates this one depends on `dynamo-router-protocol`, `dynamo-kv-hashing`, and `dynamo-kv-indexer` only. No transport and no stage logic. Attaching the hint to a request stays in `dynamo-scheduler`.

## Test

```bash
cargo test -p dynamo-kv-hints --lib
```
