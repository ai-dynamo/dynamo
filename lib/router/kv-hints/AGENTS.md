# lib/router/kv-hints

`dynamo-kv-hints` owns the KV transfer hint the router attaches to a selected request (`KvHint` and the `KV_HINT_TRANSFER_*` keys), the transfer source choice, and the worker transfer keys. Changing a wire shape changes what engines and sidecars receive, so keep the serde compatibility tests next to the type.
