# lib/router/kv-indexer

`dynamo-kv-indexer` is the KV indexer stage: it applies KV events to prefix indexes and answers which workers hold a request's prefix. Hosts outside `lib/router` import through `host` and the DC relay through `cuckoo`; the index design notes are in `src/README.md`.
