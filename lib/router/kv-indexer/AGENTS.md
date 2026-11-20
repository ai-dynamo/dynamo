# lib/router/kv-indexer

`dynamo-kv-indexer` is the KV indexer stage: it applies KV events to prefix indexes and answers which workers hold a request's prefix. Hosts outside `lib/router` import through `host` and the DC relay through `cuckoo`; the index design notes are in `src/README.md`.

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was applied.

The index holds an entry per (block, worker), tens of millions at fleet scale, and every KV event from every worker passes through it. The `dynamo-perf-best-practices` skill has the general rules; `src/concurrent_radix_tree_compressed/AGENTS.md` adds the CRTC ones.

- **Adding a field or variant to anything stored per block, per edge, or per (block, worker).** Work out the new `size_of` first. Box the rare case, intern wide ids, and keep or add a `const` size assert, as `lane_lookup.rs` does for its 12-byte slot (#17109, #17143).
- **Adding a lookup on the event-apply or match path.** Count the cache lines it touches per block, not its instructions: each lookup-table probe is a likely cache miss. Carry an id you already hold instead of probing again, as `upsert_all_held` does (#18108).
- **Adding a mutation path** (a new event kind, cleanup, recovery). Send it through the lane its stream already uses, which `event_thread_indices` picks: the worker's sticky lane, the cache owner's, or both for a reset that covers both (`ResetScope::All`). A mutation that touches both domains goes to both lanes. Applied from another thread, it can overtake that stream's queued events (#11190).
- **Adding a walk over the whole index** (a sweep or an audit). Keep it off the event lanes, as the CRTC sweep runs on the `kv-maint` thread, and scale its trigger with the size of what it walks, with a period only as a backstop (#16313, #18084).
- **Using `ArcSwap` or `DashMap` on a per-event or per-node path.** Don't store to or drop an `ArcSwap` there: each one walks every reader thread's debt list, so keep a lock in the node as `NodeChildren` does. Don't call `len` or `is_empty` on a `DashMap` or iterate it there: each locks every shard, so keep a counter. `load` and `get` are fine (#18574, #16935).
- **Changing the lower-tier walk** (`lower_tier.rs`). Its cost must stay proportional to the prefix, not to the prefix times the owners or hint sources. Make the same change in `advance_scalar` and in the general loop it stands in for (#16436, #17149).
- **Seeing dropped or lagging events.** Find what the path does per event and remove it. Don't raise a queue bound or a high-water mark.
- **Before the PR.** Run `mooncake_bench` as `lib/bench/kv_router/INDEXER_BENCH.md` describes: the overloaded replay for a write-path change, a keep-up replay for a read-path change, and lower-tier mode for `lower_tier.rs`. Dynamo's Python extension links jemalloc, so compare with jemalloc preloaded and the options from `lib/bindings/python/rust/allocator.rs` passed in `MALLOC_CONF`.
