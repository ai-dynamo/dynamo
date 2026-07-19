# dynamo-kv-router

`dynamo-kv-router` provides the core KV-aware routing data structures and scheduling primitives used
by Dynamo to steer requests toward workers with the best cache overlap.

## What This Crate Provides

- `RadixTree` and `ConcurrentRadixTree` for prefix-overlap indexing
- `ThreadPoolIndexer` and `PositionalIndexer` for higher-throughput index backends
- `KvRouterConfig`, `RouterQueuePolicy`, and `LocalScheduler` for request routing
- Protocol and hashing helpers such as `RouterEvent`, `WorkerId`,
  `compute_block_hash_for_seq`, and `compute_seq_hash_for_block`

## Basic Rust Usage

```rust
use dynamo_kv_router::{
    KvRouterConfig, RadixTree, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::protocols::BlockHashOptions;

let prompt_tokens = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
let local_hashes = compute_block_hash_for_seq(&prompt_tokens, 4, BlockHashOptions::default());
let seq_hashes = compute_seq_hash_for_block(&local_hashes);

let router_config = KvRouterConfig::default();
let index = RadixTree::new();
let scores = index.find_matches(local_hashes, false);

assert!(router_config.use_kv_events);
assert_eq!(seq_hashes.len(), 2);
assert!(scores.scores.is_empty());
```

For end-to-end routing, pair the indexers with `LocalScheduler` and the worker/config protocol
types re-exported from the crate root.

## Runtime Worker-Selector Plugins

KV routers can load a trusted Rust `cdylib` with `DYN_ROUTER_WORKER_SELECTOR_PLUGIN` and an optional `DYN_ROUTER_WORKER_SELECTOR_CONFIG`. The plugin receives independent state for each decode or prefill router and either chooses from Dynamo's already eligible candidates or delegates to the configured default selector; candidate data includes existing cache, load, taint, stable-ID, and capacity signals. Dynamo retains pinning, queueing, accounting, and result validation. Build the [example](examples/custom-worker-selector) and set the plugin path to its absolute `.so` path. The callback is on the scheduler actor's hot path, so it must be bounded and non-blocking.

## Features

- `metrics`: Prometheus metrics for router internals
- `runtime-protocols`: integration points with `dynamo-runtime`
- `standalone-indexer`: standalone indexer service support
- `bench`: internal benchmarking helpers

## Further Reading

- Router guide: <https://docs.nvidia.com/dynamo/components/router>
- Indexer internals:
  <https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/indexer/README.md>
- Benchmarking the sharded KV indexer: [INDEXER_BENCH.md](../bench/kv_router/INDEXER_BENCH.md)
- Dynamo repository: <https://github.com/ai-dynamo/dynamo>
