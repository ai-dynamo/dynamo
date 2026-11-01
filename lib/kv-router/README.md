# dynamo-kv-router

`dynamo-kv-router` provides the core KV-aware routing data structures and scheduling primitives used
by Dynamo to steer requests toward workers with the best cache overlap.

## What This Crate Provides

- `RadixTree` and `ConcurrentRadixTreeCompressed` for prefix-overlap indexing
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

## Writing a Plugin

A statically linked worker-selection policy or request classifier imports
`dynamo_kv_router::plugins::prelude::*` and depends on this crate alone; the prelude lists
every contract, signal view, wire type, configuration type, and registry type a plugin
names (`tests/plugin_api_prelude.rs` is a policy and a classifier written against it).

## Features

`default = []`: the flat configuration, the plugin registry, and the shims over the kernel
and the stage crates compile without any transport crate. Each adapter is a feature:

- `http`: the axum servers and their response envelope (`services::indexer::server`,
  `services::slot_tracker`, `services::selection::server`)
- `ingest-zmq`: the ZMQ KV-event listener, its registry, and `ZmqDirectIngress`
  (`services::indexer::{listener, registry}`, `services::selection::ZmqDirectIngress`)
- `replica-sync-zmq`: the ZMQ peer mesh that replicates slot-tracker and session-affinity
  state (`services::common::replica_sync`)
- `peer-recovery`: the HTTP client that replays a peer indexer's state on startup
  (`services::indexer::recovery`)

A service is a combination: the standalone indexer is `http` + `ingest-zmq` + `peer-recovery`;
the slot tracker is `http` + `replica-sync-zmq`. The selection service (`services::selection`)
compiles with no adapter: an embedding host supplies its own index ingress and replica
channels. Its ZMQ-direct ingress needs `ingest-zmq`, indexer peer recovery `ingest-zmq` +
`peer-recovery`, the replica mesh `replica-sync-zmq`, and its HTTP server and CLI config all
three plus `http`. The pre-split names stay for one release as aliases of those
combinations: `standalone-indexer`, `standalone-slot-tracker`, `standalone-selection`.

Other features:

- `metrics`: Prometheus metrics for router internals
- `runtime-protocols`: integration points with `dynamo-runtime`
- `bench`: internal benchmarking helpers
- `profile`: keeps hot indexer functions un-inlined for profiling
- `testing`: test-only hooks consumed by dependent crates' tests; never enable in release

## Testing the Services

The `services/` modules are feature-gated and are not compiled by default, so
a plain `cargo test -p dynamo-kv-router` silently skips their code and tests.
Enable the adapters when working on them, and run the per-feature build check
before pushing (it needs `cargo-hack`):

```bash
cargo test -p dynamo-kv-router --features http,ingest-zmq,replica-sync-zmq,peer-recovery
cargo clippy -p dynamo-kv-router --all-targets --features http,ingest-zmq,replica-sync-zmq,peer-recovery
bash scripts/router_features_check.sh
```

The crate's public surface is a committed snapshot, `public-api.txt`; `bash
scripts/router_public_api_check.sh` regenerates it with `cargo-public-api` and fails on a
difference (`--update` rewrites it; review the diff in the PR).

## Deprecated paths

`dynamo-kv-router` is the composition crate. Every path below is a shim over a stage, the kernel,
or a vocabulary crate; it carries `#[deprecated]` for one release and goes away in the next.
In-tree hosts import the owner's `host` module; plugins import `plugins::prelude`.

| Deprecated path | Owner |
|---|---|
| root re-exports of stage and kernel items (`WorkerSelectionPolicy`, `LocalScheduler`, `KvSchedulerError`, `SelectionConfig`, ...) | the owner's `host` module, `plugins::prelude`, or `config::` |
| `protocols::*` | `dynamo_router_protocol`, `dynamo_kv_events::event`, `dynamo_kv_hashing::routing`, `dynamo_kv_hints`, `dynamo_scheduler::protocols` |
| `identity::*` | `dynamo_router_protocol::identity` (indexer identity specs: `dynamo_kv_indexer::identity`) |
| `indexer::*`, `concurrent_radix_tree_compressed`, `nested_map`, `approx`, `radix_tree`, `session_prefix_index` | `dynamo_kv_indexer` (`host`, `cuckoo`, `api`) |
| `sequences`, `multi_worker_sequence`, `sequence` | `dynamo_slot_tracker` (`host`) |
| `scheduling::{queue, selector, overlap, overlap_refresh, prefill_load, protocols, router_config, shared_cache, tracking_hash, classifier_rows}` and its kernel re-exports | `dynamo_scheduler` (`host`) |
| `scheduling::{policy_queue, request_classifier, admission_config, queue_admission}` | `dynamo_admission` (`host`) |
| `scheduling::lora_filter`, `conditional_disagg` | `dynamo_worker_selection` (`host`) |
| `kv_hints` | `dynamo_kv_hints`, `dynamo_kv_indexer::kv_hints`, `dynamo_scheduler::kv_hints` |
| `recovery` | `dynamo_kv_events::cursor` |
| `zmq_wire` | `dynamo_kv_events::zmq_wire` |
| `worker_type` | `dynamo_router_protocol::WorkerType` |
| `services::selection::{affinity, SelectionCore, SelectionHost, PromptRequest, SelectRequest, ...}` | `dynamo_scheduler::host`, `dynamo_session_affinity::host` |
| `services::overlap` | `dynamo_scheduler::overlap_summary` |

The `standalone-indexer`, `standalone-slot-tracker`, and `standalone-selection` features stay as
aliases of the adapter features for the same release; `runtime-protocols` is empty.

## Further Reading

- Router guide: <https://docs.nvidia.com/dynamo/components/router>
- Indexer internals:
  <https://github.com/ai-dynamo/dynamo/blob/main/lib/router/kv-indexer/src/README.md>
- Benchmarking the sharded KV indexer: [INDEXER_BENCH.md](../bench/kv_router/INDEXER_BENCH.md)
- Dynamo repository: <https://github.com/ai-dynamo/dynamo>
