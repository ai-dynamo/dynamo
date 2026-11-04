# dynamo-kv-router

`dynamo-kv-router` is the composition crate of the router: it assembles the scheduler kernel
(`dynamo-scheduler`) and the stage crates into a running router that steers requests toward
workers with the best cache overlap, and it is the one crate plugins and standalone consumers
name.

## What This Crate Provides

- `services::selection::{SelectionService, SelectionServiceBuilder}`: the composed router an
  embedding host or the standalone service runs
- `plugins`: the plugin registry and the worker-selection and request-classifier contracts;
  `plugins::prelude` is the single import a plugin needs
- `config::KvRouterConfig` with its CLI, environment, and policy-file parsers, converted once
  into the kernel's validated configuration; `scheduling::policy_config` is the YAML policy file
- `tracking_hash`: the tracking-hash key file read and the context built from the flat config
- Adapters behind features: HTTP servers, the ZMQ KV-event listener and registry, the replica
  mesh, the peer-recovery client, and Prometheus exposition

The data structures live in their owning crates: hashing in `dynamo_kv_hashing::routing`
(`compute_block_hash_for_seq`, `compute_seq_hash_for_block`, `BlockHashOptions`), the indexers in
`dynamo_kv_indexer` (`RadixTree`, `ConcurrentRadixTreeCompressed`, `ThreadPoolIndexer`,
`PositionalIndexer`), the wire vocabulary in `dynamo_router_protocol` (`WorkerId`) and
`dynamo_kv_events` (`RouterEvent`), and the kernel's `LocalScheduler` in `dynamo_scheduler`.

## Basic Rust Usage

```rust
use dynamo_kv_hashing::routing::{
    BlockHashOptions, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_indexer::radix_tree::RadixTree;
use dynamo_kv_router::config::KvRouterConfig;

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

For end-to-end routing, build a `SelectionService` with `SelectionServiceBuilder`; in-tree hosts
reach the kernel and stage types through each owner's `host` module.

## Writing a Plugin

A statically linked worker-selection policy or request classifier imports
`dynamo_kv_router::plugins::prelude::*` and depends on this crate alone; the prelude lists
every contract, signal view, wire type, configuration type, and registry type a plugin
names (`tests/plugin_api_prelude.rs` is a policy and a classifier written against it).

## Features

`default = []`: the flat configuration, the plugin registry, and the selection service compile
without any transport crate. Each adapter is a feature:

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
three plus `http`.

Other features:

- `metrics`: Prometheus metrics for router internals
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

## Removed paths (migration map)

`dynamo-kv-router` is the composition crate. Every path below was a shim over a stage, the
kernel, or a vocabulary crate; the shims were removed in this release. In-tree hosts import the
owner's `host` module; plugins import `plugins::prelude`.

| Removed path | Owner |
|---|---|
| root re-exports of stage and kernel items (`WorkerSelectionPolicy`, `LocalScheduler`, `KvSchedulerError`, `SelectionConfig`, ...) | the owner's `host` module, `plugins::prelude`, or `config::` |
| `protocols::*` | `dynamo_router_protocol`, `dynamo_kv_events` (`event`, `placement`, `residency`), `dynamo_kv_hashing::routing`, `dynamo_kv_hints`, `dynamo_scheduler::protocols` |
| `protocols::ActiveLoad` | `dynamo_llm::kv_router::ActiveLoad` (this crate cannot see `lib/llm`, which publishes and reads the message) |
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
| features `standalone-indexer`, `standalone-slot-tracker`, `standalone-selection`, `runtime-protocols` | the adapter features they expanded to (`http` + `ingest-zmq` + `peer-recovery`; `http` + `replica-sync-zmq`; all four); `runtime-protocols` was empty |

## Working rules

- Router configuration serialized inside a model deployment card is part of the N-2 worker/frontend wire contract described in `lib/llm/AGENTS.md`; keep compatibility handling narrow and give a deprecated wire field a versioned removal TODO.
- `validator` and `rand` carry a disproportionate transitive footprint for the small amount of functionality this crate needs; prefer local validation and `fastrand`, and do not reintroduce them without a measured need. Keep `axum` and `reqwest` default features disabled and enable only the features this crate uses.
- Use `FxHashMap` / `FxHashSet` for internal numeric keys and hot paths; use the standard hash collections for text keys or externally controlled values such as `request_id`.
- Engine hash invariant: within one indexer/hash domain, engine-published local block hashes and external sequence hashes are deterministic and consistent across workers (a unique local block chain maps to one unique external sequence-hash chain, and vice versa). Treat a violation as a producer, configuration, or protocol error; do not revalidate it by scanning per-worker external-hash ownership on request lookup or scheduling hot paths.

## Further Reading

- Router guide: <https://docs.nvidia.com/dynamo/components/router>
- Indexer internals:
  <https://github.com/ai-dynamo/dynamo/blob/main/lib/router/kv-indexer/src/README.md>
- Benchmarking the sharded KV indexer: [INDEXER_BENCH.md](../bench/kv_router/INDEXER_BENCH.md)
- Dynamo repository: <https://github.com/ai-dynamo/dynamo>
