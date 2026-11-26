# lib/router/slot-tracker

`dynamo-slot-tracker` is the slot tracker stage: it records the requests booked on each worker rank and reports their load and fullness. Hosts outside `lib/router` import through `host`.

`BusyThresholds` is in the request-classifier plugin API (`dynamo_kv_router::plugins::request_classifier`): a classifier reads one from `ClassifyRequest::reject_thresholds()`, and a worker-selection policy is handed none. Keep its fields private: a plugin calls only `is_configured()` and `has_room()`, and a host builds one and reads its limits through the sealed `host::BusyThresholdsHostExt`. Do not add a limit accessor to the type or re-export the trait from a plugin module.

`record_full_worker_ranks_at` fills a caller-owned `FullRankCounts` from the derived
prompt registry at one instant. It holds one read guard and creates no projection
map or per-request ownership state. The scheduler uses it when exclusive-affinity
costing or queued readiness needs current host-reject fullness.

## Performance

PR numbers point to worked examples; open one only when you need to see how a
rule was applied.

- **Changing `project_worker_loads_into`.** It visits every rank on every
  admission. Don't add a lock, another lookup by id, or an allocation per rank, and
  keep writing into the caller's storage instead of returning a new map
  (#10935, #10967).
- **Adding a consumer of load.** If it needs a few ranks or can stop early, read
  through `worker_load_reader` instead of projecting every rank. Its values leave
  out the request's own blocks, and it holds the load table's read lock, so keep it
  short-lived (#17480).
- **Adding per-block state to the block tracker or the prompt trie.** It is
  multiplied by blocks times active requests. Both store compressed edges, so put
  the state on the edge's node; don't add an `Arc`, a map entry, or an atomic per
  block (#11508, #11515, #11644).
- **Adding an update on the response-stream path.** Apply it once per chunk with a
  count, as `add_output_blocks_if_booking` takes `num_blocks`, not once per block
  (#15448).
- **Proposing a redesign.** Report projection p99 and p999 from
  `active_sequences_bench` (`lib/bench/kv_router/INDEXER_BENCH.md`), not only
  throughput. A unified prompt tracker and a trie backed by the indexer's
  concurrent tree were both dropped on latency (#11641, #10938).
