# Selection Replay Benchmark

`selection_replay_bench` drives the KV router's selection core in-process with a
deployment-shaped workload. It is the benchmark for changes to `SelectionCore`,
the scheduler queue actor, and the default worker-selection policy.

It runs the real routing code under the state a large deployment produces:
thousands of requests in flight, long agentic prompts, KV events arriving while
requests are admitted, and prefill completions and releases on the engine's
timing. It starts no frontend, workers, or transport, so a run needs one machine
and a few minutes. It measures what the selection path costs; judge placement
quality (hit rates, TTFT) with an end-to-end run, because request timing here
does not react to where a request lands.

## What runs

The selection service is built as the frontend builds its embedded selection
(`lib/llm/src/kv_router/embedded.rs`): the worker-selection policy the router
config names (the builtin default unless it names another policy Dynamo ships),
host-managed request lifecycle, lazy replica registration, and the concurrent
indexer. Each
request follows the frontend's calls: `run_selection` with a `Lease` and the
prompt's token IDs, `commit`, the worker-origin and frontend prefill-complete
marks at its first output, and the acknowledged release at completion. KV events
reach the index through `Indexer::try_apply_event`, the frontend's ingestion call.
The bench binary allocates with jemalloc under the options the frontend builds
in; `_RJEM_MALLOC_CONF` overrides them.

Event-plane transport, HTTP, tokenization, response streaming, the Prometheus
load gauges, the lease manager, and peer frontends are not modeled, so the
router's runtime sees less interference than in a frontend.

## Workload

The corpus holds only request timing. A lane is one closed-loop slot, like one
unit of aiperf concurrency: it replays a play (conversation) of the trace one
request at a time, and starts another play when one ends. Every copy of a play
gets disjoint hash IDs, as aiperf's cache-bust prefixes do. A play's requests run
in the order they started, so a Weka play's subagents run in series with it. The
mocker simulates the lanes, pinned round-robin to engine ranks, for each request's
arrival, first output, and completion. Generation fails if a lane runs out of
plays inside the window, because the load would then fall below `--lanes`.

KV state comes from the replay's own placements. Engine issuer threads play the
workers: each worker has a cache of `--num-gpu-blocks` blocks that follows a vLLM
block pool, simplified. At a request's first output its booked worker pins the
cached prefix and stores the rest; one block per `block_size` output tokens is
stored during decode; at completion the blocks become evictable, tail first, and
a full cache evicts the least recently released. Each change reaches the index as
the matching `Stored` or `Removed` event, in order, so a session migrates exactly
when the router moves it. The cache tracks each run of a prompt's blocks inside
one trace block as one entry: hits stay exact, but eviction takes up to a trace
block's worth of blocks at once. It is one device tier; host (G2) or remote
(KVCR) tiers would be further caches.

### `agentx` workload (the defaults)

| Setting | Value |
|---|---|
| Trace | `semianalysisai/cc-traces-weka-with-subagents-060526`, Weka format |
| Lanes | 6144, the AgentX frontend benchmark's concurrency; trace delays ignored |
| Workers | 2048, one DP rank each, block size 16 |
| Engine | `max_num_seqs` 1024, `max_num_batched_tokens` 16384 (the DeepSeek-V4.1 vLLM agentic recipe); 65536 blocks (~1M tokens) per rank, so caches evict; speedup 10 |
| Router | `KvRouterConfig::default()` with replica sync on |
| Window | 30 s warmup, then 90 s measured, in trace time |

Every workload value is a flag (`--help`). With `--ignore-trace-delays false`, a
Weka turn waits the gap between its recorded start and the previous turn's.

`--router-config-json` sets `KvRouterConfig` fields over these defaults; for
example, a heavily queued router:
`--router-config-json '{"router_queue_threshold": 0.1}'`. The router config is
not part of the corpus, so one saved corpus serves every router config. A
`router_policy_config` file is applied as the frontend applies it, including the
worker-selection policy it names. The replay rejects the options it does not
model: `use_kv_events: false`, a single `router_event_threads`,
`router_track_output_blocks`, `router_predicted_ttl_secs`, and
`enable_session_prefix_index`.

```bash
mkdir -p traces/agentx && cd traces/agentx
curl -fLO https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-060526/resolve/8a5df339ba3c31a48dd79a2f5ceb81e08679266b/traces.jsonl
sha256sum traces.jsonl  # 954157040cbae47fef3bec70532bdb3406c75ed894e207e502461d70fa06639c
```

Mooncake JSONL traces also work (`--trace-format mooncake --trace-block-size N`).

## Modes

**Fixed** (`--mode fixed`): wall time is trace time divided by `--time-scale`, or
the scale that offers `--target-rate` window requests per second. Admission
latency runs from each request's deadline to its booking. The run keeps up when,
over the window, every request is booked or rejected, requests issue within 50 µs
and engine actions within 10 ms (p99), admission p99 stays within 20 ms, and the
last answer lands within 1.1x of the window.

**Saturation** (`--mode saturation`): trace time advances only as fast as the
router admits, with at most `--admission-window` admissions in flight, so the
in-flight load stays the trace's. Engine actions may trail a request's arrival by
up to 10 ms of trace time, as a real router hears of them late. If a queueing
router holds every admission slot waiting on an engine action, trace time moves
on to the next one (`stall_advances`). The result is booked requests per second,
from the first window request issued to the last one answered, booked or
rejected. `harness_wait_share` is the share of the window the bench's own
threads, not the router, held back; above 5% the run is invalid.

In both modes, an index more than 20,000 events behind during the window means
the router routed on a stale index because it could not take the load: the run
reports `kept_up=false` ("the index fell N events behind"), and
`mean_cached_blocks_error` shows how far its predictions drifted. So does a
replica channel so full that it drops workers' prefill-complete marks
(`dropped_worker_marks`), which the frontend's ingress drops and counts too.

## Running

Build without the default features, which turn on kv-router's `bench` feature and
change the code under test:

```bash
cargo bench -p dynamo-bench --no-default-features --features selection-replay \
  --bench selection_replay_bench -- \
  --trace traces/agentx/traces.jsonl --save-corpus agentx.corpus \
  --backend-cpus ROUTER_CPUS --issuer-cpus BENCH_CPUS --mode saturation
```

- Use an exclusive machine, with both CPU lists on one socket and
  `numactl --membind` to it. `--issuer-cpus` needs one CPU per bench thread: the
  request issuer, the sampler, `--engine-issuers`, and `--prep-threads`. If
  `harness_wait_share` flags a run, raise those counts.
- `--perf-ctl-fifo` and `--perf-ack-fifo` bracket the measured window for
  `perf record --control fifo:CTL,ACK`; divide by `admitted` for per-admission costs.

## A/B protocol

Results depend on the machine, so compare arms on one machine; there are no
reference numbers.

1. Generate the corpus once with `--save-corpus` and replay it in both arms with
   `--load-corpus`. Equal `corpus_digest` values confirm the same workload.
2. Build base and branch with separate `CARGO_TARGET_DIR`s.
3. Alternate base and branch processes for at least three rounds on the same CPUs,
   and compare medians.
4. Report saturation booked requests per second, admission p50/p99 at a common
   fixed rate, and, for CPU changes, actor and caller cycles per admission from a
   window-bracketed profile.
