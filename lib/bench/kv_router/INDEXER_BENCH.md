# KV Router Indexer Benchmark

## Mooncake replay

`mooncake_bench` uses deadline-driven replay for the production thread-pool
indexers. Queries and writes are issued without waiting for earlier operations to
complete, and throughput ends only after every query and event worker has
drained. There is no separate replay mode.

Supported backends are:

- `nested-map` (`PositionalIndexer`)
- `concurrent-radix-tree-compressed`

The benchmark rejects single-threaded radix, branch-sharded CRTC, approximate
writes, extra read stress, and in-process repetitions. Run each repetition in a
fresh process.

### Preparation and timing

Each worker timeline is normalized so its first entry starts at zero, rescaled
to the requested replay window, and stable-sorted by deadline, operation class,
worker ID, and source order. Queries precede writes at equal deadlines. That
ordering applies to queue acceptance; query execution may still race event
application.

Preparation consumes and releases the parsed trace, generated replay artifacts,
worker timelines, and intermediate payload owners. The timed process retains
only the prepared schedule, one flattened query-hash slab, and backend state.
On Linux, the benchmark returns free preparation pages with `malloc_trim` and
waits a fixed five seconds before constructing backend workers. It then
page-touches the schedule once and performs the fixed lookup warm-up before
timing. This explicit quiescence prevents the parallel event-generation phase
and released allocator arenas from affecting the timed backend drain; JSON
records it as `pre_run_quiescence_ms=5000`.

Outside timing:

- Trace parsing, event generation, normalization, scaling, and stable merge.
- Fixed query queues, completion buffers, observation writers, and barriers.
- Backend construction, page touching, and fixed lookup warm-up.
- Completion harvesting, ID joins, sorting, percentiles, JSON, and file I/O.

Inside timing:

- Absolute-deadline query publication.
- Owned `RouterEvent` submission through the production `ThreadPoolIndexer`
  worker lookup and Flume queues.
- Lookup service, event application, contention, required clock reads, and
  fixed-slot completion records.
- Queue drain, ending at the last query or event completion. Closing lanes,
  sealing and harvesting completion buffers, and aggregating issue records
  happen after that timestamp.

The logical denominator is Request + Stored + Removed + Cleared. The block
denominator includes request hashes and hashes in both Stored and Removed
events. Cleared contributes one logical operation and zero block hashes.

Issue lag is diagnostic. A trial is generator-valid when all operations are
issued within `1.01 × replay_window`, IDs and per-worker FIFO order are exact,
fixed buffers do not overflow, all operations succeed, and the timed drain
completes. `kept_up` additionally requires total replay plus drain to finish
within `1.10 × replay_window`.

### Trace and build

```bash
ROOT=$(git rev-parse --show-toplevel)
TRACE="$ROOT/lib/kv-router/traces/mooncake_trace.jsonl"

mkdir -p "$(dirname "$TRACE")"
curl -L \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl \
  -o "$TRACE"
echo 'b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711  '"$TRACE" \
  | shasum -a 256 -c -

cargo bench --package dynamo-bench --bench mooncake_bench \
  --no-default-features --features mooncake --no-run
BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 \
  -name 'mooncake_bench-*' | head -n1)
```

`--trace-block-size` (default 512) is the number of tokens each trace `hash_id`
represents and must match the trace; the public Mooncake traces use 512.
`--block-size` (default 128) is the separate mock-engine and indexer block size;
the engine re-chunks synthesized prompt tokens at that size.

Linux uses absolute `CLOCK_MONOTONIC` sleeps followed by the configured spin.
macOS uses a portable sleep-plus-spin timer for correctness tests only.

### Single CRTC trial

Select one logical CPU per physical core from the process's actual affinity
mask. Keep event issuers and the timed query issuer disjoint from the backend
mask. The parent coordinator shares the query-issuer CPU but remains parked in
scoped-thread joins while issuance runs.

The following command reproduces the measured 128-core binding. Adapt the masks
to the host rather than assuming these CPU numbers are portable.

```bash
RESULTS=/path/to/mooncake-results
mkdir -p "$RESULTS"

numactl --interleave=all "$BIN" "$TRACE" \
  --query-lanes 128 \
  --issuer-cpus 0-3,65-68 \
  --query-issuer-cpu 64 \
  --backend-cpus 4-63,69-127 \
  --issuer-spin-us 100 \
  --issue-lag-diagnostic-threshold-us 250 \
  --num-unique-inference-workers 128 \
  --trace-duplication-factor 20 \
  --trace-length-factor 4 \
  --benchmark-duration-ms 750 \
  --result-json-output "$RESULTS/crtc-01.json" \
  concurrent-radix-tree-compressed --num-event-workers 8
```

For 20 fresh-process repetitions:

```bash
for run in $(seq -w 1 20); do
  numactl --interleave=all "$BIN" "$TRACE" \
    --query-lanes 128 \
    --issuer-cpus 0-3,65-68 \
    --query-issuer-cpu 64 \
    --backend-cpus 4-63,69-127 \
    --issuer-spin-us 100 \
    --issue-lag-diagnostic-threshold-us 250 \
    --num-unique-inference-workers 128 \
    --trace-duplication-factor 20 \
    --trace-length-factor 4 \
    --benchmark-duration-ms 750 \
    --result-json-output "$RESULTS/crtc-$run.json" \
    concurrent-radix-tree-compressed --num-event-workers 8
done

test "$(find "$RESULTS" -maxdepth 1 -name 'crtc-*.json' | wc -l | tr -d ' ')" = 20
```

The project-root Python environment may be used for post-processing:

```bash
.venv/bin/python - <<'PY' "$RESULTS"
import json
import pathlib
import statistics
import sys

paths = sorted(pathlib.Path(sys.argv[1]).glob("crtc-*.json"))
runs = [json.loads(path.read_text()) for path in paths]
assert len(runs) == 20
assert all(run["generator_valid"] for run in runs)
values = [run["achieved_block_ops_per_sec"] for run in runs]
print(f"runs={len(values)} mean={statistics.fmean(values):.3f} median={statistics.median(values):.3f}")
PY
```

### Sweep

Sweep cells use the same completion-correct implementation. Each cell regenerates
and consumes a fresh corpus and constructs fresh backend state.

```bash
cargo bench --package dynamo-bench --bench mooncake_bench \
  --no-default-features --features mooncake -- \
  "$TRACE" \
  --sweep --sweep-min-ms 750 --sweep-max-ms 24000 --sweep-steps 6 \
  --result-json-output /path/to/sweep.json \
  concurrent-radix-tree-compressed --num-event-workers 8
```

### Result fields

The versioned Mooncake JSON reports:

- Corrected logical and block totals, offered rate, actual issue rate, and
  completion/drain-inclusive achieved rate.
- Query issue lag, queue wait, lookup service, and scheduled-to-completion
  distributions.
- Update issue lag, accepted-to-finished, and scheduled-to-finished
  distributions.
- Queue depth at producer stop, outstanding work, maximum reconstructed depth,
  drain time, timer kind, CPU masks, exact-ID validity, and compact failure
  reasons.
- `provenance`: the command line, binary path and SHA-256, trace path and
  SHA-256, and the trace, corpus, and issuer settings that shaped the replay.

Do not interpret overloaded lookup latency as an iso-throughput latency result.
At a comfortable common load, report scheduler lag, queue wait, and lookup
service separately and require negligible queue depth and drain.

### Choosing a load for A/B comparisons

An overloaded replay and a keep-up replay measure different things, so choose
the load by the path a change touches:

- **Write capacity: overloaded replay** (for example the 750 ms command above).
  Queries finish during the issue window while events drain for several times
  longer, so `achieved_block_ops_per_sec` mostly measures event application.
  During that window queries also walk a tree that lags far behind the trace.
- **Read cost and write latency: keep-up replay.** Use the shortest duration
  whose runs report `kept_up=true` with negligible drain; the sweep finds it.
  Compare `query_service` and `update_accepted_to_finished` percentiles.
  Achieved throughput is pinned to the offered rate there and is not a metric.

A sensitivity check with two deliberately regressed builds shows the difference.
Both were throwaway patches to the CRTC at `2de120f6a3`. The read variant ran
`walk_match_path` twice in `find_matches_impl` and kept the second result, and
the write variant busy-waited 6 µs at the start of `apply_event`. Each variant
ran the documented CRTC command at 750 ms and at 12 s, with 128 workers,
duplication 20, length 4, and 4 event workers, as five interleaved
fresh-process pairs against an unpatched build on 8 cores. Values are
candidate/control median ratios:

| Metric | Load | Read path ×2 | Write +6 µs/event |
|---|---|---:|---:|
| `achieved_block_ops_per_sec` | 750 ms (overloaded) | 0.997 (not detected) | 0.728 |
| `query_service` p50 | 750 ms (overloaded) | 1.086 | 0.952 |
| `query_service` p50 | 12 s (keep-up) | 1.325 | 1.007 |
| `update_accepted_to_finished` p50 | 12 s (keep-up) | 1.004 | 5.09 |

Median lookup service was 1.66 µs in the overloaded replay and 3.10 µs at keep-up:
the overloaded replay understates read cost because its queries run against an
early, small tree.

## Active Sequences replay

`active_sequences_bench` uses the same deadline-driven measurement principles
for `ActiveSequencesMultiWorker`, but it does not route operations through
`ThreadPoolIndexer` or `RouterEvent`. Each lifecycle method is synchronous, so
method return is the completion boundary.

Each mocker-replayed request contributes an Add at its engine arrival time,
which is when a router books it, a PrefillComplete at its first output, and a
Free at completion. Requests are therefore in prefill for their simulated
queueing and prefill time. `--modeled-prefill-tokens-per-sec RATE` attaches an
expected prefill duration of `ISL / RATE` to every Add, which exercises the
modeled (AIC-style) prefill-load path; without it, prefill load is unmodeled.

Preparation creates a data-only corpus with normalized per-worker deadlines,
stable operation IDs, lifecycle metadata, and one flattened sequence-hash slab.
It validates every Add → PrefillComplete → Free lifecycle before building the
execution schedule. The schedule assigns each logical worker to one persistent
operation lane, preallocates exact queue and completion capacity, materializes
the owned `SequenceRequest` values outside timing, and releases the source trace
and intermediate corpus storage. The complete schedule and payload storage are
page-touched once before the clock starts.

At a shared deadline, operations retain per-worker source order. In particular,
an Add is accepted and completed on its sticky lane before an associated
PrefillComplete or Free. The timed path queues compact operation IDs, performs
no per-operation task or timer creation, and drains every lane before throughput
ends. Projection reuses one map per runtime thread, matching the production
scheduler queue's single reused map; its digest is computed after the
projection timestamp so it does not count as projection service time.

The result schema reports:

- Add, PrefillComplete, and Free logical counts.
- Input blocks, projection visits, add/registration visits, free/release visits,
  and their total logical block visits.
- Scheduled-to-accepted issue lag, accepted-to-started queue wait,
  scheduled-to-completed latency, queue depth, outstanding work, and drain.
- Separate service distributions (mean and percentiles) for projection, add,
  composite project+add, prefill completion, and free, plus
  `tracker_service_ns_per_op`: the summed method service time per logical
  operation.
- Process and issuer-thread CPU over the timed window and
  `backend_cpu_ns_per_op` (process CPU excluding the issuer).
- Worker projections produced and inspected, exact IDs, per-worker FIFO,
  final-empty state, keep-up state, and compact failure reasons.

Logical block visits describe benchmark-level sequence traversals. They are not
CPU instructions or exact internal membership mutations, and input blocks alone
are not reported as total block operations. Issue lag remains diagnostic. A run
is generator-valid only when issuance completes within `1.01 × replay_window`,
IDs and FIFO order are exact, fixed storage does not overflow, all methods
succeed, every lane drains, and the final sequence state is empty.

Build and run one cell from the repository root:

```bash
cargo bench --package dynamo-bench --bench active_sequences_bench \
  --no-default-features --features active-sequences --no-run

cargo bench --package dynamo-bench --bench active_sequences_bench \
  --no-default-features --features active-sequences -- \
  "$TRACE" \
  --operation-lanes 128 \
  --issuer-spin-us 75 \
  --issue-lag-diagnostic-threshold-us 250 \
  --num-unique-inference-workers 128 \
  --benchmark-duration-ms 4000 \
  --result-json-output /path/to/active-sequences.json
```

On Linux, keep the deadline issuer off the measured cores. `--backend-cpus`
pins the Tokio lane runtime and sizes its worker pool to the mask, and
`--issuer-cpu` pins the issuer to a disjoint CPU. On a hybrid CPU, give the
backend one core type, for example `--backend-cpus 0-7 --issuer-cpu 16`.
`--trials N` prepares the corpus once and replays it N times against fresh
tracker state, writing one result per trial; use it to profile the timed phase
without trace-generation setup in the samples, and use separate processes for
A/B trials.

Sweep cells rebuild the prepared corpus and Active Sequences state independently:

```bash
cargo bench --package dynamo-bench --bench active_sequences_bench \
  --no-default-features --features active-sequences -- \
  "$TRACE" \
  --operation-lanes 128 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 16000 --sweep-steps 5 \
  --result-json-output /path/to/active-sequences-sweep.json
```

The macOS timer is a correctness and profiling fallback. Do not publish local
workstation throughput as an authoritative cross-system result.

## Approximate retention replay

`approximate_lru_bench` drives the production
`ThreadPoolIndexer<ConcurrentRadixTreeCompressed>` directly. It supports a TTL
arm and a capacity-bounded LRU arm in separate processes. Each request performs
a prompt lookup and predicted insertion/acquire at its raw admission timestamp,
then releases its LRU lease at the terminal output signal. Lanes are sticky by
request attempt so same-rank lookups can still execute concurrently; the
indexer's production worker/rank assignment preserves mutation FIFO.

This is deliberately a prompt-lifecycle benchmark. It accounts for partial
prompt tails as private LRU occupancy, but does not hash or materialize streamed
output blocks. The result schema identifies that scope explicitly. Capacity is
registered and acknowledged before timing, every LRU acquire must resolve to
LRU rather than TTL fallback or ignored, and the final stats fence verifies that
leases, active references, private occupancy, overcapacity, and queue backlog
have drained.

The default timed path is lean. It records only start/end wall and process CPU,
issuer CPU, issue span, drain, queue depth at producer stop, acquire-mode counts,
and final indexer state. It does not read clocks or maintain latency samples per
operation. Use `perf` or Samply for CPU attribution; production mutation-queue
statistics remain available in the final LRU snapshot.

Build and run TTL and LRU in fresh processes:

```bash
cargo bench --package dynamo-bench --bench approximate_lru_bench \
  --no-default-features --features approximate-lru --no-run

cargo bench --package dynamo-bench --bench approximate_lru_bench \
  --no-default-features --features approximate-lru -- \
  testdata/mooncake_trace_approximate_pressure.jsonl \
  --policy ttl --trace-block-size 8 --block-size 4 --benchmark-duration-ms 1000 \
  --num-unique-inference-workers 1 --pre-run-quiescence-ms 0 \
  --result-json-output /tmp/approx-ttl-smoke.json

cargo bench --package dynamo-bench --bench approximate_lru_bench \
  --no-default-features --features approximate-lru -- \
  testdata/mooncake_trace_approximate_pressure.jsonl \
  --policy lru --trace-block-size 8 --block-size 4 --capacity-blocks 2 --require-eviction \
  --benchmark-duration-ms 1000 --num-unique-inference-workers 1 \
  --pre-run-quiescence-ms 0 \
  --result-json-output /tmp/approx-lru-smoke.json
```

The pressure arm exits nonzero if its prepared unique working set does not
exceed capacity, any acquire falls back, eviction does not occur, or final
invariants fail. macOS runs are functional smoke evidence only; authoritative
CPU and throughput comparisons require pinned, exclusive Linux CPUs.

## Validation evidence

The measurements below used one fixed host and one build:

| Item | Value |
|---|---|
| CPU | AMD EPYC 7742, 2 sockets × 64 physical cores, SMT disabled |
| Memory | 1 TiB, interleaved across NUMA nodes |
| Kernel | Linux 6.17.0-23-generic |
| Rust | 1.93.1 |
| CPU policy | performance governor, boost enabled |
| Query lanes / event workers | 128 / 8 |
| Event issuers | 8, each owning a contiguous 16-worker group |
| Pre-run quiescence | Linux heap trim followed by a fixed 5,000 ms wait |
| Corpus | 472,160 requests + 2,163,500 events; 75,554,938 corrected block operations |

Temporary validation machinery was removed before the final source state. The
fully instrumented no-op pipeline passed all five 1× issue-span and correctness
runs with a 100 µs spin. At 2×, only 2/5 runs met the `1.01 ×` issue-span target;
that headroom check is diagnostic, not a trial-validity gate. Allocation auditing
found zero benchmark-owned allocations or reallocations in the issuer, query
transport, sample storage, and completion-write regions. Production Flume queue
growth and backend allocations remain attributed production work.

The temporary A/B/C reproduction used 20 fresh processes per variant at the
same 750 ms offered workload. A and B existed only for decomposition and were
removed afterward.

| Variant | Completion semantics | Mean corrected block ops/s (95% CI) | Keep-up evidence |
|---|---|---:|---|
| A | Historical closed loop; final update drain omitted | 93.257M (92.723–93.790M) | 18/20 appeared to keep up only because drain was omitted |
| B | Closed loop; completion-correct timed drain | 8.651M (8.619–8.682M) | 0/20; mean drain 7.922s |
| C | Deadline-driven; completion-correct timed drain | 9.167M (9.104–9.230M) | 0/20; mean drain 7.461s |
| Final C | Cleaned normal replay; completion-correct timed drain | 9.081M (9.044–9.118M) | 20/20 generator-valid; 0/20 keep-up; mean drain 7.534s |

The historical Stored-only A denominator was 68.133M block ops/s
(67.743–68.523M). It omitted Removed hashes and must not be compared with the
corrected denominator. Paired C/B completed-throughput ratio was 1.0596
(95% CI 1.0518–1.0675).

The 750 ms load is deliberately overloaded: it offers 100.740M corrected block
ops/s while CRTC completes about 9.08M block ops/s after drain. It is a
measurement-semantics decomposition, not a capacity ceiling. Temporary C's
lookup service p50/p99 was approximately 4.04/7.09 µs, while query queue-wait
p50/p99 was approximately 71.6/203.5 µs; these are overload diagnostics, not
headline latency.

Final C's mean lookup service p50/p99 was 4.21/7.31 µs, query queue-wait
p50/p99 was 76.0/189.6 µs, and scheduled-to-completion p50/p99 was
99.6/246.9 µs. All are overload diagnostics. Final C was 0.94% below temporary
C, satisfying the preregistered 2% mean-agreement gate; their 95% CIs overlap.

The measured base Git SHA was
`337c22c09f14dc08e92092706506453da3c353c4`. The exact unstaged source archive
SHA-256 was `c3a8493181f69fc614f1b04e815e61b6558a727bee335a6b908e92f18b4d4b88`,
the optimized binary SHA-256 was
`7c0a758cd1cfb65f676cd4550ab7780444958cf794690a4e8664c98daa031e4c`, and the
frozen final result archive SHA-256 was
`9d5dc04e03033db31a0560d239b9af62011f6342cba32fecd7410b1e738fa948`.
Raw JSON, logs, allocation-audit output, and profiles remain external artifacts
rather than repository content.
