# D=16 CRTC versus native/transposed CKF crossover experiment

This experimental branch compares the production `ConcurrentRadixTreeCompressed` (CRTC) with two production-shaped CKF frame consumers:

- `ckf-native`: authoritative per-DC Cuckoo filters queried independently.
- `ckf-transposed`: the same authoritative filters plus a derived bucket-major read table that shares probes across 16 DCs and falls back only for DCs whose generation changes.

The CKF filter, snapshot producer, and CKF1 wire behavior preserve the reference implementation from [Kaonael/dynamo PR #4](https://github.com/Kaonael/dynamo/pull/4), provided for [DEP #11225](https://github.com/ai-dynamo/dynamo/issues/11225). This branch is designed for evaluation and is not expected to merge as-is.

## Architecture and timing boundary

Reusable behavior lives in `lib/kv-router/src/indexer/cuckoo/`: filtering and probes, full/delta codec, Relay envelope/session validation, epochs, desynchronization, recovery, native/transposed lookup, and the eight-thread frame consumer. Experimental native and transposed variants are also available through the standalone indexer enum/factory.

The benchmark retains only Mooncake corpus construction, synthetic ballast, backend-specific input mapping, open-loop dispatch, metrics, and analysis.

One logical Mooncake worker represents one DC. CRTC receives the production `RouterEvent`. CKF never receives that event directly: offline Relay preparation mutates an authoritative `SnapshotProducer`, calls `publish()` once after the complete event, and retains `Unchanged`, one serialized delta, or one real full-snapshot chunk set. `Unchanged` emits no router input. The timed router consumer receives only real frame envelopes carrying expected DC and Relay-instance identity beside the unchanged CKF bytes.

Excluded from timing:

- Trace parsing, event generation, corpus I/O, and page-fault warm-up.
- Initial CRTC preload and CKF bootstrap assembly.
- Relay authoritative-state mutation, dirty-bucket discovery, and serialization.
- Correctness shadows and result parsing.

Included in timing:

- Deadline issue and fixed-concurrency query queues.
- CRTC raw-event enqueue/application and CKF frame enqueue/validation/decode/application.
- Epoch, checksum, seed, shape, length, bucket, and Relay-session checks.
- Native/transposed lookup, lock contention, transposed maintenance, and selective fallback.
- Query/update final drains.

The fixed corpus is replayed over several wall-clock windows. The corpus size never changes:

```text
offered block ops/s = fixed logical block operations / replay window
```

Shorter windows therefore increase offered load. This is not a sustained soak.

## Workload

- Mooncake FAST25 arXiv trace: `mooncake_trace.jsonl`.
- SHA-256: `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`.
- `trace_duplication_factor=20`.
- `trace_length_factor=4`.
- 16 workers/DCs, 8 event threads, and 16 query executors.
- 10,000,000 deterministic synthetic resident memberships per DC, in depth-128 prefix families.
- Capacity windows: `24000, 12000, 6000, 3000, 1500, 750` ms.
- Five fresh-process, blocked repetitions per cell.

The 160M synthetic memberships create resident-state and memory pressure. They are in a checked namespace disjoint from Mooncake request/event sequence hashes and are not presented as representative query traffic.

Download and verify the trace:

```bash
mkdir -p "$HOME/traces/mooncake-fast25"
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl \
  -o "$HOME/traces/mooncake-fast25/mooncake_trace.jsonl"
printf '%s  %s\n' \
  b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711 \
  "$HOME/traces/mooncake-fast25/mooncake_trace.jsonl" | sha256sum -c -
```

## Build and local smoke

Run from the repository root:

```bash
cargo test -p dynamo-kv-router cuckoo --no-default-features --lib
cargo test -p dynamo-kv-router cuckoo_factory \
  --no-default-features --features standalone-indexer --lib
cargo test -p dynamo-bench --bin ckf_crtc_crossover --no-default-features
cargo build --release -p dynamo-bench --bin ckf_crtc_crossover --no-default-features
```

D=2 end-to-end smoke:

```bash
BIN=target/release/ckf_crtc_crossover
SMOKE=/tmp/ckf-crtc-d2-smoke
mkdir -p "$SMOKE"
"$BIN" fixture --output "$SMOKE/corpus.msgpack" --git-sha "$(git rev-parse HEAD)"
CORPUS_SHA=$(sha256sum "$SMOKE/corpus.msgpack" | awk '{print $1}')
for backend in crtc ckf-native ckf-transposed; do
  "$BIN" run-cell \
    --corpus "$SMOKE/corpus.msgpack" \
    --expected-corpus-sha256 "$CORPUS_SHA" \
    --backend "$backend" \
    --replay-window-ms 200 \
    --repetition 1 \
    --phase smoke \
    --measured-code-sha "$(git rev-parse HEAD)" \
    --output "$SMOKE/$backend.json"
done
```

Local runs validate correctness only and are not performance evidence.

## Prepare the full corpus

Corpus preparation is untimed and should run on the same high-memory ComputeLab node used for the matrix:

```bash
BIN=target/release/ckf_crtc_crossover
RUN_ROOT=/path/to/verified/compute-node-visible/scratch/ckf-crtc-d16
TRACE="$HOME/traces/mooncake-fast25/mooncake_trace.jsonl"
mkdir -p "$RUN_ROOT/corpus"
"$BIN" prepare \
  --trace "$TRACE" \
  --output "$RUN_ROOT/corpus/mooncake-d16.msgpack" \
  --trace-sha256 b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711 \
  --trace-duplication-factor 20 \
  --trace-length-factor 4 \
  --dcs 16 \
  --event-threads 8 \
  --query-concurrency 16 \
  --num-gpu-blocks 16384 \
  --ballast-memberships-per-dc 10000000 \
  --ballast-depth 128 \
  --git-sha "$(git rev-parse HEAD)"
```

The command writes:

- `mooncake-d16.msgpack`: full prepared replay corpus.
- `mooncake-d16.resident.msgpack`: resident-state-only image for fresh memory processes.
- `mooncake-d16.manifest.json`: trace/corpus/filter metadata, checksums, and untimed Relay publisher build rates. Nondeterministic publisher timing stays in this sidecar so the versioned corpus bytes remain deterministic.

## ComputeLab CPU/NUMA binding

Use one exclusive high-memory CPU node for every cell. Discover the live allocation and use one logical sibling per physical core from the allocated cpuset:

```bash
PYTHON=.venv/bin/python
CPU_BINDING=$($PYTHON lib/bench/kv_router/ckf_crossover/select_physical_cpus.py --shell)
NUMA_NODES=$($PYTHON lib/bench/kv_router/ckf_crossover/select_physical_cpus.py \
  | $PYTHON -c 'import json,sys; print(json.load(sys.stdin)["numa_nodes"])')
export TOKIO_WORKER_THREADS=$(awk -F, '{print NF}' <<<"$CPU_BINDING")
export EVENT_THREADS=8
export QUERY_CONCURRENCY=16
export EXPECTED_CPU_BINDING="$CPU_BINDING"
export RUN_PREFIX="numactl --interleave=$NUMA_NODES taskset -c $CPU_BINDING"
$PYTHON lib/bench/kv_router/ckf_crossover/capture_hardware.py \
  --output "$RUN_ROOT/hardware.json" \
  --cpu-binding "$CPU_BINDING" \
  --numa-policy "interleave:$NUMA_NODES"
```

Record the actual hostname, CPU model, core/SMT topology, memory, NUMA topology, kernel, Rust version, allocator, governor, turbo state, binding, and Tokio/event-thread counts. Do not assume the prior node topology.

## Single cell and complete matrix

Single cell:

```bash
CORPUS="$RUN_ROOT/corpus/mooncake-d16.msgpack"
CORPUS_SHA=$(sha256sum "$CORPUS" | awk '{print $1}')
$RUN_PREFIX target/release/ckf_crtc_crossover run-cell \
  --corpus "$CORPUS" \
  --expected-corpus-sha256 "$CORPUS_SHA" \
  --backend ckf-transposed \
  --replay-window-ms 6000 \
  --repetition 1 \
  --phase capacity \
  --measured-code-sha "$(git rev-parse HEAD)" \
  --output "$RUN_ROOT/single.json"
```

Complete matrix:

```bash
CORPUS="$RUN_ROOT/corpus/mooncake-d16.msgpack"
CORPUS_SHA=$(sha256sum "$CORPUS" | awk '{print $1}')
PYTHON=.venv/bin/python \
RUN_PREFIX="$RUN_PREFIX" \
TOKIO_WORKER_THREADS="$TOKIO_WORKER_THREADS" \
EVENT_THREADS=8 \
QUERY_CONCURRENCY=16 \
HARDWARE_MANIFEST="$RUN_ROOT/hardware.json" \
lib/bench/kv_router/ckf_crossover/run_matrix.sh \
  target/release/ckf_crtc_crossover \
  "$CORPUS" \
  "$CORPUS_SHA" \
  "$(git rev-parse HEAD)" \
  "$RUN_ROOT/matrix"
```

The runner executes 90 capacity processes, derives `R_iso = 0.5 × min(CRTC ceiling, transposed ceiling)`, executes 15 common-load latency processes, and performs the one allowed half-rate retry if a headline backend fails the no-backlog criteria. It then runs four separate resident-state-only memory processes. All cells are sequential.

Expected counts:

```bash
test "$(find "$RUN_ROOT/matrix/trials" -name 'capacity_*.json' | wc -l)" -eq 90
test "$(find "$RUN_ROOT/matrix/trials" -name 'iso_rep*.json' | wc -l)" -eq 15
test "$(find "$RUN_ROOT/matrix/trials" -name 'memory_*.json' | wc -l)" -eq 4
sha256sum -c "$RUN_ROOT/matrix/raw_checksums.sha256"
```

Analysis can also be rerun independently with project Python:

```bash
.venv/bin/python lib/bench/kv_router/ckf_crossover/analyze.py \
  --stage final \
  --results-dir "$RUN_ROOT/matrix/trials" \
  --output-dir "$RUN_ROOT/matrix/aggregate" \
  --event-threads 8 \
  --query-concurrency 16
```

## Metrics and validity

Capacity keep-up requires total replay plus final drain within `1.10 × window`, a generator-valid issue span, and no pipeline/correctness error. Peak achieved throughput is selected from all six cells. CKF/CRTC claims use paired per-repetition log throughput ratios with two-sided Student-t 95% CIs.

Headline p50/p99 comes only from the common-load run. CRTC and transposed CKF must each achieve at least 99.9% of offered throughput, stop with at most one in-flight update per event thread and one query per query executor, drain both queues in at most 0.1% of the replay window, and have no generator, epoch, desync, insertion, or application error. Native CKF is an ablation and may be labelled diagnostic without lowering the common load.

The wide result table reports logical and ingress throughput, lookup and queue latency, full/delta Relay build and router apply rates, publication counts, transposed conflicts/fallbacks, final-state accuracy, pipeline errors, and separate resident-state memory modes.

Final-state CKF accuracy compares the optimized search with the untimed linear first-miss oracle after full drain. It reports inflation count/magnitude, under-reporting, full-map mismatch, and wrong best-DC selection. It does not compare asynchronous query answers with the newest producer state.

This is an internal index benchmark. AIPerf cannot inject production `RouterEvent`s and serialized Relay CKF frames into these in-process consumers, so the Rust harness is the explicitly accepted alternative.

## Authoritative results

Pending the ComputeLab run. Before opening the draft PR, replace this section with:

- Measured code SHA and result commit SHA.
- Corpus and hardware manifests.
- Ranked headline findings.
- The generated wide table from `matrix/aggregate/wide_results.md`.
- Links to committed compact per-trial/aggregate JSON and CSV.

Do not post these results to DEP #11225 or Slack without separate approval.
