# Unified-backend bridge microbenchmark

Isolates the cost the unified backend's Rust↔Python bridge adds — especially
**per-token GIL contention** — without standing up a cluster. Use it as a
regression guardrail when touching `PyLLMEngine` / `EngineAdapter`.

## What it measures

Both engines run through the *same* production path
(`EngineAdapter` → `LLMEngine::generate`), driven in-process with mock request
contexts (no NATS / etcd / frontend / HTTP):

| Engine | Path | Role |
|--------|------|------|
| `sample_engine.py` | `PyLLMEngine` bridge: `spawn_blocking` + `Python::with_gil` + `depythonize` **per token** | the bridge under test |
| `BenchFloorEngine` (Rust) | native `LLMEngine`, no Python in `generate` | GIL-free floor |

Everything except the bridge cancels out of the delta, so:

```
overhead% = (floor_tok_s - unified_tok_s) / floor_tok_s * 100
```

is the bridge + GIL tax. Sweeping **concurrency** is the key axis: the
per-token `spawn_blocking` hop serializes on the GIL across in-flight requests,
so `overhead%` climbs and the unified throughput stays flat as concurrency
rises — the GIL-contention signature.

## Build

The harness entries are gated behind the `bench-harness` cargo feature and are
**not** in the default wheel:

```bash
cd lib/bindings/python
maturin develop --features bench-harness
```

## Run

```bash
# fast smoke run (1 workload, 64 requests/point)
python -m benchmarks.unified_backend.bench_bridge --quick

# full sweep, save JSON to diff against a later run
python -m benchmarks.unified_backend.bench_bridge --json results.json

# custom concurrency / load
python -m benchmarks.unified_backend.bench_bridge --concurrency 1 8 32 --total-requests 1000
```

Output columns: `floor t/s`, `unified t/s`, `overhead%`, and the unified path's
TTFT p50, ITL p50/p99 (ms).

## Files

- `workload.py` — request shapes + concurrency sweep.
- `bench_bridge.py` — floor-vs-unified `generate()` guardrail.

The Rust side lives in:
- `lib/backend-common/src/testing.rs` (`bench` module: `run_load`,
  `BenchWorkload`, `BenchStats`, `BenchFloorEngine`) — gated behind the
  `testing` feature.
- `lib/bindings/python/rust/bench.rs` (`bench_unified_python_engine`,
  `bench_unified_rust_floor`) — gated behind `bench-harness`.

## Not covered

- **lifecycle** (`start` / `drain` / `cleanup`): one-shot crossings, untimed.
- **git baseline guardrail**: a compare script over two `--json` outputs.
- Legacy-vs-unified, KV-event publish contention, and component-metrics
  (`SnapshotPublisher`) contention were investigated and found settled (parity /
  ~10–20% but tunable via per-call block batching / negligible, respectively);
  those harness paths were removed to keep this lean.
