# Mocker AIC callback benchmark

These benchmarks demonstrate the benefit the AIC Rust core (aiconfigurator #1200)
delivers to its consumer, the Dynamo mocker: they compare the mocker driving the
Rust crate (`RustAicCallback`, wrapping `aiconfigurator_core::AicEngine`) against
the pre-#1200 pure-Python AIC. The headline numbers:

- **~1.2x** faster offline replay (end-to-end). Offline replay is a
  single-threaded discrete-event simulation — `num_workers` are logical, not OS
  threads — so predict calls never run concurrently and this is the full offline
  gain.
- **predict throughput that scales with physical cores under concurrency** — the
  advantage is bounded by core count, not a fixed factor. Pure-Python AIC is
  GIL-capped (throughput stays ~constant no matter how many threads); the Rust
  crate scales to the core count. On a 12-core box that is ~9x; on a many-core
  server it is larger. This applies to the **live/online mocker** (concurrent
  serving backend), NOT to single-threaded offline replay.

`bench_aic_rust_callback.py` covered the single-thread end-to-end case
(HISTORICAL: it relied on the Python fallback to A/B inside the mocker; that
fallback was since removed, so the mocker is Rust-only and this script no longer
produces a valid A/B — its recorded numbers below still stand).
`bench_aic_concurrency.py` covers the concurrency case and does NOT depend on
the removed fallback (it drives the Python `AicSession` directly), so it remains
the live comparison tool. A third "compiled-engine
Python" path is included as a reference point to attribute where the gain comes
from, but the number that matters for the mocker is **Rust crate vs pure Python**.

## What it compares

Three latency-prediction paths, all driven through the same mocker replay:

- **`rust`** — `RustAicCallback`, the pure-Rust crate called directly from the
  mocker scheduler (no GIL, no pyo3). Default.
- **`python_opwalk`** — the **pre-#1200 baseline**: `PyAicCallback` →
  `AicSession` pure-Python op-walk (loops over the model's ops in Python),
  reached from Rust via pyo3 on every call. "AIC with no Rust core."
- **`python_compiled`** — intermediate: `PyAicCallback` → `AicSession` on the
  compiled Rust engine (same math as `rust`, but with the pyo3 + GIL boundary
  per call). This is what #1200 already gives the Python path.

Two ratios fall out:

- **BIG WIN = `python_opwalk` / `rust`** — the whole Rust migration, end to end.
- **this-PR = `python_compiled` / `rust`** — only the pyo3 + GIL boundary this PR
  removes; the rest of BIG WIN was already delivered by #1200.

## How

Each mode runs in a separate subprocess (engine + callback are cached in
process-global statics; the toggles are read once at session/callback build):

- `rust`            — env unset                                → `RustAicCallback`
- `python_compiled` — `DYNAMO_AIC_DISABLE_RUST_CALLBACK=1`      → `PyAicCallback` + Rust engine
- `python_opwalk`   — `+ DYNAMO_AIC_DISABLE_COMPILED_ENGINE=1`  → `PyAicCallback` + Python op-walk

A discarded **warmup** replay amortises the one-time engine build
(`build_aic_engine` → Python `compile_engine` + Rust parquet load) so the timed
runs isolate steady-state replay. Equivalence of replay metrics is asserted first
(rel ≤ 1e-6); a divergence makes the speedup meaningless and fails the run.

```bash
uv pip install '.[mocker]'           # plus a build with --features aic-forward-pass
python benchmarks/mocker/bench_aic_rust_callback.py --warmup 1 --repeat 5
```

## Results (Qwen3-32B / h200_sxm / vllm 0.19.0, Apple M-series, release build)

Steady-state (min) replay seconds:

| workload (isl/osl/reqs/conc/workers) | rust | opwalk (pure-py) | compiled | BIG WIN | this-PR |
|--------------------------------------|-----:|-----------------:|---------:|--------:|--------:|
| decode_heavy_1w  1024/1024/256/64/1  | 0.076 | 0.089 | 0.082 | 1.17x | 1.07x |
| balanced_1w      2048/256/256/64/1   | 0.021 | 0.026 | 0.023 | 1.23x | 1.11x |
| decode_heavy_4w  1024/1024/512/128/4 | 0.239 | 0.289 | 0.257 | 1.21x | 1.07x |

**Takeaways**

- The mocker driving the Rust crate is **~1.2x** faster than the pre-#1200
  pure-Python AIC, single-thread end-to-end (`opwalk` → `rust`).
- It is only ~1.2x (not orders of magnitude) because single-thread the
  pre-#1200 path was not actually slow — its `op.query` is vectorised
  numpy/pandas table lookup + interpolation, so one Python `predict` is only
  ~1.2x slower than Rust (per-call micro-bench: op-walk/engine ≈ 1.2x). The large
  win is under concurrency — see below.
- `rust` vs `compiled` predictions are **bit-identical** (rel = 0) — same Rust
  engine; the `compiled` column is only there to attribute the gain.
- The one-time engine build dominates a cold run (~1.5 s) but amortises to zero
  in a long-running mocker, hence the warmup.

## Concurrency (`bench_aic_concurrency.py`)

Predict throughput (calls/s) vs concurrent thread count. NOTE: offline replay is
single-threaded (discrete-event sim), so this concurrency gain applies to the
**live/online mocker**, not to offline replay. Qwen3-32B / h200_sxm / vllm 0.19.0,
12 physical cores:

| variant (osl=256), calls/s | 1 thr | 4 thr | 8 thr | 12 thr | 16 thr | 24 thr |
|----------------------------|------:|------:|------:|-------:|-------:|-------:|
| `engine_pyo3` (Rust core)        | 12.4k | 47.5k | 81.9k | **100k** | 97k | 99k |
| `compiled_shell` (Python → Rust) | 12.7k | 48.0k | 80.5k | 97.6k | 99.6k | 100k |
| `opwalk` (pre-#1200 pure Python) | 4.2k | 10.9k | 10.9k | 11.0k | 10.9k | 10.9k |
| **ratio (engine / opwalk)**      | 3.0x | 4.4x | 7.5x | **9.1x** | 8.9x | 9.1x |

**Findings**

- The advantage is **bounded by physical core count, not thread count.** The
  pre-#1200 pure-Python op-walk is **GIL-capped**: throughput pins at ~11k calls/s
  from 2 threads on and never grows. The Rust core releases the GIL during compute
  and **scales to the core count** (~100k at 12 cores, ~67% efficiency), then
  plateaus. So the ratio climbs with cores and saturates: **~9x on this 12-core
  box**, reached at ~12 threads — adding threads beyond the cores does nothing
  (16/24 threads ≈ 12). A larger server raises the ceiling (≈ cores × efficiency
  / the GIL-capped Python constant); a smaller one lowers it. There is no single
  fixed multiplier.
- `engine_pyo3` (called from Python threads) is a **conservative** proxy: it still
  acquires the GIL per call to marshal args, which is what caps it at ~67%
  efficiency and makes it dip at very high thread counts with tiny calls (osl=2).
  The mocker's `RustAicCallback` calls the engine from Rust with **no GIL at all**,
  so it should scale closer to linear and not dip.
