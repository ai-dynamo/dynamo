# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency micro-bench: does AIC `predict` serialize under threads?

Measures predict throughput (calls/s) as the number of concurrent threads grows,
for three latency-prediction variants — isolating the GIL behaviour that decides
whether the Rust migration helps under concurrency (e.g. live serving with many
worker threads), which the offline-replay e2e bench does not exercise.

Variants
--------
- `engine_pyo3`  — `AicEngine.predict_decode_latency` (the compiled Rust engine).
  Its `#[pymethods]` wrap the compute in `py.allow_threads(...)`, so the GIL is
  RELEASED during the Rust compute. This is the closest Python-measurable proxy
  for the pure-Rust `RustAicCallback` hot path (the Rust callback additionally
  avoids the per-call GIL acquire for arg marshalling — the ~7% e2e edge).
- `compiled_shell` — `AicSession.predict_decode` over the compiled engine
  (Python wrapper bytecode + pyo3). What `PyAicCallback` calls today.
- `opwalk` — `AicSession.predict_decode` with the compiled engine DISABLED
  (`DYNAMO_AIC_DISABLE_COMPILED_ENGINE=1`): the pre-#1200 pure-Python op-walk,
  which holds the GIL through the whole compute.

A variant whose throughput scales with thread count is concurrency-friendly; one
that stays flat is GIL-serialized. The scaling gap between `opwalk` and the
engine variants is the concurrency benefit the Rust core (#1200) unlocked; the
pure-Rust callback preserves it and removes the residual marshalling cost.

Run:  python benchmarks/mocker/bench_aic_concurrency.py
"""

import argparse
import os
import threading
import time

from dynamo._internal.aic import create_session

MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
BACKEND_VERSION = "0.19.0"

BS, ISL = 16, 2048
THREAD_COUNTS = (1, 2, 4, 8, 12, 16, 24)


def _make_session():
    return create_session(BACKEND, SYSTEM, MODEL, 1, backend_version=BACKEND_VERSION)


def _throughput(call, n_threads, calls_per_thread):
    """Run `call` calls_per_thread times on each of n_threads threads; return calls/s."""
    barrier = threading.Barrier(n_threads + 1)

    def worker():
        barrier.wait()
        for _ in range(calls_per_thread):
            call()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    barrier.wait()  # release all workers together
    start = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    return (n_threads * calls_per_thread) / elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osl", type=int, default=256, help="decode osl (compute weight per call)"
    )
    parser.add_argument("--calls", type=int, default=20000, help="calls per thread")
    cli = parser.parse_args()

    # Build the compiled session (env unset) and the op-walk session (env set)
    # in this order so the disable toggle only affects the op-walk one.
    os.environ.pop("DYNAMO_AIC_DISABLE_COMPILED_ENGINE", None)
    compiled = _make_session()
    engine = compiled._engine
    assert engine is not None, "compiled engine unavailable — cannot run this bench"

    os.environ["DYNAMO_AIC_DISABLE_COMPILED_ENGINE"] = "1"
    opwalk = _make_session()
    os.environ.pop("DYNAMO_AIC_DISABLE_COMPILED_ENGINE", None)
    assert (
        opwalk._engine is None
    ), "op-walk session unexpectedly built a compiled engine"

    osl = cli.osl
    variants = {
        "engine_pyo3": lambda: engine.predict_decode_latency(BS, ISL, osl),
        "compiled_shell": lambda: compiled.predict_decode(BS, ISL, osl),
        "opwalk": lambda: opwalk.predict_decode(BS, ISL, osl),
    }

    print(f"AIC: {MODEL} / {SYSTEM} / {BACKEND} {BACKEND_VERSION}")
    print(f"bs={BS} isl={ISL} osl={osl}, {cli.calls} calls/thread")
    print("throughput = calls/s ; scale = throughput(N) / throughput(1)\n")

    for name, call in variants.items():
        base = None
        cells = []
        for n in THREAD_COUNTS:
            tput = _throughput(call, n, cli.calls)
            if base is None:
                base = tput
            cells.append((n, tput, tput / base))
        print(f"  {name}")
        for n, tput, scale in cells:
            print(f"    {n} thread(s): {tput:>12.0f} calls/s   scale={scale:.2f}x")
        print()

    print(
        "Read: engine_pyo3 / compiled_shell should scale up with threads "
        "(GIL released during\ncompute); opwalk should stay ~flat (GIL held). "
        "RustAicCallback ≈ engine_pyo3, minus\nthe per-call GIL/marshalling that "
        "caps engine_pyo3 at high thread counts."
    )


if __name__ == "__main__":
    main()
