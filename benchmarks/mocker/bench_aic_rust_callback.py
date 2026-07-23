# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: pure-Rust AIC callback vs Python (pyo3) AIC callback in the mocker.

HISTORICAL — this in-mocker A/B no longer runs. It drove the comparison by
toggling ``DYNAMO_AIC_DISABLE_RUST_CALLBACK`` to route the mocker through the
Python ``PyAicCallback``; that fallback was removed (the mocker AIC callback is
Rust-only now), so all modes would now resolve to ``RustAicCallback`` and the
``python_*`` columns are meaningless. The numbers it produced are recorded in
README.md. For a live Rust-vs-Python comparison that does NOT depend on the
removed fallback, use ``bench_aic_concurrency.py`` (it drives the Python
``AicSession`` directly).

Measures the end-to-end speedup of replacing the per-predict GIL + pyo3
round-trip (``PyAicCallback``) with the pure-Rust ``RustAicCallback`` that wraps
``aiconfigurator_core::AicEngine``.

Both paths run the SAME Rust latency math (post aiconfigurator #1200, where the
Python ``AicSession`` already dispatches to the compiled Rust engine). The only
difference is the call boundary on the mocker scheduler's hot path:

  * Python path  — acquires the GIL and crosses pyo3 on EVERY ``predict_prefill`` /
    ``predict_decode`` call. Under multiple mocker worker threads the GIL also
    serialises all predictions.
  * Rust path    — calls ``AicEngine::{prefill,decode}_latency_ms`` directly: no
    GIL, no pyo3 marshalling, fully concurrent across worker threads.

So the reported number is the cost of that boundary, NOT "Rust math beats Python
math" — the math is identical.

Method
------
Each mode runs in a SEPARATE subprocess. The compiled engine and the callback
are cached in process-global statics, and ``DYNAMO_AIC_DISABLE_RUST_CALLBACK`` is
read once at callback construction, so a clean A/B requires process isolation:

  * ``rust``   — default (env var unset)            -> RustAicCallback
  * ``python`` — ``DYNAMO_AIC_DISABLE_RUST_CALLBACK=1`` -> PyAicCallback

Two things are reported per workload:
  1. EQUIVALENCE — replay metrics (TTFT / TPOT / throughput) must match within
     tolerance. If they diverge the speedup is meaningless, so this gates the run.
  2. SPEEDUP — median wall-clock ratio (python / rust) over ``--repeat`` runs.

Run:  python benchmarks/mocker/bench_aic_rust_callback.py
Requires: aiconfigurator SDK installed with loadable systems/perf data for the
model/system/backend tuple, and the bindings built with the ``aic-forward-pass``
feature.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

# AIC tuple — matches components/.../tests/unit/test_replay_aic_parity.py so the
# perf database is one already exercised by CI.
MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
BACKEND_VERSION = "0.19.0"

# Workloads chosen to stress the predict hot path differently. ``num_workers`` > 1
# is where the Python GIL serialises predictions, so the Rust win is largest there.
WORKLOADS = [
    # name, isl, osl, request_count, concurrency, num_workers
    ("decode_heavy_1w", 1024, 1024, 256, 64, 1),
    ("balanced_1w", 2048, 256, 256, 64, 1),
    ("decode_heavy_4w", 1024, 1024, 512, 128, 4),
]

# Metrics compared for equivalence (both paths share the same Rust math, so these
# should match to well within tolerance).
EQUIV_KEYS = ("mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s")
EQUIV_REL_TOL = 1e-6


def _engine_args():
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 200000,
        "speedup_ratio": 1.0,
        "aic_backend": BACKEND,
        "aic_system": SYSTEM,
        "aic_backend_version": BACKEND_VERSION,
        "aic_tp_size": 1,
        "aic_model_path": MODEL,
    }
    return MockEngineArgs.from_json(json.dumps(payload))


def _run_one(isl, osl, request_count, concurrency, num_workers):
    """Run a single replay, return (report, wall_clock_seconds)."""
    args = _engine_args()
    start = time.perf_counter()
    report = run_synthetic_trace_replay(
        isl,
        osl,
        request_count,
        extra_engine_args=args,
        num_workers=num_workers,
        replay_mode="offline",
        replay_concurrency=concurrency,
    )
    elapsed = time.perf_counter() - start
    return report, elapsed


def _worker_main():
    """Subprocess entry: time one workload `repeat` times, emit JSON to stdout.

    Runs `warmup` UNTIMED replays first. The one-time engine build
    (``build_aic_engine`` -> Python ``compile_engine`` + Rust parquet load) and
    cold OS-cache costs are paid then and cached process-globally, so the timed
    runs isolate steady-state replay — which is where the per-predict dispatch
    cost (the thing RustAicCallback removes) actually lives.
    """
    isl = int(os.environ["_AIC_BENCH_ISL"])
    osl = int(os.environ["_AIC_BENCH_OSL"])
    request_count = int(os.environ["_AIC_BENCH_REQS"])
    concurrency = int(os.environ["_AIC_BENCH_CONC"])
    num_workers = int(os.environ["_AIC_BENCH_WORKERS"])
    repeat = int(os.environ["_AIC_BENCH_REPEAT"])
    warmup = int(os.environ.get("_AIC_BENCH_WARMUP", "1"))

    report = None
    for _ in range(warmup):
        report, _ = _run_one(isl, osl, request_count, concurrency, num_workers)

    times = []
    for _ in range(repeat):
        report, elapsed = _run_one(isl, osl, request_count, concurrency, num_workers)
        times.append(elapsed)

    out = {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "times_s": times,
        "report": {k: report[k] for k in EQUIV_KEYS if k in report},
    }
    # Sentinel-framed so any import chatter on stdout does not corrupt parsing.
    print("__AIC_BENCH_RESULT__" + json.dumps(out))


def _run_mode_subprocess(
    mode, isl, osl, request_count, concurrency, num_workers, repeat, warmup
):
    env = dict(os.environ)
    env["_AIC_BENCH_WORKER"] = "1"
    env["_AIC_BENCH_ISL"] = str(isl)
    env["_AIC_BENCH_OSL"] = str(osl)
    env["_AIC_BENCH_REQS"] = str(request_count)
    env["_AIC_BENCH_CONC"] = str(concurrency)
    env["_AIC_BENCH_WORKERS"] = str(num_workers)
    env["_AIC_BENCH_REPEAT"] = str(repeat)
    env["_AIC_BENCH_WARMUP"] = str(warmup)
    # rust            : RustAicCallback (pure-Rust crate, direct call)
    # python_compiled : PyAicCallback -> AicSession on the compiled Rust engine
    #                   (= same math, pyo3+GIL boundary per call)
    # python_opwalk   : PyAicCallback -> AicSession pure-Python op-walk
    #                   (= the pre-#1200 "no Rust core" baseline, via pyo3)
    env.pop("DYNAMO_AIC_DISABLE_RUST_CALLBACK", None)
    env.pop("DYNAMO_AIC_DISABLE_COMPILED_ENGINE", None)
    if mode == "python_compiled":
        env["DYNAMO_AIC_DISABLE_RUST_CALLBACK"] = "1"
    elif mode == "python_opwalk":
        env["DYNAMO_AIC_DISABLE_RUST_CALLBACK"] = "1"
        env["DYNAMO_AIC_DISABLE_COMPILED_ENGINE"] = "1"

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{mode} worker failed (rc={proc.returncode})")
    for line in proc.stdout.splitlines():
        if line.startswith("__AIC_BENCH_RESULT__"):
            return json.loads(line[len("__AIC_BENCH_RESULT__") :])
    raise RuntimeError(
        f"{mode} worker produced no result\n{proc.stdout}\n{proc.stderr}"
    )


def _check_equivalence(name, rust_report, py_report):
    ok = True
    for k in EQUIV_KEYS:
        if k not in rust_report or k not in py_report:
            continue
        r, p = rust_report[k], py_report[k]
        denom = max(abs(r), abs(p), 1e-12)
        rel = abs(r - p) / denom
        flag = "OK" if rel <= EQUIV_REL_TOL else "MISMATCH"
        if rel > EQUIV_REL_TOL:
            ok = False
        print(f"    {k:<26} rust={r:<14.6f} python={p:<14.6f} rel={rel:.2e} [{flag}]")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed runs per mode (median + min reported)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed warmup runs (amortize engine build)",
    )
    cli = parser.parse_args()

    print(f"AIC: {MODEL} / {SYSTEM} / {BACKEND} {BACKEND_VERSION}")
    print(
        f"warmup={cli.warmup} repeat={cli.repeat} (steady-state = min over timed runs)\n"
    )

    rows = []
    for name, isl, osl, request_count, concurrency, num_workers in WORKLOADS:
        print(
            f"[{name}] isl={isl} osl={osl} reqs={request_count} "
            f"conc={concurrency} workers={num_workers}"
        )

        def run(m):
            return _run_mode_subprocess(
                m,
                isl,
                osl,
                request_count,
                concurrency,
                num_workers,
                cli.repeat,
                cli.warmup,
            )

        rust = run("rust")
        opwalk = run("python_opwalk")
        compiled = run("python_compiled")

        # opwalk is the pre-#1200 baseline (pure Python). It will NOT be
        # bit-identical to the Rust engine (different algorithm), so only
        # cross-check the compiled path for equivalence.
        equiv = _check_equivalence(name, rust["report"], compiled["report"])

        big_win = opwalk["min_s"] / rust["min_s"] if rust["min_s"] else float("nan")
        this_pr = compiled["min_s"] / rust["min_s"] if rust["min_s"] else float("nan")
        print(
            f"    steady-state(min)  rust={rust['min_s']:.3f}s  "
            f"opwalk(pure-py)={opwalk['min_s']:.3f}s  compiled={compiled['min_s']:.3f}s"
        )
        print(
            f"    >>> BIG WIN (rust vs pure-python)={big_win:.2f}x   "
            f"this-PR (rust vs compiled-shell)={this_pr:.2f}x\n"
        )
        rows.append(
            (
                name,
                rust["min_s"],
                opwalk["min_s"],
                compiled["min_s"],
                big_win,
                this_pr,
                equiv,
            )
        )

    print("=" * 88)
    print(
        f"{'workload':<18}{'rust(s)':>9}{'pure-py(s)':>12}{'compiled(s)':>12}{'BIG WIN':>10}{'this-PR':>9}{'equiv':>7}"
    )
    for name, r, o, c, bw, pr, e in rows:
        print(
            f"{name:<18}{r:>9.3f}{o:>12.3f}{c:>12.3f}{bw:>9.2f}x{pr:>8.2f}x{('ok' if e else 'NO'):>7}"
        )
    print("=" * 88)
    print(
        "BIG WIN = Rust crate direct vs pre-#1200 pure-Python AIC (both via the mocker)."
    )


if __name__ == "__main__":
    if os.environ.get("_AIC_BENCH_WORKER"):
        _worker_main()
    else:
        main()
