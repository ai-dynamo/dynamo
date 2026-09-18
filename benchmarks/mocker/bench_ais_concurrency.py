# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure AIS canonical Python session query throughput under threads.

This measures the Python facade and FPM serialization as well as Rust estimation.
It does not measure Dynamo's pure-Rust Router/Mocker callback hot path. Historical
op-walk comparisons are no longer available because that implementation was removed.

Run: python benchmarks/mocker/bench_ais_concurrency.py
"""

import argparse
import threading
import time

from dynamo._internal.ais import create_session

MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
BACKEND_VERSION = "current"

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

    session = _make_session()
    osl = cli.osl
    variants = {"canonical_session": lambda: session.predict_decode(BS, ISL, osl)}

    print(f"AIS: {MODEL} / {SYSTEM} / {BACKEND} {BACKEND_VERSION}")
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


if __name__ == "__main__":
    main()
