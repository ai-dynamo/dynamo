# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for image diffusion postprocessing: PNG encoding and base64 encoding.

Compares Python (PIL + stdlib base64) vs Rust (dynamo._core) implementations.

Usage:
    python benchmarks/image_postprocessing_bench.py
"""

import base64
import io
import statistics
import time

import numpy as np
from PIL import Image

ITERATIONS = 100
WIDTH, HEIGHT, CHANNELS = 1024, 1024, 3


def _make_random_image_data() -> np.ndarray:
    """Generate random RGB image data (HWC layout, uint8)."""
    return np.random.randint(0, 256, (HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)


def bench_python_png_encode(data: np.ndarray) -> list[float]:
    """Benchmark PIL Image.save(format='PNG')."""
    img = Image.fromarray(data)
    times = []
    for _ in range(ITERATIONS):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        img.save(buf, format="PNG")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
        buf.seek(0)
        buf.truncate()
    return times


def bench_python_base64_encode(payload: bytes) -> list[float]:
    """Benchmark stdlib base64.b64encode()."""
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        base64.b64encode(payload)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_python_combined(data: np.ndarray) -> list[float]:
    """Benchmark combined PNG encode + base64 encode (Python path)."""
    img = Image.fromarray(data)
    times = []
    for _ in range(ITERATIONS):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        base64.b64encode(png_bytes)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        buf.seek(0)
        buf.truncate()
    return times


def _print_stats(label: str, times: list[float]) -> None:
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    p99 = sorted(times)[int(len(times) * 0.99)]
    mean = statistics.mean(times)
    print(f"  {label:40s}  mean={mean:7.2f}ms  p50={p50:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms")


def main():
    print(f"Image postprocessing benchmark ({WIDTH}x{HEIGHT}x{CHANNELS}, {ITERATIONS} iterations)")
    print("=" * 90)

    data = _make_random_image_data()

    # Generate a reference PNG for base64 benchmarks
    img = Image.fromarray(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    print(f"  Reference PNG size: {len(png_bytes):,} bytes ({len(png_bytes)/1024/1024:.2f} MB)")
    print()

    # --- Python baselines ---
    print("Python (PIL + stdlib base64):")
    _print_stats("PNG encode (PIL)", bench_python_png_encode(data))
    _print_stats("base64 encode (stdlib)", bench_python_base64_encode(png_bytes))
    _print_stats("Combined (PNG + base64)", bench_python_combined(data))
    print()

    # --- Rust encoders ---
    try:
        from dynamo._core import encode_base64 as rust_encode_base64
        from dynamo._core import encode_image as rust_encode_image

        raw_bytes = data.tobytes()

        # Rust PNG encode
        times_rust_png = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            rust_encode_image(raw_bytes, WIDTH, HEIGHT, CHANNELS, "png")
            t1 = time.perf_counter()
            times_rust_png.append((t1 - t0) * 1000)

        # Rust base64 encode
        times_rust_b64 = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            rust_encode_base64(png_bytes)
            t1 = time.perf_counter()
            times_rust_b64.append((t1 - t0) * 1000)

        # Rust combined
        times_rust_combined = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            png_out = bytes(rust_encode_image(raw_bytes, WIDTH, HEIGHT, CHANNELS, "png"))
            rust_encode_base64(png_out)
            t1 = time.perf_counter()
            times_rust_combined.append((t1 - t0) * 1000)

        print("Rust (dynamo._core, GIL released):")
        _print_stats("PNG encode (Rust)", times_rust_png)
        _print_stats("base64 encode (Rust)", times_rust_b64)
        _print_stats("Combined (PNG + base64, Rust)", times_rust_combined)
        print()

    except ImportError:
        print("Rust encoders not available (dynamo._core not built). Skipping.")
        print()


if __name__ == "__main__":
    main()
