# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for video postprocessing: H.264 MP4 encoding.

Compares Python (imageio + ffmpeg subprocess) vs Rust (dynamo._core.encode_video).

Usage:
    python benchmarks/video_postprocessing_bench.py
"""

import io
import statistics
import time

import numpy as np

# Default T2V dimensions (Wan 1.3B)
NUM_FRAMES = 17
WIDTH, HEIGHT = 832, 480
FPS = 16
ITERATIONS = 100


def _make_random_frames() -> np.ndarray:
    """Generate random NHWC RGB24 video data."""
    return np.random.randint(0, 256, (NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


def bench_python_imageio(frames: np.ndarray) -> list[float]:
    """Benchmark imageio + ffmpeg subprocess encoding."""
    import imageio

    frame_list = [frames[i] for i in range(frames.shape[0])]
    times = []
    for _ in range(ITERATIONS):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        with imageio.get_writer(
            buf,
            format="mp4",
            fps=FPS,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for frame in frame_list:
                writer.append_data(frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_rust_encode_video(frames: np.ndarray) -> list[float]:
    """Benchmark Rust in-process H.264 encoding."""
    from dynamo._core import encode_video

    data = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    n, h, w, _c = frames.shape
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        encode_video(data, w, h, n, FPS)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def _print_stats(label: str, times: list[float]) -> None:
    s = sorted(times)
    p50 = statistics.median(s)
    p95 = s[int(len(s) * 0.95)]
    p99 = s[int(len(s) * 0.99)]
    mean = statistics.mean(s)
    print(
        f"  {label:45s}  mean={mean:7.2f}ms  p50={p50:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms"
    )


def main():
    print(
        f"Video postprocessing benchmark ({NUM_FRAMES} frames, {WIDTH}x{HEIGHT}, {ITERATIONS} iterations)"
    )
    print("=" * 100)

    frames = _make_random_frames()
    raw_size = frames.nbytes
    print(f"  Raw frame data: {raw_size:,} bytes ({raw_size / 1024 / 1024:.1f} MB)")
    print()

    # Python baseline
    try:
        print("Python (imageio + ffmpeg subprocess):")
        py_times = bench_python_imageio(frames)
        _print_stats("Video encode (imageio)", py_times)
        print()
    except ImportError:
        print("imageio not available, skipping Python baseline.")
        py_times = None
        print()

    # Rust encoder
    try:
        print("Rust (dynamo._core.encode_video, GIL released):")
        rs_times = bench_rust_encode_video(frames)
        _print_stats("Video encode (Rust)", rs_times)
        print()
    except ImportError:
        print(
            "Rust encoder not available (dynamo._core.encode_video not built). Skipping."
        )
        rs_times = None
        print()

    # Summary
    if py_times and rs_times:
        py_mean = statistics.mean(py_times)
        rs_mean = statistics.mean(rs_times)
        speedup = py_mean / rs_mean
        print(f"Speedup: {speedup:.1f}x (Python {py_mean:.1f}ms -> Rust {rs_mean:.1f}ms)")


if __name__ == "__main__":
    main()
