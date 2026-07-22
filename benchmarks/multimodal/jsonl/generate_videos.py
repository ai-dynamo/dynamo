# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating and sampling video pools."""

import hashlib
import random
from pathlib import Path

import numpy as np


def _synthetic_video_key(
    seed: int,
    video_idx: int,
    width: int,
    height: int,
    fps: int,
    seconds: int,
) -> str:
    return f"seed{seed}_idx{video_idx:04d}_{width}x{height}_{fps}fps_{seconds}s"


def _derive_synthetic_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _write_synthetic_video(
    path: Path,
    seed: int,
    width: int,
    height: int,
    fps: int,
    seconds: int,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Synthetic video generation requires imageio. Install imageio and "
            "imageio-ffmpeg."
        ) from exc

    frame_count = fps * seconds
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=3, dtype=np.uint8)
    accent = rng.integers(0, 256, size=3, dtype=np.uint8)
    rect_w = max(4, width // 4)
    rect_h = max(4, height // 4)
    span_x = max(1, width - rect_w + 1)
    span_y = max(1, height - rect_h + 1)
    offset_x = int(rng.integers(0, span_x))
    offset_y = int(rng.integers(0, span_y))
    speed_x = int(rng.integers(1, max(2, width // 8)))
    speed_y = int(rng.integers(1, max(2, height // 8)))
    xx = np.arange(width, dtype=np.uint16)[None, :]
    yy = np.arange(height, dtype=np.uint16)[:, None]

    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-threads",
            "1",
        ],
    ) as writer:
        for frame_idx in range(frame_count):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = ((xx + int(base[0]) + frame_idx * speed_x) % 256).astype(
                np.uint8
            )
            frame[:, :, 1] = ((yy + int(base[1]) + frame_idx * speed_y) % 256).astype(
                np.uint8
            )
            frame[:, :, 2] = (
                (xx // 2 + yy // 2 + int(base[2]) + frame_idx * 7) % 256
            ).astype(np.uint8)

            x = (offset_x + frame_idx * speed_x) % span_x
            y = (offset_y + frame_idx * speed_y) % span_y
            frame[y : y + rect_h, x : x + rect_w, :] = accent
            writer.append_data(frame)


def generate_synthetic_video_pool(
    pool_size: int,
    video_dir: Path,
    video_size: tuple[int, int],
    fps: int,
    seconds: int,
    seed: int,
) -> list[str]:
    """Generate pool_size deterministic local MP4 files and return their paths."""
    width, height = video_size
    if width <= 0 or height <= 0:
        raise ValueError(f"synthetic video size must be positive, got {video_size}")
    if width % 2 or height % 2:
        raise ValueError(
            f"synthetic video dimensions must be even for yuv420p, got {video_size}"
        )
    if fps <= 0:
        raise ValueError(f"synthetic video fps must be positive, got {fps}")
    if seconds <= 0:
        raise ValueError(f"synthetic video seconds must be positive, got {seconds}")

    video_dir.mkdir(parents=True, exist_ok=True)
    pool: list[str] = []
    for idx in range(pool_size):
        key = _synthetic_video_key(seed, idx, width, height, fps, seconds)
        path = video_dir / f"synthetic_{key}.mp4"
        if not path.exists() or path.stat().st_size == 0:
            video_seed = _derive_synthetic_seed(key)
            _write_synthetic_video(path, video_seed, width, height, fps, seconds)
        pool.append(str(path.resolve()))

    print(
        f"  {pool_size} synthetic {width}x{height} videos "
        f"({fps} fps, {seconds}s) saved to {video_dir}"
    )
    return pool


def sample_video_slots(
    py_rng: random.Random,
    pool: list[str],
    num_requests: int,
    videos_per_request: int,
) -> list[str]:
    """Sample video slots from a fixed pool, no duplicates within each request.

    Every video in the pool is guaranteed to appear at least once.
    """
    pool_size = len(pool)
    total_slots = num_requests * videos_per_request
    if pool_size < videos_per_request:
        raise ValueError(
            f"videos-pool ({pool_size}) must be >= "
            f"videos-per-request ({videos_per_request})"
        )
    if total_slots < pool_size:
        raise ValueError(
            f"total slots ({num_requests}x{videos_per_request}={total_slots}) < "
            f"videos-pool ({pool_size}). Increase --num-requests or "
            f"--videos-per-request, or reduce --videos-pool."
        )

    # Round-robin every pool video into requests so each appears at least once
    shuffled = list(pool)
    py_rng.shuffle(shuffled)
    requests: list[list[str]] = [[] for _ in range(num_requests)]
    for i, video in enumerate(shuffled):
        requests[i % num_requests].append(video)

    # Fill remaining slots with random pool samples (no intra-request duplicates)
    for req in requests:
        remaining = videos_per_request - len(req)
        if remaining > 0:
            used = set(req)
            available = [video for video in pool if video not in used]
            req.extend(py_rng.sample(available, remaining))
        py_rng.shuffle(req)

    slot_refs = [video for req in requests for video in req]
    num_unique = len(set(slot_refs))
    print(
        f"Generated {total_slots} video slots from pool of {pool_size}: "
        f"{num_unique} unique in use, "
        f"{total_slots - num_unique} duplicate references "
        f"({(total_slots - num_unique) / total_slots:.1%} reuse)"
    )
    return slot_refs
