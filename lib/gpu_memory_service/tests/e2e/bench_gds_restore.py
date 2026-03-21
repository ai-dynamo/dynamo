#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: GDS vs O_DIRECT restore throughput (standalone, no GMS server needed).

Creates synthetic shard files on disk and benchmarks reading them into GPU memory
using (a) NIXL GDS (direct file→GPU) and (b) O_DIRECT + H2D copy baseline.

Usage:
    python bench_gds_restore.py --data-dir /mnt/weka/gds_bench --total-gib 135 --device 0
"""

import argparse
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_shard_files(data_dir: str, total_bytes: int, shard_size: int) -> list[str]:
    """Create synthetic shard files filled with a pattern."""
    shards_dir = os.path.join(data_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    paths = []
    remaining = total_bytes
    shard_idx = 0
    while remaining > 0:
        size = min(shard_size, remaining)
        path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.bin")
        if os.path.exists(path) and os.path.getsize(path) == size:
            logger.info("Shard %s already exists (%d bytes), skipping", path, size)
        else:
            logger.info("Creating shard %s (%d bytes)...", path, size)
            # Write in 256MB chunks to avoid huge memory alloc
            chunk_size = 256 * 1024 * 1024
            with open(path, "wb") as f:
                written = 0
                while written < size:
                    n = min(chunk_size, size - written)
                    f.write(b"\xba" * n)
                    written += n
        paths.append(path)
        remaining -= size
        shard_idx += 1

    logger.info(
        "Created %d shard files (%.2f GiB total)", len(paths), total_bytes / (1024**3)
    )
    return paths


def bench_gds(shard_paths: list[str], device: int) -> float:
    """Benchmark: read shards into GPU memory using NIXL GDS."""
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    agent_name = f"bench_gds_{device}"
    agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
    agent.create_backend("GDS_MT")

    # Pre-allocate GPU buffer for all shards
    total_bytes = sum(os.path.getsize(p) for p in shard_paths)
    gpu_buf = torch.empty(total_bytes, dtype=torch.uint8, device=f"cuda:{device}")
    gpu_ptr = gpu_buf.data_ptr()

    # Register VRAM
    vram_reg = agent.register_memory([(gpu_ptr, total_bytes, device, "bench")], "VRAM")

    # Warm up: small read
    warmup_size = min(4096, total_bytes)
    fd = os.open(shard_paths[0], os.O_RDONLY)
    file_reg = agent.register_memory([(0, warmup_size, fd, "")], "FILE")
    vram_xfer = agent.get_xfer_descs([(gpu_ptr, warmup_size, device)], "VRAM")
    file_xfer = file_reg.trim()
    h = agent.initialize_xfer("READ", vram_xfer, file_xfer, agent_name)
    s = agent.transfer(h)
    while s == "PROC":
        s = agent.check_xfer_state(h)
    agent.release_xfer_handle(h)
    agent.deregister_memory(file_reg)
    os.close(fd)

    torch.cuda.synchronize(device)

    # Benchmark
    offset = 0
    pending = []

    t0 = time.monotonic()

    for path in shard_paths:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        file_reg = agent.register_memory([(0, size, fd, "")], "FILE")
        vram_xfer = agent.get_xfer_descs([(gpu_ptr + offset, size, device)], "VRAM")
        file_xfer = file_reg.trim()

        h = agent.initialize_xfer("READ", vram_xfer, file_xfer, agent_name)
        state = agent.transfer(h)
        if state == "ERR":
            raise RuntimeError(f"GDS transfer failed for {path}")
        pending.append((h, file_reg, fd))
        offset += size

    # Wait for all
    for h, file_reg, fd in pending:
        state = agent.check_xfer_state(h)
        while state == "PROC":
            state = agent.check_xfer_state(h)
        if state == "ERR":
            raise RuntimeError("GDS transfer error during wait")

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - t0

    # Cleanup
    for h, file_reg, fd in pending:
        agent.release_xfer_handle(h)
        agent.deregister_memory(file_reg)
        os.close(fd)
    agent.deregister_memory(vram_reg)

    del gpu_buf
    return elapsed


def bench_odirect(shard_paths: list[str], device: int) -> float:
    """Benchmark: read shards into GPU via O_DIRECT + H2D copy (baseline)."""
    import torch

    # Warm up
    torch.cuda.synchronize(device)

    t0 = time.monotonic()

    # Read each shard with O_DIRECT into pinned memory, then copy to GPU
    for path in shard_paths:
        size = os.path.getsize(path)
        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        arr = pinned.numpy()

        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        done = 0
        mv = memoryview(arr)
        while done < size:
            n = os.readv(fd, [mv[done:]])
            if n == 0:
                raise RuntimeError(f"Short read: {done}/{size}")
            done += n
        mv.release()
        os.close(fd)

        pinned.to(f"cuda:{device}", non_blocking=True)

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - t0

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDS vs O_DIRECT restore")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory for shard files (e.g. /mnt/weka/gds_bench)",
    )
    parser.add_argument(
        "--total-gib",
        type=float,
        default=135.0,
        help="Total data size in GiB (default: 135 for Qwen-72B BF16)",
    )
    parser.add_argument(
        "--shard-gib",
        type=float,
        default=4.0,
        help="Shard file size in GiB (default: 4)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index (default: 0)"
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip shard creation (reuse existing)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip O_DIRECT baseline"
    )
    parser.add_argument("--skip-gds", action="store_true", help="Skip GDS benchmark")
    args = parser.parse_args()

    total_bytes = int(args.total_gib * 1024**3)
    shard_size = int(args.shard_gib * 1024**3)

    logger.info("=== GDS Restore Benchmark ===")
    logger.info(
        "Total: %.2f GiB, Shard size: %.2f GiB, Device: cuda:%d",
        args.total_gib,
        args.shard_gib,
        args.device,
    )

    # Create shard files
    if not args.skip_create:
        shard_paths = create_shard_files(args.data_dir, total_bytes, shard_size)
    else:
        shards_dir = os.path.join(args.data_dir, "shards")
        shard_paths = sorted(
            os.path.join(shards_dir, f)
            for f in os.listdir(shards_dir)
            if f.startswith("shard_") and f.endswith(".bin")
        )
        actual = sum(os.path.getsize(p) for p in shard_paths)
        logger.info(
            "Reusing %d existing shards (%.2f GiB)",
            len(shard_paths),
            actual / (1024**3),
        )

    total_gib = sum(os.path.getsize(p) for p in shard_paths) / (1024**3)

    # Drop page cache before each benchmark
    def drop_cache():
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            logger.info("Page cache dropped")
        except PermissionError:
            logger.warning("Cannot drop page cache (not root?)")

    # GDS benchmark
    if not args.skip_gds:
        drop_cache()
        logger.info("--- GDS benchmark ---")
        gds_time = bench_gds(shard_paths, args.device)
        logger.info(
            "GDS:      %.2f GiB in %.3fs = %.2f GiB/s",
            total_gib,
            gds_time,
            total_gib / gds_time,
        )
    else:
        gds_time = None

    # O_DIRECT baseline
    if not args.skip_baseline:
        drop_cache()
        logger.info("--- O_DIRECT + H2D baseline ---")
        baseline_time = bench_odirect(shard_paths, args.device)
        logger.info(
            "O_DIRECT: %.2f GiB in %.3fs = %.2f GiB/s",
            total_gib,
            baseline_time,
            total_gib / baseline_time,
        )
    else:
        baseline_time = None

    # Summary
    logger.info("=== Summary ===")
    logger.info("Data size: %.2f GiB (%d shards)", total_gib, len(shard_paths))
    if gds_time is not None:
        logger.info("GDS:      %.3fs  (%.2f GiB/s)", gds_time, total_gib / gds_time)
    if baseline_time is not None:
        logger.info(
            "O_DIRECT: %.3fs  (%.2f GiB/s)", baseline_time, total_gib / baseline_time
        )
    if gds_time and baseline_time:
        logger.info("Speedup:  %.2fx", baseline_time / gds_time)


if __name__ == "__main__":
    main()
