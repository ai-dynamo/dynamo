#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repack an existing GMS save directory with a different shard size.

Reads all allocations from the source shards and rewrites them into new shard
files capped at *--shard-size-bytes*.  The allocation IDs, sizes, and tags are
preserved; only tensor_file and tensor_offset change.

Usage
-----
    .venv/bin/python reshard.py \
        --source /mnt/nvme2/gms_bench_save_72b \
        --dest   /mnt/nvme2/gms_bench_save_1gib \
        --shard-size-bytes 1073741824
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def reshard(source_dir: Path, dest_dir: Path, shard_size_bytes: int) -> None:
    from gpu_memory_service.client.gms_storage_client import (
        SaveManifest,
        _group_entries_by_shard,
        _read_shard_sequential,
        _ShardWriter,
    )

    manifest = SaveManifest.from_dict(
        json.loads((source_dir / "manifest.json").read_text())
    )
    total_gib = sum(e.aligned_size for e in manifest.allocations) / 1024**3
    logger.info(
        "Source: %s  (%.2f GiB, %d allocs, %d shards)",
        source_dir,
        total_gib,
        len(manifest.allocations),
        len({e.tensor_file for e in manifest.allocations}),
    )
    logger.info(
        "Dest:   %s  (shard_size=%.2f GiB)",
        dest_dir,
        shard_size_bytes / 1024**3,
    )

    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError(
            f"Destination directory {dest_dir} must not already contain files"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = dest_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Read source shards in order; write to new shards
    groups = _group_entries_by_shard(manifest.allocations)

    new_by_id: dict = {}  # allocation_id → new AllocationEntry
    t0 = time.monotonic()
    bytes_written = 0

    with _ShardWriter(str(shards_dir), shard_size_bytes=shard_size_bytes) as writer:
        for rel_path, sorted_entries in sorted(groups.items()):
            abs_path = str(source_dir / rel_path)
            logger.info(
                "  reading %s  (%d allocs, %.2f GiB)",
                rel_path,
                len(sorted_entries),
                sum(e.aligned_size for e in sorted_entries) / 1024**3,
            )
            tensors = _read_shard_sequential(abs_path, sorted_entries, device=-1)
            for entry in sorted_entries:
                src = tensors[entry.allocation_id]
                new_rel, new_off = writer.write(src)
                new_by_id[entry.allocation_id] = replace(
                    entry, tensor_file=new_rel, tensor_offset=new_off
                )
                bytes_written += entry.aligned_size

    elapsed = time.monotonic() - t0
    gib_written = bytes_written / 1024**3
    logger.info(
        "Resharding done: %.2f GiB in %.1fs  (%.2f GiB/s)",
        gib_written,
        elapsed,
        gib_written / elapsed if elapsed > 0 else 0,
    )

    # Write new manifest preserving original allocation order
    new_allocs = [new_by_id[e.allocation_id] for e in manifest.allocations]
    new_manifest = SaveManifest(
        version=manifest.version,
        timestamp=manifest.timestamp,
        layout_hash=manifest.layout_hash,
        device=manifest.device,
        allocations=new_allocs,
    )
    (dest_dir / "manifest.json").write_text(
        json.dumps(new_manifest.to_dict(), indent=2)
    )
    n_new_shards = len({e.tensor_file for e in new_allocs})
    logger.info(
        "Wrote manifest: %d allocs across %d shards", len(new_allocs), n_new_shards
    )

    # Copy metadata
    meta_src = source_dir / "gms_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, dest_dir / "gms_metadata.json")

    logger.info("Done → %s", dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, type=Path, help="Source save directory"
    )
    parser.add_argument(
        "--dest", required=True, type=Path, help="Destination directory"
    )
    parser.add_argument(
        "--shard-size-bytes",
        type=int,
        default=1 * 1024**3,
        help="Soft shard size limit in bytes (default: 1 GiB)",
    )
    args = parser.parse_args()
    reshard(args.source, args.dest, args.shard_size_bytes)


if __name__ == "__main__":
    main()
