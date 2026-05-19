#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrate collected power data into an AIConfigurator source checkout.

The `aic_[h,b]200_power_data` directories contain kernel-level benchmark
CSV files (`*_perf.txt`) with ``latency,power,power_limit`` columns.
These need to be placed into the AIC package's ``systems/data/{system}/``
directories so that ``aiconfigurator.sdk.perf_database.get_database()``
can load them and ``estimate_perf()`` returns real ``power_w`` values.

Usage
-----
    python tools/integrate_aic_power_data.py \\
        --aic-checkout /path/to/aiconfigurator \\
        --h200-data    /path/to/aic_h200_power_data \\
        --b200-data    /path/to/aic_b200_power_data

    # Preview without writing:
    python tools/integrate_aic_power_data.py \\
        --aic-checkout /path/to/aiconfigurator --dry-run

Mapping (source → AIC systems/data/)
-------------------------------------
H200 (h200_sxm):
  trtllm/1.3.0rc2/*.txt          → h200_sxm/trtllm/1.3.0rc2/
  trtllm_allreduce/*.txt          → h200_sxm/trtllm/1.3.0rc2/  (merged)
  vllm/0.19.1/*.txt               → h200_sxm/vllm/0.19.1/
  nccl/nccl_perf.txt              → h200_sxm/nccl/2.29.2/nccl_perf.txt

B200 (b200_sxm):
  trtllm/1.3.0rc6/*.txt          → b200_sxm/trtllm/1.3.0rc6/
  vllm/0.19.0/*.txt               → b200_sxm/vllm/0.19.0/
  sglang/0.5.10.post1/*.txt       → b200_sxm/sglang/0.5.10.post1/
  systems/b200_sxm.yaml           → {aic_checkout}/src/aiconfigurator/systems/b200_sxm.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Transfer spec: (source_relative_to_data_root, destination_relative_to_systems_data)
#   Each entry is (src_glob_dir, dst_dir).
#   Files are matched as "*.txt" unless a specific filename is given.
# ---------------------------------------------------------------------------

H200_TRANSFERS: list[tuple[str, str, str]] = [
    # (src_dir_relative,            dst_dir_relative,                  file_glob)
    ("trtllm/1.3.0rc2", "h200_sxm/trtllm/1.3.0rc2", "*.txt"),
    ("trtllm_allreduce", "h200_sxm/trtllm/1.3.0rc2", "*.txt"),
    ("vllm/0.19.1", "h200_sxm/vllm/0.19.1", "*.txt"),
    ("nccl", "h200_sxm/nccl/2.29.2", "nccl_perf.txt"),
]

B200_TRANSFERS: list[tuple[str, str, str]] = [
    ("trtllm/1.3.0rc6", "b200_sxm/trtllm/1.3.0rc6", "*.txt"),
    ("vllm/0.19.0", "b200_sxm/vllm/0.19.0", "*.txt"),
    ("sglang/0.5.10.post1", "b200_sxm/sglang/0.5.10.post1", "*.txt"),
]


def _systems_data_root(aic_checkout: Path) -> Path:
    """Return the AIC ``systems/data`` directory inside an AIC checkout."""
    candidates = [
        aic_checkout / "src" / "aiconfigurator" / "systems" / "data",
        aic_checkout / "aiconfigurator" / "systems" / "data",
        aic_checkout / "systems" / "data",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate 'systems/data' under {aic_checkout}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _systems_root(aic_checkout: Path) -> Path:
    """Return the AIC ``systems`` directory (holds *.yaml specs)."""
    candidates = [
        aic_checkout / "src" / "aiconfigurator" / "systems",
        aic_checkout / "aiconfigurator" / "systems",
        aic_checkout / "systems",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate 'systems/' under {aic_checkout}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _copy_files(
    src_dir: Path,
    dst_dir: Path,
    file_glob: str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Copy matching files from src_dir → dst_dir.  Returns count of files handled."""
    files = sorted(src_dir.glob(file_glob))
    if not files:
        print(f"  [WARN] No files matching '{file_glob}' in {src_dir}")
        return 0

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src in files:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            print(f"  [SKIP] {dst} already exists (use --overwrite to replace)")
            continue
        action = "COPY" if not dry_run else "DRY-RUN"
        print(f"  [{action}] {src} → {dst}")
        if not dry_run:
            shutil.copy2(src, dst)
        count += 1
    return count


def integrate(
    aic_checkout: Path,
    h200_data: Path | None,
    b200_data: Path | None,
    *,
    dry_run: bool,
    overwrite: bool,
) -> None:
    systems_data = _systems_data_root(aic_checkout)
    systems_dir = _systems_root(aic_checkout)
    total = 0

    # ---- H200 ----
    if h200_data is not None:
        print(f"\n=== H200 data: {h200_data} ===")
        for src_rel, dst_rel, glob in H200_TRANSFERS:
            src_dir = h200_data / src_rel
            dst_dir = systems_data / dst_rel
            if not src_dir.is_dir():
                print(f"  [WARN] Source directory not found: {src_dir}")
                continue
            print(f"  {src_rel}  →  {dst_rel}/")
            total += _copy_files(
                src_dir, dst_dir, glob, dry_run=dry_run, overwrite=overwrite
            )

    # ---- B200 ----
    if b200_data is not None:
        print(f"\n=== B200 data: {b200_data} ===")

        # 1. Perf data files
        for src_rel, dst_rel, glob in B200_TRANSFERS:
            src_dir = b200_data / src_rel
            dst_dir = systems_data / dst_rel
            if not src_dir.is_dir():
                print(f"  [WARN] Source directory not found: {src_dir}")
                continue
            print(f"  {src_rel}  →  {dst_rel}/")
            total += _copy_files(
                src_dir, dst_dir, glob, dry_run=dry_run, overwrite=overwrite
            )

        # 2. b200_sxm.yaml system spec
        spec_src = b200_data / "systems" / "b200_sxm.yaml"
        spec_dst = systems_dir / "b200_sxm.yaml"
        if spec_src.is_file():
            print(f"\n  b200_sxm.yaml  →  {spec_dst}")
            if spec_dst.exists() and not overwrite:
                print(
                    f"  [SKIP] {spec_dst} already exists (use --overwrite to replace)"
                )
            else:
                action = "COPY" if not dry_run else "DRY-RUN"
                print(f"  [{action}] {spec_src} → {spec_dst}")
                if not dry_run:
                    shutil.copy2(spec_src, spec_dst)
                total += 1
        else:
            print(f"  [WARN] b200_sxm.yaml not found at {spec_src}")

    print(f"\n{'Would transfer' if dry_run else 'Transferred'} {total} file(s).")
    if dry_run:
        print("Re-run without --dry-run to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate collected GPU power data into an AIC source checkout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--aic-checkout",
        required=True,
        type=Path,
        help="Path to the aiconfigurator source checkout (contains src/aiconfigurator/).",
    )
    parser.add_argument(
        "--h200-data",
        type=Path,
        default=None,
        help="Path to aic_h200_power_data directory. Skip H200 integration if omitted.",
    )
    parser.add_argument(
        "--b200-data",
        type=Path,
        default=None,
        help="Path to aic_b200_power_data directory. Skip B200 integration if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without actually writing any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files at the destination (default: skip).",
    )
    args = parser.parse_args()

    aic_checkout: Path = args.aic_checkout.resolve()
    if not aic_checkout.is_dir():
        print(
            f"ERROR: --aic-checkout directory not found: {aic_checkout}",
            file=sys.stderr,
        )
        sys.exit(1)

    h200_data = args.h200_data.resolve() if args.h200_data else None
    b200_data = args.b200_data.resolve() if args.b200_data else None

    if h200_data is None and b200_data is None:
        print(
            "ERROR: at least one of --h200-data or --b200-data must be specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    integrate(
        aic_checkout,
        h200_data,
        b200_data,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
