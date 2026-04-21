#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert BuildKit TSV extraction output to attribution CSV files.

Reads dpkg.tsv and python.tsv from a target extraction directory, writes a
sorted CSV, and optionally computes a diff against a base extraction directory.

Usage:
    python process_results.py --target-dir <dir> --output attribution.csv
    python process_results.py --target-dir <dir> --base-dir <dir> --output attribution.csv
    # Produces: attribution.csv and attribution_diff.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def read_tsv(tsv_path: Path, pkg_type: str) -> list[dict[str, str]]:
    """Parse a tab-separated extraction output file into package dicts."""
    packages = []
    if not tsv_path.is_file():
        return packages
    for line in tsv_path.read_text(errors="replace").strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        pkg_name, version, spdx_license = parts
        packages.append(
            {
                "package_name": pkg_name,
                "version": version,
                "type": pkg_type,
                "spdx_license": spdx_license,
            }
        )
    return packages


def read_extraction_dir(directory: Path) -> list[dict[str, str]]:
    """Read dpkg.tsv and python.tsv from an extraction directory."""
    packages = read_tsv(directory / "dpkg.tsv", "dpkg")
    packages += read_tsv(directory / "python.tsv", "python")
    return packages


def compute_diff(
    target_packages: list[dict[str, str]],
    base_packages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return packages in target that are new or have a different version vs base."""
    base_lookup = {
        (pkg["package_name"], pkg["type"]): pkg["version"] for pkg in base_packages
    }
    return [
        pkg
        for pkg in target_packages
        if base_lookup.get((pkg["package_name"], pkg["type"])) != pkg["version"]
    ]


def write_csv(packages: list[dict[str, str]], output_path: Path | None) -> None:
    """Write packages to CSV sorted by (type, package_name)."""
    sorted_packages = sorted(packages, key=lambda p: (p["type"], p["package_name"]))
    fieldnames = ["package_name", "version", "type", "spdx_license"]
    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_packages)
        print(f"Wrote {len(sorted_packages)} entries to {output_path}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_packages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BuildKit TSV extraction output to attribution CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target-dir ./output --output attribution.csv
  %(prog)s --target-dir ./output --base-dir ./base-output --output attribution.csv
        """,
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory containing dpkg.tsv and python.tsv from the target image extraction",
    )
    parser.add_argument(
        "--base-dir",
        help="Directory containing dpkg.tsv and python.tsv from the base image extraction (enables diff output)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV file path (default: stdout)",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        print(f"ERROR: --target-dir does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)

    target_packages = read_extraction_dir(target_dir)
    if not target_packages:
        print(f"ERROR: no packages found in {target_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    write_csv(target_packages, output_path)

    if args.base_dir:
        base_dir = Path(args.base_dir)
        if not base_dir.is_dir():
            print(f"ERROR: --base-dir does not exist: {base_dir}", file=sys.stderr)
            sys.exit(1)
        base_packages = read_extraction_dir(base_dir)
        diff_packages = compute_diff(target_packages, base_packages)
        print(
            f"Diff: {len(diff_packages)} new/changed packages vs base", file=sys.stderr
        )

        if output_path:
            diff_path = output_path.with_stem(output_path.stem + "_diff")
            write_csv(diff_packages, diff_path)
        else:
            print("\n# --- DIFF (new/changed packages vs base) ---", file=sys.stderr)
            write_csv(diff_packages, None)


if __name__ == "__main__":
    main()
