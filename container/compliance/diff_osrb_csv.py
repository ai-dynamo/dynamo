#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff two new-format OSRB dependency CSVs into a change-typed diff CSV.

The OSRB CSV is the cross-ecosystem ``osrb-deps.csv`` written by
``compliance.generators.common.write_merged_csv`` — columns::

    ecosystem,name,version,spdx,source_url,notes

This tool compares a *current* CSV against a *base* CSV and emits a diff CSV
that notes exactly four kinds of change: additions, version bumps, license
changes, and removals (a single dependency may be both a version bump and a
license change). It never fails on a missing base: when ``--base`` is omitted or
empty, it writes a single ``baseline_unavailable`` row carrying ``--baseline-label``
so a diff file always ships alongside the full CSV.

Usage:
    diff_osrb_csv.py --current cur.csv --base base.csv --output out.diff.csv
    diff_osrb_csv.py --current cur.csv --output out.diff.csv \\
        --baseline-label "release baseline v1.2.1 (artifact expired)"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Diff CSV column order. Kept stable so downstream OSRB tooling can rely on it.
DIFF_FIELDNAMES = [
    "change",
    "ecosystem",
    "name",
    "old_version",
    "new_version",
    "old_spdx",
    "new_spdx",
    "source_url",
    "notes",
]

# ``change`` values, listed in the primary sort order requested by OSRB:
# additions first, then version bumps that also change license, then plain
# version bumps, then license-only changes, then removals.
ADDED = "added"
VERSION_AND_LICENSE_CHANGED = "version_and_license_changed"
VERSION_CHANGED = "version_changed"
LICENSE_CHANGED = "license_changed"
REMOVED = "removed"
BASELINE_UNAVAILABLE = "baseline_unavailable"

CHANGE_ORDER = {
    ADDED: 0,
    VERSION_AND_LICENSE_CHANGED: 1,
    VERSION_CHANGED: 2,
    LICENSE_CHANGED: 3,
    REMOVED: 4,
    BASELINE_UNAVAILABLE: 5,
}

# Secondary sort: ecosystem "type" in this fixed order, then name alphabetically.
ECOSYSTEM_ORDER = {"rust": 0, "python": 1, "go": 2, "dpkg": 3, "native": 4}


def read_osrb_csv(path: Path) -> list[dict[str, str]]:
    """Read an OSRB deps CSV into a list of row dicts (empty list if absent)."""
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "ecosystem": (row.get("ecosystem") or "").strip(),
                    "name": (row.get("name") or "").strip(),
                    "version": (row.get("version") or "").strip(),
                    "spdx": (row.get("spdx") or "").strip(),
                    "source_url": (row.get("source_url") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
        return rows


def _by_name(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """Group rows by (ecosystem, name) -> {version: row}.

    Keying by name (not version) lets a version bump surface as a change rather
    than an add+remove pair. Packages shipped at multiple versions keep every
    version under the same key so nothing is silently dropped.
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["ecosystem"], row["name"])
        grouped.setdefault(key, {})[row["version"]] = row
    return grouped


def _added_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "change": ADDED,
        "ecosystem": row["ecosystem"],
        "name": row["name"],
        "old_version": "",
        "new_version": row["version"],
        "old_spdx": "",
        "new_spdx": row["spdx"],
        "source_url": row["source_url"],
        "notes": row["notes"],
    }


def _removed_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "change": REMOVED,
        "ecosystem": row["ecosystem"],
        "name": row["name"],
        "old_version": row["version"],
        "new_version": "",
        "old_spdx": row["spdx"],
        "new_spdx": "",
        "source_url": row["source_url"],
        "notes": row["notes"],
    }


def _changed_row(base: dict[str, str], cur: dict[str, str]) -> dict[str, str]:
    """A row present in both sides with a differing version and/or license."""
    version_changed = base["version"] != cur["version"]
    license_changed = base["spdx"] != cur["spdx"]
    if version_changed and license_changed:
        change = VERSION_AND_LICENSE_CHANGED
    elif version_changed:
        change = VERSION_CHANGED
    else:
        change = LICENSE_CHANGED
    return {
        "change": change,
        "ecosystem": cur["ecosystem"],
        "name": cur["name"],
        "old_version": base["version"],
        "new_version": cur["version"],
        "old_spdx": base["spdx"],
        "new_spdx": cur["spdx"],
        # Prefer the current metadata; fall back to the base's.
        "source_url": cur["source_url"] or base["source_url"],
        "notes": cur["notes"] or base["notes"],
    }


def compute_diff(
    current_rows: list[dict[str, str]],
    base_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compute the change rows between a base and a current OSRB CSV."""
    cur = _by_name(current_rows)
    base = _by_name(base_rows)
    out: list[dict[str, str]] = []

    for key in cur.keys() | base.keys():
        cur_vers = cur.get(key, {})
        base_vers = base.get(key, {})
        if cur_vers and not base_vers:  # name is brand new
            out.extend(_added_row(r) for r in cur_vers.values())
        elif base_vers and not cur_vers:  # name dropped entirely
            out.extend(_removed_row(r) for r in base_vers.values())
        elif set(cur_vers) == set(base_vers):
            # Same version set: only a license move can be a change here.
            for v in cur_vers:
                if cur_vers[v]["spdx"] != base_vers[v]["spdx"]:
                    out.append(_changed_row(base_vers[v], cur_vers[v]))
        elif len(cur_vers) == 1 and len(base_vers) == 1:
            # Clean single-version bump (the common case).
            out.append(
                _changed_row(
                    next(iter(base_vers.values())),
                    next(iter(cur_vers.values())),
                )
            )
        else:
            # Multi-version mismatch: degrade to per-version add/remove, plus
            # license changes for versions shared across both sides.
            for v in set(cur_vers) - set(base_vers):
                out.append(_added_row(cur_vers[v]))
            for v in set(base_vers) - set(cur_vers):
                out.append(_removed_row(base_vers[v]))
            for v in set(cur_vers) & set(base_vers):
                if cur_vers[v]["spdx"] != base_vers[v]["spdx"]:
                    out.append(_changed_row(base_vers[v], cur_vers[v]))

    return sort_diff_rows(out)


def sort_diff_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort by change bucket, then ecosystem type, then name (case-insensitive)."""
    return sorted(
        rows,
        key=lambda r: (
            CHANGE_ORDER.get(r["change"], len(CHANGE_ORDER)),
            ECOSYSTEM_ORDER.get(r["ecosystem"], len(ECOSYSTEM_ORDER)),
            r["name"].lower(),
            r["new_version"] or r["old_version"],
        ),
    )


def baseline_unavailable_rows(label: str) -> list[dict[str, str]]:
    """A single marker row emitted when no base CSV is available to diff against."""
    return [
        {
            "change": BASELINE_UNAVAILABLE,
            "ecosystem": "",
            "name": label or "baseline unavailable",
            "old_version": "",
            "new_version": "",
            "old_spdx": "",
            "new_spdx": "",
            "source_url": "",
            "notes": "no baseline CSV was available to diff against",
        }
    ]


def write_diff_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the diff rows to output_path (stable byte-order)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DIFF_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff two new-format OSRB dependency CSVs into a change-typed diff CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--current", required=True, help="Path to the current commit's OSRB CSV"
    )
    parser.add_argument(
        "--base",
        default="",
        help="Path to the baseline OSRB CSV (omit/empty for the baseline_unavailable marker)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to write the diff CSV"
    )
    parser.add_argument(
        "--baseline-label",
        default="",
        help="Human label for the baseline (used in the baseline_unavailable marker row)",
    )
    args = parser.parse_args()

    current_path = Path(args.current)
    if not current_path.is_file():
        print(f"ERROR: --current does not exist: {current_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    base_path = Path(args.base) if args.base else None

    if base_path is None or not base_path.is_file():
        reason = "no --base given" if base_path is None else f"missing: {base_path}"
        print(
            f"Baseline unavailable ({reason}); writing marker diff.", file=sys.stderr
        )
        write_diff_csv(baseline_unavailable_rows(args.baseline_label), output_path)
        return

    rows = compute_diff(read_osrb_csv(current_path), read_osrb_csv(base_path))
    write_diff_csv(rows, output_path)
    print(f"Wrote {len(rows)} diff rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
