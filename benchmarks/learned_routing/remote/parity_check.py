# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field-by-field parity of remote lr-eval results against local ones (stdlib only).

    parity_check.py --remote DIR [DIR...] --local DIR [DIR...] [--out parity.json]

Each DIR is a result-cache directory (``.../runs/cache/results``) or a directory containing one.
Every remote record is matched to the local record with the same cache key, and every field is
compared for exact equality (JSON values, floats bit-for-bit through their repr), except the
fields that describe where or how fast the replay ran (``ENVIRONMENT_FIELDS``). Path fields that
end in a content-addressed file name (``CONTENT_PATH_FIELDS``) are compared by file name. The
gzip per-request rows of both sides are decompressed and compared byte for byte, besides their
``per_request_canonical_sha256`` (records kept without rows on both sides, as lr-train does by
default, are counted as not compared). Exit 0 iff every remote record has a local counterpart
with no differing field and no differing per-request rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ENVIRONMENT_FIELDS = frozenset(
    {
        "cached",
        "eval_wall_s",
        "ingested_from",
        "native_wall_time_ms",
        "per_request_path",
        "run_id",
        "slot",
        "wall_s",
        "worker_peak_rss_mib",
        "worker_pid",
    }
)
CONTENT_PATH_FIELDS = frozenset({"policy_spec_path", "policy_yaml"})


def results_dir(path: Path) -> Path:
    for candidate in (
        path / "runs" / "cache" / "results",
        path / "cache" / "results",
        path / "results",
        path,
    ):
        if candidate.name == "results" and candidate.is_dir():
            return candidate
    raise SystemExit(f"{path}: no runs/cache/results directory found")


def load(dirs: list[Path]) -> dict[str, tuple[dict, Path]]:
    records = {}
    for d in dirs:
        for path in results_dir(d).glob("??/*.json"):
            records[path.stem] = (json.loads(path.read_text()), path)
    return records


def lookup(dirs: list[Path], key: str) -> tuple[dict, Path] | None:
    """Local records are looked up by key (the campaign cache is too large to load whole)."""
    for d in dirs:
        path = results_dir(d) / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text()), path
    return None


def per_request_bytes(record_path: Path) -> bytes | None:
    gz = record_path.with_name(record_path.stem + ".per_request.jsonl.gz")
    if not gz.exists():
        return None
    return gzip.decompress(gz.read_bytes())


def compare(remote: dict, local: dict) -> list[str]:
    diffs = []
    for key in sorted(set(remote) | set(local)):
        if key in ENVIRONMENT_FIELDS:
            continue
        a, b = remote.get(key, "<missing>"), local.get(key, "<missing>")
        if key in CONTENT_PATH_FIELDS and isinstance(a, str) and isinstance(b, str):
            a, b = Path(a).name, Path(b).name
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            diffs.append(key)
    return diffs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--remote", nargs="+", required=True, type=Path)
    p.add_argument("--local", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--exclude-ingested",
        action="store_true",
        help="treat local records that were ingested from a bundle as missing (not independent)",
    )
    args = p.parse_args(argv)
    remote = load(args.remote)
    rows, unmatched = [], []
    fields_compared: set[str] = set()
    for key, (record, path) in sorted(remote.items()):
        found = lookup(args.local, key)
        if (
            found is not None
            and args.exclude_ingested
            and found[0].get("ingested_from")
        ):
            found = None
        if found is None:
            unmatched.append(
                {
                    "key": key,
                    "cell_id": record.get("cell_id"),
                    "policy": record.get("policy_name"),
                    "k": record.get("repeat"),
                }
            )
            continue
        other, other_path = found
        diffs = compare(record, other)
        fields_compared |= {k for k in record if k not in ENVIRONMENT_FIELDS}
        rb, lb = per_request_bytes(path), per_request_bytes(other_path)
        if rb is None or lb is None:
            per_request = "missing on " + (
                "both"
                if rb is None and lb is None
                else "remote"
                if rb is None
                else "local"
            )
        else:
            per_request = "identical" if rb == lb else "DIFFERENT"
        rows.append(
            {
                "key": key,
                "cell_id": record.get("cell_id"),
                "policy": record.get("policy_name"),
                "k": record.get("repeat"),
                "num_workers": record.get("num_workers"),
                "family": record.get("family"),
                "fields_differing": diffs,
                "per_request_bytes": per_request,
                "per_request_rows": record.get("per_request_rows"),
                "per_request_sha256": record.get("per_request_canonical_sha256"),
                "per_request_gunzip_sha256_remote": hashlib.sha256(rb).hexdigest()
                if rb
                else None,
                "goodput_rps_window": record.get("goodput_rps_window"),
                "wall_s_remote": record.get("wall_s"),
                "wall_s_local": other.get("wall_s"),
            }
        )
    # lr-train keeps no per-request rows by default, so rows missing on one or both sides are
    # "not compared" (counted), not a mismatch; differing rows are.
    identical = [
        r
        for r in rows
        if not r["fields_differing"] and r["per_request_bytes"] != "DIFFERENT"
    ]
    summary = {
        "remote_records": len(remote),
        "matched": len(rows),
        "unmatched_remote": unmatched,
        "records_identical": len(identical),
        "records_with_field_diffs": [r for r in rows if r["fields_differing"]],
        "per_request_identical": sum(
            r["per_request_bytes"] == "identical" for r in rows
        ),
        "per_request_not_compared": sum(
            r["per_request_bytes"].startswith("missing") for r in rows
        ),
        "per_request_mismatch": [
            r for r in rows if r["per_request_bytes"] == "DIFFERENT"
        ],
        "fields_compared": sorted(fields_compared),
        "environment_fields_excluded": sorted(ENVIRONMENT_FIELDS),
        "content_path_fields_compared_by_name": sorted(CONTENT_PATH_FIELDS),
        "per_request_rows_compared": sum(
            r["per_request_rows"] or 0
            for r in rows
            if r["per_request_bytes"] == "identical"
        ),
        "exclude_ingested": args.exclude_ingested,
        "remote_dirs": [str(d) for d in args.remote],
        "local_dirs": [str(d) for d in args.local],
        "rows": rows,
    }
    ok = not unmatched and len(identical) == len(rows) and rows
    summary["verdict"] = "IDENTICAL" if ok else "MISMATCH"
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    keys = (
        "verdict",
        "remote_records",
        "matched",
        "records_identical",
        "per_request_identical",
        "per_request_not_compared",
        "per_request_rows_compared",
    )
    print(
        json.dumps(
            {k: summary[k] for k in keys}
            | {
                "unmatched": len(unmatched),
                "field_diff_records": len(summary["records_with_field_diffs"]),
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
