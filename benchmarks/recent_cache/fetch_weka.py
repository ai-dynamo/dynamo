# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the seven-play public WEKA subset used in this experiment.

Selection: among the first 32 dataset rows, keep complete, supported plays using
only Claude Opus 4.8 and with a recorded end at most eight hours. This excludes
two same-model plays with long idle periods; it is not a representative sample
of the full corpus. The selected plays retain every request, gap and dependency.

The dataset viewer does not accept a revision parameter. Downloads therefore
require the pinned repository revision both before and after fetching, plus
exact SHA256 matches for every selected serialized play. No credentials are used.

With --copies 4, byte-identical plays get distinct relative filenames. Dynamo's
WEKA importer uses these names to give each copy a private cache namespace.
This increases synthetic demand; copies are not independent observations.
"""

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "semianalysisai/cc-traces-weka-062126-256k"
REVISION = "8fecd2fc56694469f758f0afbbb6335ad3043740"
METADATA_URL = f"https://huggingface.co/api/datasets/{DATASET}"
ROWS_URL = "https://datasets-server.huggingface.co/rows"

EXPECTED_SHA256 = {
    "001-006c98de37d819e95b0840e25426bb7ca99d.json": (
        "7863a55e2b5b5a0e7a0748f327b4b984b3a48097eb49890ee6e933f3ae95c7c0"
    ),
    "007-04dba6fe621301a1d01fd63b8d02c9645c48.json": (
        "256e1756c7436b1ae8cc54b7a60baa9e797acbbe10496af3b90466e2a2ee97e3"
    ),
    "011-069d7bf5f1efb4e76e3c84510367e6af78a5.json": (
        "50f9cd6d3cae309854364fe36f75be259130efa04364afe65855426bfb237fe6"
    ),
    "016-0a279af1bec84c6ab033bb03f5d8bfd6b08d.json": (
        "9d5deba65b9429042fcd454cf7d7578a66c84aef5b2334d4928b52cc77578032"
    ),
    "024-0db32e2f852a2ea24ed616b31975a84da5e4.json": (
        "d2da4b2ac99789cdf92aec12f00dcae721c1a5f6028d776bd3bd210c07fc1fbd"
    ),
    "025-0dedb07b5e4362b028db9d8ab90fff7fd4ad.json": (
        "a1f507951416a06416a8cbae163fbc4578a73ff7a2aecff78d7efe22d9b3a453"
    ),
    "028-0f909e7d63c0d5e1f060d727ada25d910d89.json": (
        "b6e5f825a8cdf0a09a72cf9f6359ced7c7ced3a2cfe350e961ef2e3add748fd0"
    ),
}


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "dynamo-recent-cache-reproducer"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"public dataset request failed: HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("public dataset request failed: connection error") from None
    if not isinstance(result, dict):
        raise ValueError("public dataset response must be a JSON object")
    return result


def check_revision() -> None:
    metadata = fetch_json(METADATA_URL)
    if metadata.get("sha") != REVISION:
        raise ValueError("dataset revision changed; refusing an unpinned download")
    if metadata.get("private") is not False or metadata.get("gated") is not False:
        raise ValueError("dataset must remain public and ungated")


def verify_play(filename: str, content: bytes) -> None:
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256[filename]:
        raise ValueError(f"play SHA256 mismatch at source row {int(filename[:3])}")


def download_plays() -> dict[str, bytes]:
    check_revision()
    plays = {}
    # Fetch the fixed selected offsets, without downloading the rejected 25 plays.
    for filename in EXPECTED_SHA256:
        row_index = int(filename[:3])
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": row_index,
                "length": 1,
            }
        )
        response = fetch_json(f"{ROWS_URL}?{query}")
        rows = response.get("rows", [])
        if len(rows) != 1 or rows[0].get("row_idx") != row_index:
            raise ValueError(f"dataset row missing or reordered: {row_index}")
        entry = rows[0]
        if entry.get("truncated_cells") != []:
            raise ValueError(f"dataset row truncated: {row_index}")
        row = entry.get("row")
        if not isinstance(row, dict) or row.get("id") != Path(filename).stem[4:]:
            raise ValueError(f"dataset row identity mismatch: {row_index}")
        content = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        verify_play(filename, content)
        plays[filename] = content
    check_revision()
    return plays


def read_plays(source: Path) -> dict[str, bytes]:
    plays = {}
    for filename in EXPECTED_SHA256:
        content = (source / filename).read_bytes()
        verify_play(filename, content)
        plays[filename] = content
    return plays


def validate_output(output: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    repository = Path(result.stdout.strip()).resolve()
    output = output.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists")
    resolved = output.resolve()
    if resolved.is_relative_to(repository):
        raise ValueError("trace output must be outside the repository")
    return resolved


def write_plays(output: Path, plays: dict[str, bytes], copies: int) -> int:
    output.mkdir(parents=True, exist_ok=False)
    trace_directory = output / "plays"
    trace_directory.mkdir()
    files = []
    for filename, content in plays.items():
        for copy_index in range(copies):
            target = (
                filename
                if copies == 1
                else f"{Path(filename).stem}-copy{copy_index:02d}.json"
            )
            (trace_directory / target).write_bytes(content)
            files.append(
                {
                    "path": f"plays/{target}",
                    "source_row": int(filename[:3]),
                    "copy_index": copy_index,
                    "sha256": EXPECTED_SHA256[filename],
                    "bytes": len(content),
                }
            )
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "selection": {
            "initial_rows": 32,
            "source_rows": [int(name[:3]) for name in EXPECTED_SHA256],
            "model": "claude-opus-4-8",
            "maximum_recorded_end_seconds": 28800,
            "whole_plays": True,
            "gap_cap_seconds": None,
        },
        "serialization": "json.dumps(row, separators=(',', ':')) + newline",
        "copies": copies,
        "duplication": (
            "none"
            if copies == 1
            else "byte-identical plays with distinct filenames for importer-private "
            "cache namespaces; synthetic demand, not independent observations"
        ),
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory outside the repository; replay its plays/ subdirectory",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="reuse seven previously downloaded play files, with exact hash checks",
    )
    parser.add_argument(
        "--copies",
        type=int,
        choices=(1, 4),
        default=1,
        help="1 original subset or 4 synthetic private copies per play (default: 1)",
    )
    args = parser.parse_args()
    try:
        output = validate_output(args.output_dir)
        plays = (
            read_plays(args.source_dir.expanduser())
            if args.source_dir is not None
            else download_plays()
        )
        count = write_plays(output, plays, args.copies)
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except (OSError, subprocess.CalledProcessError):
        parser.exit(1, "error: filesystem or repository access failed\n")
    print(json.dumps({"verified_source_plays": len(plays), "output_plays": count}))


if __name__ == "__main__":
    main()
