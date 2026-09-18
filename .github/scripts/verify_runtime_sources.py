# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that a runtime image tests the requested Dynamo Python sources."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from zipfile import ZipFile


def compare_sources(repo: Path, installed: Path, wheel: Path) -> dict:
    checked = 0
    mismatches = []
    with ZipFile(wheel) as archive:
        for name in sorted(archive.namelist()):
            path = Path(name)
            if path.suffix != ".py" or path.name == "_version.py":
                continue
            source = repo / "components/src" / path
            if not source.is_file():
                continue
            checked += 1
            installed_path = installed / path
            payloads = {
                "source": source.read_bytes(),
                "wheel": archive.read(name),
                "installed": (
                    installed_path.read_bytes() if installed_path.is_file() else None
                ),
            }
            hashes = {
                key: hashlib.sha256(value).hexdigest() if value is not None else None
                for key, value in payloads.items()
            }
            if len(set(hashes.values())) != 1:
                mismatches.append({"path": name, **hashes})
    if not checked:
        raise RuntimeError("No Dynamo component sources found in the shipped wheel")
    return {"checked": checked, "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--wheelhouse", type=Path, default=Path("/opt/dynamo/wheelhouse")
    )
    args = parser.parse_args()
    wheels = list(args.wheelhouse.glob("ai_dynamo-*-py3-none-any.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one ai-dynamo wheel, found {len(wheels)}")
    distribution = importlib.metadata.distribution("ai-dynamo")
    result = compare_sources(args.repo, Path(distribution.locate_file("")), wheels[0])
    result["expected_revision"] = args.expected_sha
    result["image_revision"] = os.environ.get("DYNAMO_COMMIT_SHA")
    result["wheel"] = str(wheels[0])
    print(json.dumps(result, indent=2))
    return int(
        bool(result["mismatches"])
        or result["image_revision"] != result["expected_revision"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
