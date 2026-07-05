#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require the active BFCL environment to equal the committed constraints lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def requirement_lines(text: str) -> list[str]:
    return sorted(
        (
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ),
        key=str.casefold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--freeze-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    lock_bytes = args.lock.read_bytes()
    expected = requirement_lines(lock_bytes.decode())
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observed = requirement_lines(result.stdout)
    if [line.casefold() for line in observed] != [line.casefold() for line in expected]:
        expected_set = {line.casefold(): line for line in expected}
        observed_set = {line.casefold(): line for line in observed}
        missing = [
            expected_set[key]
            for key in sorted(expected_set.keys() - observed_set.keys())
        ]
        unexpected = [
            observed_set[key]
            for key in sorted(observed_set.keys() - expected_set.keys())
        ]
        raise SystemExit(
            f"BFCL environment differs from constraints.lock; "
            f"missing={missing}, unexpected={unexpected}"
        )

    freeze = "\n".join(observed) + "\n"
    args.freeze_output.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_output.write_text(freeze)
    metadata = {
        "schema_version": 1,
        "constraints_sha256": sha256(lock_bytes),
        "freeze_sha256": sha256(freeze.encode()),
        "package_count": len(observed),
        "python": sys.version.split()[0],
    }
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
