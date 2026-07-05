#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize a freeze and verify it exactly against the SWE constraints lock."""

from __future__ import annotations

import argparse
from pathlib import Path


EDITABLE_REPOS = {"mini-swe-agent", "SWE-bench"}


def normalized_lock(path: Path) -> list[str]:
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(line.startswith("-e ") for line in lines):
        raise ValueError("constraints lock must not contain editable requirements")
    if len(lines) != len(set(lines)):
        raise ValueError("constraints lock contains duplicate requirements")
    return sorted(lines)


def normalized_freeze(path: Path) -> list[str]:
    requirements = []
    editable_repos = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-e file://"):
            editable_repos.append(Path(line.removeprefix("-e file://")).name)
        elif line.startswith("-e "):
            raise ValueError(f"unsupported editable freeze entry: {line}")
        else:
            requirements.append(line)
    if set(editable_repos) != EDITABLE_REPOS or len(editable_repos) != 2:
        raise ValueError(
            "freeze must contain exactly the mini-swe-agent and SWE-bench editables"
        )
    if len(requirements) != len(set(requirements)):
        raise ValueError("freeze contains duplicate requirements")
    return sorted(requirements)


def verify(lock: Path, freeze: Path) -> list[str]:
    expected = normalized_lock(lock)
    actual = normalized_freeze(freeze)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "normalized environment freeze differs from constraints lock; "
            f"missing={missing}, extra={extra}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    normalized = verify(args.lock, args.freeze)
    args.output.write_text("\n".join(normalized) + "\n")


if __name__ == "__main__":
    main()
