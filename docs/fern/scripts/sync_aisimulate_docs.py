#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync canonical AI Simulate documentation into the Dynamo Fern site."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "aisimulate/docs/sweeper"
DESTINATION_ROOT = (
    REPO_ROOT
    / "docs/fern/pages/developer-guide/knowledge-base/modular-components"
    / "ai-simulate-experimental/sweeper-experimental"
)
DOCUMENTS = (
    "overview.md",
    "quickstart.md",
    "tutorial.md",
    "architecture.md",
    "configuration.md",
    "traffic.md",
    "optimization-goals.md",
    "results.md",
    "sweep-config-provider.md",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of updating Fern copies when they are out of sync",
    )
    args = parser.parse_args()

    stale = []
    for name in DOCUMENTS:
        source = SOURCE_ROOT / name
        destination = DESTINATION_ROOT / name
        content = source.read_text()
        if destination.exists() and destination.read_text() == content:
            continue
        stale.append(destination.relative_to(REPO_ROOT))
        if not args.check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)

    if args.check and stale:
        print("AI Simulate Fern copies are out of sync:", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        print(
            "Run `python3 docs/fern/scripts/sync_aisimulate_docs.py`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
