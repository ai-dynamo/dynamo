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
INDEX_PATH = REPO_ROOT / "docs/fern/index.yml"
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
DYNAMO_OWNED_DOCUMENTS = (
    "dynamo-integration.md",
    "glm-5-fp8-pareto-sweep.md",
    "planner-goodput-per-gpu-sweep.md",
    "router-end-to-end-latency-sweep.md",
)


def _fern_content(name: str) -> str:
    """Render a canonical document with an edit warning for its Fern copy."""
    source_path = SOURCE_ROOT / name
    content = source_path.read_text()
    frontmatter, separator, body = content.partition("\n---\n")
    if not content.startswith("---\n") or not separator:
        raise ValueError(f"{source_path.relative_to(REPO_ROOT)} has no frontmatter")

    source = source_path.relative_to(REPO_ROOT)
    notice = (
        "<!--\n"
        f"Generated from `{source}` by "
        "`docs/fern/scripts/sync_aisimulate_docs.py`.\n"
        "Edit the canonical source instead of this Fern copy.\n"
        "-->"
    )
    return f"{frontmatter}{separator}\n{notice}\n{body}"


def _integrity_errors() -> list[str]:
    """Find canonical, Fern-copy, and navigation registration drift."""
    errors: list[str] = []
    configured = set(DOCUMENTS)
    canonical = {path.name for path in SOURCE_ROOT.glob("*.md")}
    missing_canonical = configured - canonical
    unregistered_canonical = canonical - configured
    if missing_canonical:
        errors.append(
            "configured canonical documents are missing: "
            + ", ".join(sorted(missing_canonical))
        )
    if unregistered_canonical:
        errors.append(
            "canonical documents are not registered in DOCUMENTS: "
            + ", ".join(sorted(unregistered_canonical))
        )

    expected_destination = configured | set(DYNAMO_OWNED_DOCUMENTS)
    actual_destination = {path.name for path in DESTINATION_ROOT.glob("*.md")}
    unexpected_destination = actual_destination - expected_destination
    missing_dynamo_owned = set(DYNAMO_OWNED_DOCUMENTS) - actual_destination
    if unexpected_destination:
        errors.append(
            "Fern documents are neither canonical copies nor Dynamo-owned: "
            + ", ".join(sorted(unexpected_destination))
        )
    if missing_dynamo_owned:
        errors.append(
            "Dynamo-owned Fern documents are missing: "
            + ", ".join(sorted(missing_dynamo_owned))
        )

    index = INDEX_PATH.read_text()
    for name in sorted(expected_destination):
        path = (DESTINATION_ROOT / name).relative_to(REPO_ROOT / "docs/fern")
        registration = f"path: {path}"
        count = index.count(registration)
        if count != 1:
            errors.append(
                f"{name} has {count} Fern navigation registrations; expected exactly 1"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of updating Fern copies when they are out of sync",
    )
    args = parser.parse_args()

    integrity_errors = _integrity_errors()
    if integrity_errors:
        print("AI Simulate documentation registration is invalid:", file=sys.stderr)
        for error in integrity_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    stale = []
    for name in DOCUMENTS:
        destination = DESTINATION_ROOT / name
        content = _fern_content(name)
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
