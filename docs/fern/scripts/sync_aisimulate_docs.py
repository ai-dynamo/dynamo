# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync canonical AI Simulate documentation into the Dynamo Fern site."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = REPO_ROOT / "docs/fern/index.yml"
FERN_AI_SIMULATE_ROOT = (
    REPO_ROOT
    / "docs/fern/pages/developer-guide/knowledge-base/modular-components"
    / "ai-simulate-experimental"
)


@dataclass(frozen=True)
class DocumentSet:
    source_root: Path
    destination_root: Path
    documents: tuple[str, ...]
    dynamo_owned_documents: tuple[str, ...] = ()


DOCUMENT_SETS = (
    DocumentSet(
        source_root=REPO_ROOT / "aisimulate/docs/engine",
        destination_root=FERN_AI_SIMULATE_ROOT / "engine-experimental",
        documents=("architecture.md",),
    ),
    DocumentSet(
        source_root=REPO_ROOT / "aisimulate/docs/replay",
        destination_root=FERN_AI_SIMULATE_ROOT / "replayer-experimental",
        documents=("architecture.md", "cli-reference.md"),
    ),
    DocumentSet(
        source_root=REPO_ROOT / "aisimulate/docs/sweeper",
        destination_root=FERN_AI_SIMULATE_ROOT / "sweeper-experimental",
        documents=(
            "overview.md",
            "quickstart.md",
            "tutorial.md",
            "architecture.md",
            "configuration.md",
            "traffic.md",
            "optimization-goals.md",
            "results.md",
            "sweep-config-provider.md",
        ),
        dynamo_owned_documents=(
            "dynamo-integration.md",
            "glm-5-fp8-pareto-sweep.md",
            "planner-goodput-per-gpu-sweep.md",
            "router-end-to-end-latency-sweep.md",
        ),
    ),
)


def _fern_content(document_set: DocumentSet, name: str) -> str:
    """Render a canonical document with an edit warning for its Fern copy."""
    source_path = document_set.source_root / name
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
    index = INDEX_PATH.read_text()

    for document_set in DOCUMENT_SETS:
        configured = set(document_set.documents)
        canonical = {path.name for path in document_set.source_root.glob("*.md")}
        missing_canonical = configured - canonical
        unregistered_canonical = canonical - configured
        if missing_canonical:
            errors.append(
                f"{document_set.source_root.relative_to(REPO_ROOT)} is missing "
                "configured documents: " + ", ".join(sorted(missing_canonical))
            )
        if unregistered_canonical:
            errors.append(
                f"{document_set.source_root.relative_to(REPO_ROOT)} has documents "
                "not registered in DOCUMENT_SETS: "
                + ", ".join(sorted(unregistered_canonical))
            )

        expected_destination = configured | set(document_set.dynamo_owned_documents)
        actual_destination = {
            path.name for path in document_set.destination_root.glob("*.md")
        }
        unexpected_destination = actual_destination - expected_destination
        missing_dynamo_owned = (
            set(document_set.dynamo_owned_documents) - actual_destination
        )
        if unexpected_destination:
            errors.append(
                f"{document_set.destination_root.relative_to(REPO_ROOT)} has documents "
                "that are neither canonical copies nor Dynamo-owned: "
                + ", ".join(sorted(unexpected_destination))
            )
        if missing_dynamo_owned:
            errors.append(
                f"{document_set.destination_root.relative_to(REPO_ROOT)} is missing "
                "Dynamo-owned documents: " + ", ".join(sorted(missing_dynamo_owned))
            )

        for name in sorted(expected_destination):
            path = (document_set.destination_root / name).relative_to(
                REPO_ROOT / "docs/fern"
            )
            registration = f"path: {path}"
            count = index.count(registration)
            if count != 1:
                errors.append(
                    f"{name} has {count} Fern navigation registrations; "
                    "expected exactly 1"
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
    for document_set in DOCUMENT_SETS:
        for name in document_set.documents:
            destination = document_set.destination_root / name
            content = _fern_content(document_set, name)
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
