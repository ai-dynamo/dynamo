#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SBOM-diff verifier.

Cross-checks the inline-compliance pipeline's per-image output against the
base-image SBOM corpus. Catches two classes of drift:

  (1) Packages we ADDED to a runtime image (vs its base) that are NOT
      enumerated in /legal/NOTICES-*.txt. This means the NOTICES generator
      missed something — e.g., pip-licenses missed a wheel that was
      vendored, or syft saw a dpkg the dpkg generator's dpkg-query call
      missed. Either way it's a generation bug; fail the build.

  (2) Packages enumerated in NOTICES that are NOT actually shipped in the
      target image (or the base). Means the generator is reading stale
      data — e.g., copying a NOTICES from a different image, or a
      dist-info dir orphaned in site-packages from a previous install.

Wired into the compliance-extract action (.github/actions/compliance-extract)
as a final step after extracting /legal and /sboms artifacts from the inline
build. Failure blocks the PR.

Inputs:
  --target-sboms-dir   /tmp/sboms/   (per-ecosystem deps CSVs from --target sboms)
  --target-notices-dir /tmp/legal/   (NOTICES-*.txt from --target legal)
  --base-sbom          path to a slim CycloneDX JSON in
                       container/compliance/base_sboms/

Output: prints a structured report; exit non-zero on any drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PkgKey:
    ecosystem: str
    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.ecosystem}:{self.name}@{self.version}"


# ---- Target inputs --------------------------------------------------------------


def parse_target_csvs(sboms_dir: Path) -> set[PkgKey]:
    """Read every <ecosystem>-deps.csv under sboms_dir, return the unique set."""
    seen: set[PkgKey] = set()
    csv_files = sorted(sboms_dir.rglob("*-deps.csv"))
    if not csv_files:
        # Fail hard: an empty target set makes the diff pass vacuously, hiding a
        # generator/extraction failure as a clean result.
        raise FileNotFoundError(
            f"no *-deps.csv found under {sboms_dir} — the compliance generators "
            "produced no attribution CSVs; refusing to verify a vacuous diff"
        )
    for csv_path in csv_files:
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(PkgKey(row["ecosystem"], row["name"], row["version"]))
    logger.info(
        "Target: %d unique packages across %d ecosystems", len(seen), len(csv_files)
    )
    return seen


# ---- Base-SBOM input ------------------------------------------------------------


# Map CycloneDX `purl` prefix → our ecosystem name. The SBOMs we generate
# come from syft, which emits standard purls.
_PURL_PREFIXES: list[tuple[str, str]] = [
    ("pkg:cargo/", "rust"),
    ("pkg:pypi/", "python"),
    ("pkg:golang/", "go"),
    ("pkg:deb/", "dpkg"),
    ("pkg:rpm/", "dpkg"),  # treat rpm as dpkg-ecosystem for our policy purposes
    ("pkg:alpine/", "dpkg"),
]


def _ecosystem_from_purl(purl: str) -> str | None:
    for prefix, ecosystem in _PURL_PREFIXES:
        if purl.startswith(prefix):
            return ecosystem
    return None


def parse_base_sbom(path: Path) -> set[PkgKey]:
    """Read a CycloneDX 1.5/1.6 JSON, return the set of (ecosystem, name, version)."""
    if not path.is_file():
        raise FileNotFoundError(f"base SBOM not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: set[PkgKey] = set()
    for c in doc.get("components", []) or []:
        purl = c.get("purl") or ""
        ecosystem = _ecosystem_from_purl(purl)
        if ecosystem is None:
            continue
        name = c.get("name")
        version = c.get("version")
        if not name or not version:
            continue
        out.add(PkgKey(ecosystem, name, str(version)))
    logger.info("Base SBOM %s: %d resolvable components", path.name, len(out))
    return out


# ---- NOTICES input --------------------------------------------------------------


# NOTICES-*.txt format (per common.render_notices):
#   ## <name> <version>
#   License: <SPDX>
#   ...
#   (optional license text wrapped in ``` ... ``` fenced blocks)
_NOTICE_HEADER_RE = re.compile(r"^## (.+?) (\S+)\s*$")
_FENCE_RE = re.compile(r"^```")


def parse_notices(notices_dir: Path) -> set[PkgKey]:
    """Read every NOTICES-*.txt under notices_dir, return enumerated (ecosystem, name, version) set.

    The ecosystem is inferred from the filename: NOTICES-Rust.txt → rust,
    NOTICES-Apt.txt → dpkg, etc. (matches the generator's lowercased ecosystem ID).

    Section headers (`## <name> <version>`) live OUTSIDE fenced code blocks;
    license text inside ``` ... ``` blocks can contain its own markdown
    headers (e.g. `## Table of Contents` in long-form licenses) that would
    spuriously match the header regex. Track fence state line-by-line and
    only treat headers outside fences as real section markers.
    """
    out: set[PkgKey] = set()
    notices_files = sorted(notices_dir.rglob("NOTICES-*.txt"))
    if not notices_files:
        logger.warning("No NOTICES-*.txt found under %s", notices_dir)
        return out
    for path in notices_files:
        # NOTICES-Rust.txt → "rust"; NOTICES-Apt.txt → "apt" (a.k.a. "dpkg")
        eco_label = path.stem.removeprefix("NOTICES-").lower()
        ecosystem = "dpkg" if eco_label == "apt" else eco_label
        in_fence = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _NOTICE_HEADER_RE.match(line)
            if m:
                name, version = m.group(1).strip(), m.group(2).strip()
                out.add(PkgKey(ecosystem, name, version))
    logger.info(
        "NOTICES: %d enumerated packages across %d files", len(out), len(notices_files)
    )
    return out


# ---- Diff + report ---------------------------------------------------------------


@dataclass(frozen=True)
class DiffReport:
    target: set[PkgKey]
    base: set[PkgKey]
    notices: set[PkgKey]

    @property
    def added(self) -> set[PkgKey]:
        """Packages in target but not in base — what we contributed on top."""
        return self.target - self.base

    @property
    def missing_from_notices(self) -> set[PkgKey]:
        """Added packages NOT enumerated in NOTICES. Generator bug — fail."""
        return self.added - self.notices

    @property
    def stale_notices(self) -> set[PkgKey]:
        """NOTICES entries that aren't shipped (not in target, not in base).
        Generator read stale data."""
        return self.notices - self.target - self.base


def render_report(report: DiffReport) -> str:
    parts = [
        "SBOM-diff verification report:",
        f"  target packages: {len(report.target)}",
        f"  base packages:   {len(report.base)}",
        f"  added on top:    {len(report.added)}",
        f"  NOTICES entries: {len(report.notices)}",
        "",
    ]
    missing = sorted(
        report.missing_from_notices, key=lambda p: (p.ecosystem, p.name, p.version)
    )
    stale = sorted(report.stale_notices, key=lambda p: (p.ecosystem, p.name, p.version))

    if missing:
        parts.append(
            f"FAIL: {len(missing)} package(s) added vs. base but missing from NOTICES:"
        )
        for k in missing[:200]:  # truncate insanely-long output
            parts.append(f"  - {k}")
        if len(missing) > 200:
            parts.append(f"  ... and {len(missing) - 200} more")
        parts.append("")
    if stale:
        parts.append(
            f"FAIL: {len(stale)} NOTICES entry/entries not present in target SBOM "
            "(stale generator output):"
        )
        for k in stale[:200]:
            parts.append(f"  - {k}")
        if len(stale) > 200:
            parts.append(f"  ... and {len(stale) - 200} more")
        parts.append("")

    if not missing and not stale:
        parts.append(
            "OK: every added package is documented in NOTICES; no stale entries."
        )
    return "\n".join(parts)


# ---- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="container.compliance.verify_sbom_diff",
        description="Cross-check inline-compliance per-image output against the base-image SBOM.",
    )
    parser.add_argument(
        "--target-sboms-dir",
        type=Path,
        required=True,
        help="Directory with per-ecosystem *-deps.csv from `--target sboms` extraction",
    )
    parser.add_argument(
        "--target-notices-dir",
        type=Path,
        required=True,
        help="Directory with NOTICES-*.txt from `--target legal` extraction",
    )
    parser.add_argument(
        "--base-sbom",
        type=Path,
        required=True,
        help="Path to the slim CycloneDX JSON for the FROM base, "
        "from container/compliance/base_sboms/",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    target = parse_target_csvs(args.target_sboms_dir)
    base = parse_base_sbom(args.base_sbom)
    notices = parse_notices(args.target_notices_dir)

    report = DiffReport(target=target, base=base, notices=notices)
    print(render_report(report))

    if report.missing_from_notices or report.stale_notices:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
