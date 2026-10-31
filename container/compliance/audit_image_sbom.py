# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completeness audit: syft scan of the shipped image vs our attribution CSV.

The inline-compliance generators attribute what they can see (Python dist-info,
Rust/Go SBOMs, dpkg metadata, the native_packages.yaml overlay). This script is
the INDEPENDENT cross-check the older verify_sbom_diff never was: it takes an
authoritative full syft scan of the FINAL pushed image, subtracts the baseline
SBOM (what the base owns), and reports any delta component that does NOT appear
in our unified osrb-deps.csv — i.e. something shipped that we failed to attribute
(a scan-location gap, a binary syft sees that our ecosystems don't, etc.).

Inputs:
  --syft-sbom    CycloneDX JSON from `syft <image> -o cyclonedx-json`
  --syft-json    optional native Syft JSON with package file ownership
  --base-sbom    baseline CycloneDX (container/compliance/base_sboms/<stem>-<arch>.cdx.json)
  --our-csv      the build's unified osrb-deps.csv
  --image        image ref/label, for the report header (optional)
  --report       optional path to write the unattributed list (one per line)
  --fail-on-findings  exit non-zero if any unattributed component is found
                      (default: report-only / exit 0 for local investigations;
                      CI passes --fail-on-findings)

Name matching is ecosystem-scoped and version-agnostic. Python names use
PEP 503 normalization; other ecosystems retain their package names. Generic
binary identities must have native rows, not an unrelated Rust/Python name.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from posixpath import normpath

from .identity import component_identity, identity

logger = logging.getLogger("compliance.audit_image_sbom")


def _normalize(name: str) -> str:
    """Lowercase + unify separators — a coarse key that absorbs common
    dist-vs-import / packaging name skew (e.g. sgl_kernel vs sglang-kernel)
    without over-merging."""
    return (name or "").strip().lower().replace("_", "-").replace(" ", "-")


def _load_cdx_components(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("components", []) or []


def _name_version_keys(components: list[dict]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for c in components:
        name, version = c.get("name"), c.get("version")
        if name and version is not None:
            keys.add((name, str(version)))
    return keys


def _csv_names(csv_path: Path) -> set[str]:
    names: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("name"):
                names.add(_normalize(row["name"]))
    return names


def _csv_identities(csv_path: Path) -> set[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {
            identity(row.get("ecosystem", "native"), row["name"])
            for row in csv.DictReader(stream)
            if row.get("name")
        }


def _purl_type(purl: str | None) -> str:
    if not purl or not purl.startswith("pkg:"):
        return "?"
    return purl[len("pkg:") :].split("/", 1)[0]


# ai-dynamo-owned crates — in-repo (dynamo-/kvbm- + the lib/runtime/examples/*
# crates) or separate ai-dynamo releases (aisimulate, nixl, velo).
# They're first-party, not third-party, so they're absent from osrb-deps.csv by
# design — skip them instead of flagging them as attribution gaps. Keep roughly
# in sync with collect_sources._FIRST_PARTY_* (which feeds source archival).
_FIRST_PARTY_PREFIXES = ("aisimulate-", "dynamo-", "kvbm-", "nixl-", "velo-")
_FIRST_PARTY_GO_PREFIXES = ("github.com/ai-dynamo/",)
_FIRST_PARTY_NAMES = {"service-metrics", "system-metrics", "hello-world", "velo"}


def _is_first_party(name: str) -> bool:
    norm = _normalize(name)
    if norm in _FIRST_PARTY_NAMES or norm.startswith(_FIRST_PARTY_PREFIXES):
        return True
    return name.startswith(_FIRST_PARTY_GO_PREFIXES)


def _rootfs_path(path: str) -> str:
    """Use one root-relative representation for Syft and package-owned paths."""
    return normpath("/" + path.lstrip("/"))


def _locations(component: dict) -> set[str]:
    return {
        _rootfs_path(p["value"])
        for p in component.get("properties", [])
        if p["name"].startswith("syft:location:") and p["name"].endswith(":path")
    }


def _ownership_coverage(native: dict, our_names: set[str]) -> dict[str, set[str]]:
    """Known owning-package records; never infer coverage from directory alone."""
    owners: dict[str, set[str]] = {}
    for artifact in native.get("artifacts", []):
        name = _normalize(artifact["name"])
        if name not in our_names or artifact["type"] != "deb":
            continue
        for item in artifact.get("metadata", {}).get("files", []):
            owners.setdefault(_rootfs_path(item["path"]), set()).add(name)
    return owners


def _covered_by_owner(component: dict, owners: dict[str, set[str]]) -> bool:
    locations = _locations(component)
    if not locations:
        return False
    # Python metadata is covered under its owning dpkg package. Do not use a
    # parent wheel to hide vendored distributions with independent licenses.
    if _purl_type(component.get("purl")) == "pypi" and all(
        path.startswith("/usr/lib/python3/dist-packages/") for path in locations
    ):
        return all(
            any(owner.startswith("python3-") for owner in owners.get(path, set()))
            for path in locations
        )
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compliance.audit_image_sbom")
    p.add_argument("--syft-sbom", type=Path, required=True)
    p.add_argument(
        "--syft-json",
        type=Path,
        default=None,
        help="Native Syft JSON with package file ownership",
    )
    p.add_argument("--base-sbom", type=Path, default=None)
    p.add_argument("--our-csv", type=Path, required=True)
    p.add_argument("--image", default="")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--fail-on-findings", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    syft_components = _load_cdx_components(args.syft_sbom)
    baseline_keys: set[tuple[str, str]] = set()
    if args.base_sbom and args.base_sbom.is_file():
        baseline_keys = _name_version_keys(_load_cdx_components(args.base_sbom))
    elif args.base_sbom:
        logger.warning(
            "baseline SBOM not found: %s (auditing full image)", args.base_sbom
        )

    our_names = _csv_names(args.our_csv)
    our_identities = _csv_identities(args.our_csv)
    owners = (
        _ownership_coverage(json.loads(args.syft_json.read_text()), our_names)
        if args.syft_json
        else {}
    )
    owner_matched = 0

    # syft_delta = image components above the baseline (exact name+version, as
    # the generators subtract). Then flag any whose normalized name is absent
    # from our attribution.
    unattributed: list[dict] = []
    delta = 0
    for c in syft_components:
        name, version = c.get("name"), c.get("version")
        if not name or version is None:
            continue
        if (name, str(version)) in baseline_keys:
            continue
        delta += 1
        if component_identity(c) not in our_identities and not _is_first_party(name):
            if _covered_by_owner(c, owners):
                owner_matched += 1
                continue
            unattributed.append(c)

    # De-dup by normalized name for the report.
    seen: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    for c in sorted(unattributed, key=lambda c: _normalize(c.get("name", ""))):
        key = _normalize(c.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            (c.get("name", ""), str(c.get("version", "")), _purl_type(c.get("purl")))
        )

    hdr = f"Compliance completeness audit{' for ' + args.image if args.image else ''}"
    print(hdr)
    print(
        f"  syft components: {len(syft_components)} | "
        f"above baseline: {delta} | attributed in osrb-deps.csv: {len(our_names)} | "
        f"owner-matched: {owner_matched} | POTENTIALLY UNATTRIBUTED: {len(rows)}"
    )
    for name, version, ptype in rows:
        print(f"  - {ptype:8} {name} {version}")

    if args.report:
        args.report.write_text(
            "\n".join(f"{ptype}\t{name}\t{version}" for name, version, ptype in rows)
            + "\n",
            encoding="utf-8",
        )

    # GitHub step summary, if running in Actions.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as s:
            s.write(f"### {hdr}\n\n")
            s.write(f"- above baseline: **{delta}**\n")
            s.write(f"- potentially unattributed: **{len(rows)}**\n\n")
            if rows:
                s.write("| type | name | version |\n|---|---|---|\n")
                for name, version, ptype in rows:
                    s.write(f"| {ptype} | {name} | {version} |\n")
            else:
                s.write("✅ every shipped component above the baseline is attributed.\n")

    if rows and args.fail_on_findings:
        logger.error("%d potentially-unattributed component(s) found", len(rows))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
