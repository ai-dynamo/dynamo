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

Matching requires ecosystem, name and version. Python names use PEP 503
normalization; Go versions accept the conventional leading v. Generic binary
identities require native rows, not an unrelated Rust/Python name. With --policy,
the resolved inventory licenses must also pass the build's license policy.
Scanner metadata is retained in the SBOM; it does not silently replace the
inventory's reviewed license corrections.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from posixpath import normpath

from .identity import component_identity, identity
from .policy.validate import load_policy, validate_row

logger = logging.getLogger("compliance.audit_image_sbom")


def _normalize(name: str) -> str:
    """Lowercase + unify separators — a coarse key that absorbs common
    dist-vs-import / packaging name skew (e.g. sgl_kernel vs sglang-kernel)
    without over-merging."""
    return (name or "").strip().lower().replace("_", "-").replace(" ", "-")


def _load_cdx_components(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("components", []) or []


def _version(ecosystem: str, version: str) -> str:
    value = str(version or "UNKNOWN").strip()
    return re.sub(r"^v(?=\d)", "", value) if ecosystem == "go" else value


def _key(ecosystem: str, name: str, version: str) -> tuple[str, str, str]:
    return (*identity(ecosystem, name), _version(ecosystem, version))


def _component_key(component: dict) -> tuple[str, str, str]:
    ecosystem, name = component_identity(component)
    return _key(ecosystem, name, component.get("version", ""))


def _csv_inventory(csv_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Read the full current inventory; reject diff/legacy or ambiguous input."""
    records = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"ecosystem", "name", "version"}.issubset(reader.fieldnames or []):
            raise ValueError(f"not a full attribution inventory: {csv_path}")
        for row in reader:
            if not all(row.get(field) for field in ("ecosystem", "name", "version")):
                raise ValueError(f"incomplete component identity in {csv_path}")
            key = _key(row["ecosystem"], row["name"], row["version"])
            if key in records and records[key].get("spdx") != row.get("spdx"):
                raise ValueError(f"conflicting license records for {key} in {csv_path}")
            records[key] = row
    return records


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


def _rootfs_path(path: str, rootfs_prefix: str = "") -> str:
    """Use one root-relative representation for Syft and package-owned paths."""
    normalized = normpath("/" + path.lstrip("/"))
    prefix = rootfs_prefix.rstrip("/")
    if prefix and (normalized == prefix or normalized.startswith(prefix + "/")):
        normalized = normalized[len(prefix) :] or "/"
    return normalized


def _locations(component: dict, rootfs_prefix: str = "") -> set[str]:
    return {
        _rootfs_path(p["value"], rootfs_prefix)
        for p in component.get("properties", [])
        if p["name"].startswith("syft:location:") and p["name"].endswith(":path")
    }


def _ownership_coverage(
    native: dict, our_keys: set[tuple[str, str, str]], rootfs_prefix: str = ""
) -> dict[str, set[str]]:
    """Known owning-package records; never infer coverage from directory alone."""
    owners: dict[str, set[str]] = {}
    for artifact in native.get("artifacts", []):
        name = artifact["name"]
        key = _key("dpkg", name, artifact.get("version", ""))
        if key not in our_keys or artifact["type"] != "deb":
            continue
        for item in artifact.get("metadata", {}).get("files", []):
            owners.setdefault(_rootfs_path(item["path"], rootfs_prefix), set()).add(
                name
            )
    return owners


def _covered_by_owner(
    component: dict, owners: dict[str, set[str]], rootfs_prefix: str = ""
) -> bool:
    locations = _locations(component, rootfs_prefix)
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
    p.add_argument("--policy", type=Path, help="Validate resolved inventory licenses")
    p.add_argument(
        "--rootfs-prefix",
        default="",
        help="Physical root of an external directory scan",
    )
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
    baseline_keys: set[tuple[str, str, str]] = set()
    if args.base_sbom and args.base_sbom.is_file():
        baseline_keys = {
            _component_key(c)
            for c in _load_cdx_components(args.base_sbom)
            if c.get("name")
        }
    elif args.base_sbom:
        p.error(f"requested baseline SBOM not found: {args.base_sbom}")

    inventory = _csv_inventory(args.our_csv)
    our_keys = set(inventory)
    violations = []
    if args.policy:
        policy = load_policy(args.policy)
        for row in inventory.values():
            violation = validate_row(
                policy,
                row["ecosystem"],
                row["name"],
                row["version"],
                row.get("spdx", ""),
            )
            if violation:
                violations.append(violation)
    owners = (
        _ownership_coverage(
            json.loads(args.syft_json.read_text()), our_keys, args.rootfs_prefix
        )
        if args.syft_json
        else {}
    )
    owner_matched = 0

    # Only an exact ecosystem/name/version match can establish attribution.
    # A same-name package from a different ecosystem or release is not evidence.
    unattributed: list[dict] = []
    delta = 0
    for c in syft_components:
        # Syft's CycloneDX includes source files used as package evidence and
        # the operating-system descriptor. These are not package identities.
        if c.get("type") in {"file", "operating-system"}:
            continue
        name, version = c.get("name"), c.get("version")
        if not name:
            continue
        key = _component_key(c)
        if key in baseline_keys:
            continue
        delta += 1
        if key not in our_keys and not _is_first_party(name):
            if _covered_by_owner(c, owners, args.rootfs_prefix):
                owner_matched += 1
                continue
            unattributed.append(c)

    # Keep distinct versions and ecosystems visible in the report.
    seen: set[tuple[str, str, str]] = set()
    rows: list[tuple[str, str, str]] = []
    for c in sorted(unattributed, key=lambda c: _normalize(c.get("name", ""))):
        key = _component_key(c)
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
        f"above baseline: {delta} | attributed in osrb-deps.csv: {len(our_keys)} | "
        f"owner-matched: {owner_matched} | POTENTIALLY UNATTRIBUTED: {len(rows)} | "
        f"POLICY VIOLATIONS: {len(violations)}"
    )
    for name, version, ptype in rows:
        print(f"  - {ptype:8} {name} {version}")
    for violation in violations:
        print(str(violation))

    if args.report:
        args.report.write_text(
            "\n".join(
                [f"{ptype}\t{name}\t{version}" for name, version, ptype in rows]
                + [f"policy\t{violation}" for violation in violations]
            )
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
            s.write(f"- policy violations: **{len(violations)}**\n\n")
            if rows:
                s.write("| type | name | version |\n|---|---|---|\n")
                for name, version, ptype in rows:
                    s.write(f"| {ptype} | {name} | {version} |\n")
            else:
                s.write("✅ every shipped component above the baseline is attributed.\n")

    if (rows or violations) and args.fail_on_findings:
        logger.error(
            "%d unattributed components and %d policy violations",
            len(rows),
            len(violations),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
