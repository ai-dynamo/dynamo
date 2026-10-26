#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a CycloneDX SBOM for one Cargo binary's dependency closure.

Wheels built through maturin carry their own CycloneDX SBOM, which is what
``generators/rust.py`` normally reads. Binaries built straight from the Cargo
workspace ship in no wheel, so nothing describes them -- the case this script
exists for is ``dynamo-ext-proc``, whose ``/epp`` binary is built in
``deploy/inference-gateway/ext-proc/Dockerfile`` and copied into the frontend
image. Without an SBOM emitted here, its crates are attributed nowhere (crates
unique to it) or at the wheel's version rather than the version actually linked
into the binary (crates it shares with the wheel, which resolves against a
separate lockfile because ``lib/bindings/python`` is deliberately outside the
root workspace).

The output is deliberately the same shape ``cargo-cyclonedx`` produces, since
``generators/rust.py`` and ``collect_sources.py`` both key off `pkg:cargo/`
purls and a CycloneDX ``licenses[]`` array.

Scope: the NORMAL dependency closure of the named package, exactly as
``cargo tree -p <package> -e normal`` reports it. Dev-dependencies and
build-dependencies are excluded -- neither is linked into a release binary.

The closure comes from ``cargo tree`` rather than from walking ``cargo
metadata``'s resolve graph because the two disagree. ``cargo build -p X`` under
resolver 2/3 resolves features for X alone, while ``cargo metadata`` reports the
workspace-unified resolve; walking the latter over-reports this binary's closure
by roughly 70 crates. Over-reporting is the safe direction for attribution, but
it would also feed crates the binary never links into the license policy gate,
inviting exceptions for code we do not ship. ``cargo metadata`` is still used,
for the authoritative per-crate license and repository fields.

Usage:
    python3 cargo_sbom.py --package dynamo-ext-proc \\
        --target x86_64-unknown-linux-gnu --output /sbom-rust-epp.cdx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("compliance.cargo_sbom")

# cargo's `license` field is an SPDX expression, but crates predating SPDX
# support in cargo use the legacy slash form ("MIT/Apache-2.0").
_LEGACY_LICENSE_SEP = "/"


# `cargo tree --format "{p}"` prints "<name> v<version>" and may append a source
# or kind marker, e.g. "async-trait v0.1.89 (proc-macro)", plus " (*)" when a
# subtree is elided as already-shown. Name and version never contain spaces.
_TREE_LINE = re.compile(r"^(?P<name>\S+) v(?P<version>\S+)")


def _run(cmd: list[str]) -> str:
    logger.debug("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{cmd[1]} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def load_metadata(
    manifest_path: Path,
    cargo_cmd: str = "cargo",
) -> dict:
    """Run `cargo metadata` and return the parsed JSON.

    No `--filter-platform`: this call is only the source of per-crate license and
    repository fields, so the full package list is wanted and platform filtering
    would just risk a crate in the tree having no metadata entry to join to.

    `--locked` keeps this honest: it fails rather than silently re-resolving
    Cargo.lock, so the SBOM always describes the versions the build linked.
    """
    return json.loads(
        _run(
            [
                cargo_cmd,
                "metadata",
                "--format-version",
                "1",
                "--locked",
                "--manifest-path",
                str(manifest_path),
            ]
        )
    )


def load_tree(
    package: str,
    manifest_path: Path,
    target: str | None,
    cargo_cmd: str = "cargo",
) -> str:
    """Run `cargo tree` for `package`'s normal-dependency closure."""
    cmd = [
        cargo_cmd,
        "tree",
        "--locked",
        "--manifest-path",
        str(manifest_path),
        "--package",
        package,
        "--edges",
        "normal",
        "--prefix",
        "none",
        "--format",
        "{p}",
    ]
    if target:
        cmd += ["--target", target]
    return _run(cmd)


def parse_tree(tree_output: str) -> set[tuple[str, str]]:
    """Parse `cargo tree` output into the set of (name, version) it lists.

    Includes the root package itself: first-party crates are kept in NOTICES on
    purpose (see generators/rust.py), so `dynamo-ext-proc` must be visible.
    """
    crates: set[tuple[str, str]] = set()
    for line in tree_output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TREE_LINE.match(line)
        if match is None:
            logger.debug("skipping unparseable cargo tree line: %r", line)
            continue
        crates.add((match["name"], match["version"]))
    if not crates:
        raise RuntimeError("cargo tree produced no parseable packages")
    return crates


def packages_by_key(metadata: dict) -> dict[tuple[str, str], dict]:
    """Index `cargo metadata` packages by (name, version)."""
    return {
        (p["name"], str(p["version"])): p
        for p in metadata.get("packages", [])
        if p.get("name") and p.get("version")
    }


def _licenses_field(pkg: dict) -> list[dict]:
    """Render a cargo `license` string as a CycloneDX licenses[] array.

    Emitting `expression` (rather than `license.id`) matches cargo-cyclonedx and
    round-trips compound expressions like "MIT OR Apache-2.0" unchanged through
    generators/rust.py::_normalize_license. A crate carrying only `license_file`
    yields an empty array, which that generator maps to UNKNOWN so
    license_overrides.yaml can supply the answer.
    """
    raw = (pkg.get("license") or "").strip()
    if not raw:
        return []
    if _LEGACY_LICENSE_SEP in raw and " OR " not in raw and " AND " not in raw:
        raw = " OR ".join(part.strip() for part in raw.split(_LEGACY_LICENSE_SEP))
    return [{"expression": raw}]


def build_sbom(
    crates: set[tuple[str, str]],
    pkg_index: dict[tuple[str, str], dict],
    root_package: str,
) -> dict:
    """Assemble the CycloneDX 1.5 document for `crates`.

    A crate present in the tree but absent from `pkg_index` still gets a
    component (name, version and purl are already known); it just carries no
    license, which routes it through license_overrides.yaml rather than
    dropping it from attribution entirely.
    """
    components = []
    for name, version in sorted(crates):
        entry: dict = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:cargo/{name}@{version}",
        }
        pkg = pkg_index.get((name, version))
        if pkg is None:
            logger.warning(
                "%s-%s is in the dependency tree but not in cargo metadata; "
                "emitting it without license data",
                name,
                version,
            )
        else:
            licenses = _licenses_field(pkg)
            if licenses:
                entry["licenses"] = licenses
            repo = pkg.get("repository")
            if repo:
                entry["externalReferences"] = [{"type": "vcs", "url": repo}]
        components.append(entry)

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "tools": [{"vendor": "NVIDIA", "name": "compliance.cargo_sbom"}],
            "component": {
                "type": "application",
                "name": root_package,
                "purl": f"pkg:cargo/{root_package}",
            },
        },
        "components": components,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compliance.cargo_sbom",
        description="Emit a CycloneDX SBOM for a Cargo binary's normal-dependency closure",
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Root package name whose closure is described (e.g. dynamo-ext-proc)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("Cargo.toml"),
        help="Workspace Cargo.toml to resolve against (default: ./Cargo.toml)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Rust target triple to filter the resolve graph to (e.g. "
            "x86_64-unknown-linux-gnu). Omit to include every platform's deps."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the CycloneDX JSON document to",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help=(
            "Read a pre-captured `cargo metadata` document from this file "
            "instead of invoking cargo. Used by the unit tests."
        ),
    )
    parser.add_argument(
        "--tree-output",
        type=Path,
        default=None,
        help=(
            "Read pre-captured `cargo tree` output from this file instead of "
            "invoking cargo. Used by the unit tests."
        ),
    )
    parser.add_argument("--cargo-cmd", default="cargo", help="Path to the cargo binary")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    else:
        metadata = load_metadata(args.manifest_path, args.cargo_cmd)

    if args.tree_output is not None:
        tree_output = args.tree_output.read_text(encoding="utf-8")
    else:
        tree_output = load_tree(
            args.package, args.manifest_path, args.target, args.cargo_cmd
        )

    crates = parse_tree(tree_output)
    if not any(name == args.package for name, _ in crates):
        raise RuntimeError(
            f"cargo tree output does not contain the root package {args.package!r}"
        )
    sbom = build_sbom(crates, packages_by_key(metadata), args.package)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    unlicensed = sum(1 for c in sbom["components"] if "licenses" not in c)
    logger.info(
        "Wrote %s: %d crates in %s's normal-dependency closure (%d without a "
        "declared license, left to license_overrides.yaml)",
        args.output,
        len(sbom["components"]),
        args.package,
        unlicensed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
