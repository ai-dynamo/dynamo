#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a CycloneDX inventory for selected Rust binary packages.

Unlike the Python runtime images, standalone Rust binaries do not carry a
wheel-embedded SBOM. This helper uses Cargo's resolved dependency graph and
keeps only normal (runtime-linked) dependencies reachable from the requested
packages. Build and development dependencies are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def _is_normal_dependency(dep: dict) -> bool:
    """Return whether a resolved dependency has a normal dependency edge."""
    kinds = dep.get("dep_kinds") or [{"kind": None}]
    return any(kind.get("kind") in (None, "normal") for kind in kinds)


def runtime_package_ids(metadata: dict, root_names: set[str]) -> set[str]:
    """Return package IDs reachable from ``root_names`` through normal edges."""
    packages = {package["id"]: package for package in metadata.get("packages", [])}
    nodes = {
        node["id"]: node for node in (metadata.get("resolve") or {}).get("nodes", [])
    }
    workspace_members = set(metadata.get("workspace_members", []))
    roots = {
        package_id
        for package_id in workspace_members
        if packages.get(package_id, {}).get("name") in root_names
    }

    found_names = {packages[package_id]["name"] for package_id in roots}
    missing = sorted(root_names - found_names)
    if missing:
        raise ValueError(f"Cargo metadata is missing requested package(s): {missing}")

    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        package_id = pending.pop()
        if package_id in reachable:
            continue
        if package_id not in packages:
            raise ValueError(f"resolved package is missing metadata: {package_id}")
        reachable.add(package_id)
        for dep in nodes.get(package_id, {}).get("deps", []):
            dep_id = dep.get("pkg")
            if dep_id and _is_normal_dependency(dep) and dep_id not in reachable:
                pending.append(dep_id)

    return reachable


def _component(package: dict, root_names: set[str]) -> dict:
    name = package["name"]
    version = package["version"]
    purl = f"pkg:cargo/{quote(name, safe='')}@{quote(version, safe='')}"
    component = {
        "type": "application" if name in root_names else "library",
        "bom-ref": purl,
        "name": name,
        "version": version,
        "purl": purl,
        "licenses": [{"expression": _license_expression(package.get("license"))}],
    }
    source_url = package.get("repository") or package.get("homepage")
    if source_url:
        component["externalReferences"] = [{"type": "vcs", "url": source_url}]
    return component


def _license_expression(value: str | None) -> str:
    """Normalize Cargo's legacy slash-separated dual-license notation."""
    if not value:
        return "UNKNOWN"
    # Older crates commonly declare `MIT/Apache-2.0`. Cargo accepts that
    # legacy spelling, but SPDX and CycloneDX require an explicit operator.
    parts = [part.strip() for part in re.split(r"\s*/\s*", value) if part.strip()]
    return " OR ".join(parts)


def cyclonedx_document(
    metadata: dict,
    root_names: set[str],
    included_packages: set[tuple[str, str]] | None = None,
) -> dict:
    """Build a deterministic CycloneDX document for selected binary roots."""
    packages = {package["id"]: package for package in metadata.get("packages", [])}
    reachable = runtime_package_ids(metadata, root_names)
    selected = []
    for package_id in reachable:
        package = packages[package_id]
        key = (package["name"], package["version"])
        if included_packages is None or key in included_packages:
            selected.append(_component(package, root_names))
    components = sorted(
        selected,
        key=lambda component: (
            component["name"].lower(),
            component["version"],
            component["purl"],
        ),
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "components": components,
    }


def _parse_cargo_tree(output: str) -> set[tuple[str, str]]:
    """Parse package name/version keys from ``cargo tree --format {p}``."""
    packages: set[tuple[str, str]] = set()
    for line in output.splitlines():
        match = re.match(r"^(\S+) v(\S+)", line)
        if match:
            packages.add((match.group(1), match.group(2)))
    return packages


def cargo_tree_packages(package: str, target: str | None) -> set[tuple[str, str]]:
    """Return packages Cargo links through normal edges for one binary root."""
    command = [
        "cargo",
        "tree",
        "--locked",
        "--edges",
        "normal",
        "--prefix",
        "none",
        "--format",
        "{p}",
        "--package",
        package,
    ]
    if target:
        command.extend(["--target", target])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cargo tree failed for {package}:\n{result.stderr.rstrip()}"
        )
    packages = _parse_cargo_tree(result.stdout)
    if not packages:
        raise RuntimeError(f"cargo tree returned no packages for {package}")
    return packages


def cargo_metadata(manifest_path: Path, target: str | None) -> dict:
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--manifest-path",
        str(manifest_path),
    ]
    if target:
        command.extend(["--filter-platform", target])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cargo metadata failed for {manifest_path}:\n{result.stderr.rstrip()}"
        )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CycloneDX SBOM for selected Rust binary packages"
    )
    parser.add_argument("--manifest-path", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--package", action="append", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = cargo_metadata(args.manifest_path, args.target or None)
    root_names = set(args.package)
    included_packages: set[tuple[str, str]] = set()
    for package in sorted(root_names):
        included_packages.update(cargo_tree_packages(package, args.target or None))
    document = cyclonedx_document(metadata, root_names, included_packages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
