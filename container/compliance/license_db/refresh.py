#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh license-db.json by querying upstream registries.

Runs in the weekly cron workflow (.github/workflows/license-db-refresh.yml).
Resolution path per ecosystem:

  rust    -> crates.io  https://crates.io/api/v1/crates/<name>/<version>
  go      -> deps.dev   https://api.deps.dev/v3alpha/systems/go/packages/<module>/versions/<version>
  python  -> PyPI       https://pypi.org/pypi/<name>/<version>/json
  dpkg    -> DEP-5 copyright in /usr/share/doc/<pkg>/copyright (parsed from a base image)

Inputs (the (ecosystem, name, version) tuples to resolve) come from running the
ecosystem-specific SBOM generator against trunk:

  rust   -> CycloneDX SBOMs embedded in dist-info/sboms/ inside the
            ai_dynamo_runtime / kvbm / nixl wheels we build (cargo-cyclonedx
            output via maturin).
  go     -> cyclonedx-gomod app -licenses against each Go module
            (deploy/operator, deploy/snapshot, deploy/inference-gateway/epp).
  python -> pip-licenses --format=json against the runtime venv inside a
            built container.
  dpkg   -> syft scan -o cyclonedx-json on cuda-dl-base / NGC base images,
            unioned with what we install on top in each runtime image.

Output: container/compliance/license_db/license-db.json — sorted, stable.

Policy: this script does NOT enforce the licenses.toml policy. That's
container/compliance/policy/validate.py's job, and it's invoked by the
refresh workflow AFTER refresh.py emits the new JSON. If validate.py exits
non-zero, the bot's PR fails CI and a human has to update licenses.toml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent / "license-db.json"


def _registry_lookup_rust(name: str, version: str) -> str | None:
    """Query crates.io for a Rust crate's SPDX expression. TODO: implement."""
    raise NotImplementedError("Rust registry lookup not yet implemented")


def _registry_lookup_go(module: str, version: str) -> str | None:
    """Query api.deps.dev for a Go module's SPDX. TODO: implement."""
    raise NotImplementedError("Go registry lookup not yet implemented")


def _registry_lookup_python(name: str, version: str) -> str | None:
    """Query PyPI for a Python package's license. TODO: implement.

    Reuse the PEP-440 base-version retry + GitHub-license fallback pattern
    from PR #8654's license_lookup.py.
    """
    raise NotImplementedError("PyPI registry lookup not yet implemented")


def _registry_lookup_dpkg(name: str, version: str) -> str | None:
    """Resolve a dpkg license. TODO: implement (parse DEP-5 from a base image)."""
    raise NotImplementedError("dpkg lookup not yet implemented")


def _enumerate_trunk_dependencies() -> list[tuple[str, str, str]]:
    """Walk trunk source for the (ecosystem, name, version) set to resolve.

    TODO: implement. The right path is to invoke each ecosystem's SBOM
    generator against trunk (cargo cyclonedx, cyclonedx-gomod, pip-licenses,
    syft) and union the components. See module docstring.
    """
    raise NotImplementedError("Trunk enumeration not yet implemented")


def write_db(entries: list[dict], path: Path) -> None:
    """Write license-db.json with stable, deterministic ordering."""
    sorted_entries = sorted(
        entries,
        key=lambda e: (e["ecosystem"], e["name"].lower(), e["version"]),
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "container.compliance.license_db.refresh",
        "entries": sorted_entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh license-db.json from upstream registries",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_DB_PATH,
        help="Path to license-db.json (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve entries and print summary; don't write the JSON",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    deps = _enumerate_trunk_dependencies()
    logger.info("Resolving %d (ecosystem, name, version) tuples", len(deps))

    entries: list[dict] = []
    unresolved: list[tuple[str, str, str]] = []

    for ecosystem, name, version in deps:
        lookup_fn = {
            "rust": _registry_lookup_rust,
            "go": _registry_lookup_go,
            "python": _registry_lookup_python,
            "dpkg": _registry_lookup_dpkg,
        }.get(ecosystem)
        if lookup_fn is None:
            logger.warning("Unknown ecosystem %r for %s@%s", ecosystem, name, version)
            unresolved.append((ecosystem, name, version))
            continue

        try:
            spdx = lookup_fn(name, version)
        except NotImplementedError:
            raise
        except Exception as exc:  # pragma: no cover — network failures, etc.
            logger.warning("Lookup failed for %s/%s@%s: %s", ecosystem, name, version, exc)
            unresolved.append((ecosystem, name, version))
            continue

        if spdx is None:
            unresolved.append((ecosystem, name, version))
            continue

        entries.append(
            {
                "ecosystem": ecosystem,
                "name": name,
                "version": version,
                "spdx": spdx,
                "source_url": None,  # populated by individual lookup_fn implementations
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    logger.info(
        "Resolved %d entries; %d unresolved", len(entries), len(unresolved)
    )

    if args.dry_run:
        return 0

    write_db(entries, args.db)
    logger.info("Wrote %s (%d entries)", args.db, len(entries))

    if unresolved:
        logger.warning("Unresolved entries (will be UNKNOWN at policy-check time):")
        for eco, name, ver in unresolved:
            logger.warning("  %s %s@%s", eco, name, ver)
        return 1  # signal to refresh-PR CI that human review is needed

    return 0


if __name__ == "__main__":
    sys.exit(main())
