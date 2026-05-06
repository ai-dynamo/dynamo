# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Rust.txt generator.

Reads CycloneDX 1.5/1.6 SBOMs embedded in installed wheels under the runtime
venv. The dynamo runtime + kvbm wheels ship these via cargo-cyclonedx (run
through maturin); NIXL ships its own once we wire cargo-cyclonedx into the
NIXL block in wheel_builder.

First-party crates (`dynamo-*`, `kvbm-*`, `nixl-*`, `nvidia-*`) are KEPT in
the output — auditors and customers should see every crate that's actually
in the binary, including ours.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .common import UNKNOWN, Component, dedupe_by_name_version

logger = logging.getLogger(__name__)

ECOSYSTEM = "rust"


def _normalize_license(licenses_field: list[dict] | None) -> str:
    """Render a CycloneDX `licenses[]` array to a single SPDX expression.

    CycloneDX 1.5/1.6 each license entry is one of:
      {"license": {"id": "MIT"}}
      {"license": {"name": "Some Custom License"}}
      {"expression": "MIT OR Apache-2.0"}

    Multiple entries are joined with " AND ".
    """
    if not licenses_field:
        return UNKNOWN

    parts: list[str] = []
    for entry in licenses_field:
        if "expression" in entry:
            parts.append(entry["expression"])
        elif "license" in entry:
            inner = entry["license"]
            if "id" in inner:
                parts.append(inner["id"])
            elif "name" in inner:
                # Custom license name; surface as LicenseRef-* so policy
                # validation treats it as a deliberate non-SPDX entry.
                parts.append(f"LicenseRef-{inner['name'].replace(' ', '-')}")
            else:
                continue
        else:
            continue

    if not parts:
        return UNKNOWN
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({p})" if " " in p else p for p in parts)


def _component_from_sbom_entry(entry: dict) -> Component | None:
    """Convert one CycloneDX components[] entry to a Component."""
    name = entry.get("name")
    version = entry.get("version")
    if not name or not version:
        logger.debug("Skipping component with missing name/version: %r", entry)
        return None

    spdx = _normalize_license(entry.get("licenses"))

    purl = entry.get("purl")
    source_url: str | None = None
    if purl and purl.startswith("pkg:cargo/"):
        # Strip the ?download_url=file://... query that cargo-cyclonedx attaches
        # to first-party crates (uninformative, sometimes references workspace paths).
        if "?" in purl:
            purl_clean = purl.split("?", 1)[0]
        else:
            purl_clean = purl
        source_url = purl_clean

    return Component(
        ecosystem=ECOSYSTEM,
        name=name,
        version=str(version),
        spdx=spdx,
        source_url=source_url,
    )


def _find_wheel_sboms(venv_dir: Path) -> list[Path]:
    """Locate every CycloneDX SBOM embedded in installed wheels.

    Walks ${VIRTUAL_ENV}/lib/python*/site-packages/*.dist-info/sboms/*.cyclonedx.json.
    """
    sboms: list[Path] = []
    site_packages_glob = list(venv_dir.glob("lib/python*/site-packages"))
    for sp in site_packages_glob:
        sboms.extend(sp.glob("*.dist-info/sboms/*.cyclonedx.json"))
    return sorted(sboms)


def collect_components(venv_dir: Path) -> list[Component]:
    """Read every wheel SBOM under venv_dir and return deduped Components.

    SBOMs that are not Rust-flavored (e.g. NIXL's auditwheel.cdx.json which
    enumerates RPM libs, not Rust crates) are skipped: we only consume
    components whose purl starts with `pkg:cargo/`.
    """
    sboms = _find_wheel_sboms(venv_dir)
    if not sboms:
        logger.warning("No wheel SBOMs found under %s", venv_dir)
        return []

    components: list[Component] = []
    for sbom_path in sboms:
        try:
            doc = json.loads(sbom_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping unreadable SBOM %s: %s", sbom_path, exc)
            continue

        cargo_count = 0
        for entry in doc.get("components", []) or []:
            purl = entry.get("purl") or ""
            if not purl.startswith("pkg:cargo/"):
                continue
            comp = _component_from_sbom_entry(entry)
            if comp is None:
                continue
            components.append(comp)
            cargo_count += 1
        logger.info(
            "Read %s: %d cargo components (skipped %d non-cargo)",
            sbom_path.name,
            cargo_count,
            len(doc.get("components", []) or []) - cargo_count,
        )

    deduped = dedupe_by_name_version(components)
    logger.info(
        "Collected %d unique cargo crates after dedupe (from %d raw entries across %d SBOMs)",
        len(deduped),
        len(components),
        len(sboms),
    )
    return deduped


def generate(venv_dir: Path, output_dir: Path) -> list[Component]:
    """Read SBOMs from the venv, write NOTICES-Rust.txt + rust-deps.csv."""
    from . import common

    components = collect_components(venv_dir)
    common.write_notices(ECOSYSTEM, components, output_dir)
    common.write_deps_csv(ECOSYSTEM, components, output_dir)
    return components
