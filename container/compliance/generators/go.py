# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Go.txt generator.

Reads CycloneDX SBOMs produced by `cyclonedx-gomod app -licenses -json` in
each Go builder stage (operator/snapshot-agent/EPP). Same shape as rust.py:
walk components, filter by purl prefix, normalize licenses, dedupe.

Expected upstream Dockerfile pattern (in each Go builder):

    RUN go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@v1.7.0
    RUN cyclonedx-gomod app -licenses -json -main ./cmd/<binary> -output /sbom-go.cdx.json .

The licenses stage then COPYs /sbom-go.cdx.json out and runs:

    python3 -m compliance.generators --ecosystem go \\
        --go-sbom /sbom-go.cdx.json --output-dir /legal

First-party modules (anything under github.com/ai-dynamo/, github.com/nvidia/,
github.com/NVIDIA/) are KEPT in the output — same principle as Rust.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .common import UNKNOWN, Component, dedupe_by_name_version

logger = logging.getLogger(__name__)

ECOSYSTEM = "go"


def _normalize_license(licenses_field: list[dict] | None) -> str:
    """Render a CycloneDX `licenses[]` array to a single SPDX expression.

    Same shape as cargo-cyclonedx output (see rust.py); cyclonedx-gomod uses
    the same CycloneDX schema.
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
    name = entry.get("name")
    version = entry.get("version")
    if not name or not version:
        return None

    # cyclonedx-gomod v1.7.0 emits detected licenses under evidence.licenses
    # (CycloneDX evidence-based provenance) rather than the top-level
    # licenses field. cargo-cyclonedx, by contrast, populates the top-level
    # field directly. Read both so this generator handles either source
    # without caring which tool emitted the SBOM.
    licenses_field = entry.get("licenses")
    if not licenses_field:
        evidence = entry.get("evidence") or {}
        licenses_field = evidence.get("licenses")
    spdx = _normalize_license(licenses_field)

    purl = entry.get("purl") or ""
    source_url: str | None = None
    if purl.startswith("pkg:golang/"):
        # Strip the canonical purl down for use as a source URL pointer; the
        # full purl is the actual fetch URL via go module proxy.
        source_url = purl.split("?", 1)[0]

    return Component(
        ecosystem=ECOSYSTEM,
        name=name,
        version=str(version),
        spdx=spdx,
        source_url=source_url,
    )


def collect_components(sbom_paths: list[Path]) -> list[Component]:
    """Read one or more Go SBOMs, return deduped Components.

    cyclonedx-gomod produces one SBOM per Go binary. Multi-SBOM inputs arise
    when an image combines binaries from upstream stages — e.g. frontend
    pulls EPP's SBOM in alongside any Go SBOM produced in the frontend
    build itself. Components present in multiple inputs dedupe by
    (name, version) and merge license info via dedupe_by_name_version.
    """
    components: list[Component] = []
    for sbom_path in sbom_paths:
        if not sbom_path.is_file():
            logger.warning("Go SBOM not found: %s", sbom_path)
            continue

        try:
            doc = json.loads(sbom_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Could not parse Go SBOM %s: %s", sbom_path, exc)
            continue

        golang_count = 0
        for entry in doc.get("components", []) or []:
            purl = entry.get("purl") or ""
            if not purl.startswith("pkg:golang/"):
                continue
            comp = _component_from_sbom_entry(entry)
            if comp is None:
                continue
            components.append(comp)
            golang_count += 1
        logger.info(
            "Read %s: %d golang components (skipped %d non-golang)",
            sbom_path.name,
            golang_count,
            len(doc.get("components", []) or []) - golang_count,
        )

    deduped = dedupe_by_name_version(components)
    logger.info(
        "Collected %d unique Go modules after dedupe (from %d raw entries across %d SBOMs)",
        len(deduped),
        len(components),
        len(sbom_paths),
    )
    return deduped


def generate(
    sbom_paths: list[Path],
    output_dir: Path,
    subtract: set[tuple[str, str]] | None = None,
) -> list[Component]:
    """Read one or more Go SBOMs, write NOTICES-Go.txt + go-deps.csv.

    When `subtract` is provided, components matching (name, version) are
    filtered before writing — used to drop baseline-owned components.
    """
    from . import common

    components = collect_components(sbom_paths)
    if subtract:
        components = common.subtract_baseline(components, subtract)
    common.write_notices(ECOSYSTEM, components, output_dir)
    common.write_deps_csv(ECOSYSTEM, components, output_dir)
    return components
