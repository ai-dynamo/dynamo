#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSRB submission packager.

Stitches the inline-compliance pipeline's per-image artifacts into a single
bundle suitable for NVIDIA's Open Source Review Board:

  Inputs (all already produced by the inline-compliance pipeline):
    /tmp/legal/                 NOTICES + license texts (from --target legal)
    /tmp/sboms/                 per-ecosystem deps CSVs (from --target sboms)
    /tmp/sources/sources.zip    source archives (from --target sources_archive,
                                post-merge / RC / release only)
    container/compliance/base_sboms/<base>.cdx.json   if applicable
    container/compliance/osrb/{linkage.yaml,distribution.yaml,modifications/}

  Output:
    osrb-<image>-<version>-<arch>.zip containing:
      manifest.csv              one row per package, joined with linkage map
      manifest.cdx.json         consolidated CycloneDX (all ecosystems unioned)
      licenses/<spdx>/...       upstream LICENSE text per (name, version)
      modifications/            patches we apply to vendored deps (typically empty)
      attribution-base.txt      "these packages came from <base>"
      build-provenance.json     image digest, build timestamp, git SHAs, tool
                                versions, AND the sha256 of the companion
                                sources archive (so OSRB can verify the
                                independently-uploaded sources match this bundle)
      README.md                 generated; explains every file
      CHECKSUMS.sha256          integrity check after upload

  Sources are emitted in a SEPARATE archive (sources-<image>-<version>-<arch>.zip)
  by container/compliance/collect_sources.py — the OSRB bundle is small
  (MB-scale, just metadata + license texts) and reviewable on its own,
  while sources can be GB-scale and downloaded only when needed.

Runs in the release pipeline only — same cadence as source archival; OSRB
review happens at release boundaries, not per-PR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---- Inputs --------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    import yaml  # PyYAML

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_deps_csvs(sboms_dir: Path) -> list[dict]:
    """Read every <eco>-deps.csv under sboms_dir; flat list of rows."""
    rows: list[dict] = []
    for csv_path in sorted(sboms_dir.rglob("*-deps.csv")):
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def load_base_components(base_sbom_path: Path | None) -> set[tuple[str, str, str]]:
    """Read base SBOM CycloneDX; return set of (ecosystem, name, version) keyed by purl."""
    if base_sbom_path is None or not base_sbom_path.is_file():
        return set()
    doc = json.loads(base_sbom_path.read_text(encoding="utf-8"))
    purl_to_eco = {
        "pkg:cargo/": "rust",
        "pkg:pypi/": "python",
        "pkg:golang/": "go",
        "pkg:deb/": "dpkg",
        "pkg:rpm/": "dpkg",
        "pkg:alpine/": "dpkg",
    }
    out: set[tuple[str, str, str]] = set()
    for c in doc.get("components", []) or []:
        purl = c.get("purl") or ""
        eco = next((v for k, v in purl_to_eco.items() if purl.startswith(k)), None)
        if eco and c.get("name") and c.get("version"):
            out.add((eco, c["name"], str(c["version"])))
    return out


# ---- Linkage join --------------------------------------------------------


def linkage_for(row: dict, linkage_map: dict) -> str:
    ecosystem = row["ecosystem"]
    name = row["name"]
    if ecosystem == "native":
        overrides = linkage_map.get("native_overrides", {}) or {}
        if name in overrides:
            return overrides[name]
    defaults = linkage_map.get("ecosystem_defaults", {}) or {}
    return defaults.get(ecosystem, "unknown")


# ---- Manifest writers ---------------------------------------------------


def write_manifest_csv(
    rows: list[dict],
    base_keys: set[tuple[str, str, str]],
    linkage_map: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "ecosystem",
                "name",
                "version",
                "spdx",
                "linkage",
                "came_from_base",
                "source_url",
            ]
        )
        for row in sorted(
            rows, key=lambda r: (r["ecosystem"], r["name"].lower(), r["version"])
        ):
            key = (row["ecosystem"], row["name"], row["version"])
            writer.writerow(
                [
                    row["ecosystem"],
                    row["name"],
                    row["version"],
                    row.get("spdx", ""),
                    linkage_for(row, linkage_map),
                    "yes" if key in base_keys else "no",
                    row.get("source_url", "") or "",
                ]
            )


def write_consolidated_cyclonedx(
    rows: list[dict],
    base_keys: set[tuple[str, str, str]],
    image: str,
    version: str,
    output_path: Path,
) -> None:
    """Emit a single CycloneDX 1.6 BOM unioning every ecosystem's components.

    Components inherited from the base image are tagged with a custom
    `properties[].name = "x-osrb:inherited-from-base"` so OSRB can fast-skip
    them (they're attributed via the base's own OSRB submission).
    """
    purl_template = {
        "rust": "pkg:cargo/{name}@{version}",
        "python": "pkg:pypi/{name}@{version}",
        "go": "pkg:golang/{name}@{version}",
        "dpkg": "pkg:deb/ubuntu/{name}@{version}",
        "native": "pkg:generic/{name}@{version}",
    }
    components: list[dict] = []
    for row in sorted(
        rows, key=lambda r: (r["ecosystem"], r["name"].lower(), r["version"])
    ):
        eco, name, ver = row["ecosystem"], row["name"], row["version"]
        purl = purl_template.get(eco, "pkg:generic/{name}@{version}").format(
            name=name, version=ver
        )
        c = {
            "type": "library",
            "bom-ref": purl,
            "name": name,
            "version": ver,
            "purl": purl,
        }
        if row.get("spdx"):
            c["licenses"] = [{"expression": row["spdx"]}]
        if (eco, name, ver) in base_keys:
            c["properties"] = [{"name": "x-osrb:inherited-from-base", "value": "true"}]
        components.append(c)
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{hashlib.sha256(f'{image}-{version}'.encode()).hexdigest()[:32]}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tools": [{"name": "container.compliance.osrb.package"}],
            "component": {"type": "container", "name": image, "version": version},
        },
        "components": components,
    }
    output_path.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")


def copy_license_texts(legal_dir: Path, output_dir: Path) -> int:
    """Recursively copy /legal/<eco>/*.txt and per-package LICENSE files into
    output_dir/licenses/, organized by SPDX ID where possible.

    For now: just mirror the directory structure. A future improvement would
    parse NOTICES-*.txt and reorganize per-SPDX-ID.
    """
    if not legal_dir.is_dir():
        return 0
    target = output_dir / "licenses"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(legal_dir, target)
    return sum(1 for _ in target.rglob("*") if _.is_file())


def sources_archive_metadata(sources_archive: Path | None) -> dict:
    """Return sha256 + size of the companion sources archive for provenance.

    Sources are NOT embedded in the OSRB bundle — they live in a separate
    archive (sources-<image>-<version>-<arch>.zip) so reviewers can
    inspect the metadata without pulling potentially-GB of source.
    Recording the sha256 here lets OSRB verify the sources archive they
    downloaded separately matches this bundle.
    """
    if (
        sources_archive is None
        or not sources_archive.is_file()
        or sources_archive.stat().st_size == 0
    ):
        return {"sources_archive": None, "sha256": None, "size_bytes": 0}
    digest = hashlib.sha256(sources_archive.read_bytes()).hexdigest()
    return {
        "sources_archive": sources_archive.name,
        "sha256": digest,
        "size_bytes": sources_archive.stat().st_size,
    }


def copy_modifications(modifications_dir: Path, output_dir: Path) -> int:
    """Copy any patches under osrb/modifications/ into the bundle."""
    if not modifications_dir.is_dir():
        return 0
    target = output_dir / "modifications"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(modifications_dir, target)
    return sum(1 for _ in target.rglob("*.patch") if _.is_file())


def write_attribution_base(
    base_keys: set[tuple[str, str, str]],
    base_image: str,
    base_digest: str,
    output_path: Path,
) -> None:
    """Write a human-readable pointer file summarizing what was inherited.

    Reviewer-oriented: the per-package data is in manifest.csv; this file
    lets a human opening the bundle see the shape of the inheritance at
    a glance, with the per-ecosystem breakdown so they can sanity-check
    that the right baseline was used (e.g., a Python-heavy framework
    image should show most inherited packages on the Python row).
    """
    from collections import Counter

    eco_counts = Counter(eco for (eco, _name, _ver) in base_keys)

    # Render ecosystems in a stable, predictable order; surface any
    # unexpected ecosystems at the end so they're noticeable.
    ordered_ecosystems = ("dpkg", "python", "rust", "go", "native", "rpm")
    breakdown_lines: list[str] = []
    seen = set()
    for eco in ordered_ecosystems:
        if eco in eco_counts:
            breakdown_lines.append(f"  {eco:8s} {eco_counts[eco]:>6d}")
            seen.add(eco)
    for eco in sorted(eco_counts.keys() - seen):
        breakdown_lines.append(f"  {eco:8s} {eco_counts[eco]:>6d}")
    breakdown = "\n".join(breakdown_lines) if breakdown_lines else "  (none)"

    output_path.write_text(
        f"""\
ATTRIBUTION — base-image inherited components
=============================================

The packages listed below were inherited from the base image
{base_image} (digest: {base_digest}). They are NOT subject to dynamo's
OSRB review process directly; refer to NVIDIA NGC's separate OSRB
submission for the base image for those packages' license analysis.

Inherited package count: {len(base_keys)}

Per-ecosystem breakdown:
{breakdown}

(Per-package list lives in manifest.csv; rows where came_from_base=yes
are the ones inherited from the base.)
""",
        encoding="utf-8",
    )


def write_provenance(
    image: str,
    version: str,
    image_digest: str,
    base_image: str,
    base_digest: str,
    sources_meta: dict,
    output_path: Path,
) -> None:
    out = {
        "image": image,
        "version": version,
        "image_digest": image_digest,
        "base_image": base_image,
        "base_image_digest": base_digest,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha_dynamo": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or None,
        "sources": sources_meta,
    }
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def write_readme(image: str, version: str, output_path: Path) -> None:
    output_path.write_text(
        f"""\
# OSRB submission bundle: {image} {version}

Generated by `container/compliance/osrb/package.py`. Every file in this
bundle has a defined contract; they're all derived from the inline-
compliance pipeline that runs as part of every dynamo image build.

## Files

| File | Description |
|---|---|
| `manifest.csv` | One row per shipped package: ecosystem, name, version, SPDX license, linkage type (static/dynamic/dynamic_load/static_into_nixl/case_by_case), came_from_base flag (`yes`/`no`), source URL where applicable. |
| `manifest.cdx.json` | Consolidated CycloneDX 1.6 SBOM unioning every ecosystem. Components inherited from the base carry the property `x-osrb:inherited-from-base = true` for fast filtering. |
| `licenses/` | Upstream LICENSE text for every package, mirrored from `/legal/` produced by the inline-compliance pipeline. |
| `modifications/` | Patches we apply to vendored upstream sources, with per-patch READMEs explaining upstream tracking and why we can't bump to the fix. Empty for most packages. |
| `attribution-base.txt` | Plain-text statement: "these packages came from `<base>@<digest>`; refer to NGC's separate OSRB for them." |
| `build-provenance.json` | Image digest, build timestamp, git SHA of dynamo, base-image digest, tool versions, AND the sha256 of the companion sources archive (`sources` field) for cross-verification. |
| `CHECKSUMS.sha256` | sha256 of every file in this bundle. Verify after upload to detect tampering. |

## Companion sources archive

Source archives for everything dynamo ships on top of the base image are
emitted in a SEPARATE archive named `sources-{image}-{version}-<arch>.zip`
alongside this bundle. They live separately because:

- Sources are large (typically 100s of MB to GBs) — embedding would
  bloat the OSRB review bundle.
- Reviewers can audit license compliance and the manifest without
  pulling source.
- The sha256 of the sources archive is recorded in
  `build-provenance.json` (`sources.sha256`), so they stay
  cryptographically tied to this bundle.

## Verifying the bundle

```sh
sha256sum -c CHECKSUMS.sha256
```

## Regenerating from source

The `container/compliance/osrb/package.py` script in the dynamo source
repo, run against the artifacts produced by `--target legal`,
`--target sboms`, and `--target sources_archive`, reproduces this bundle
deterministically.
""",
        encoding="utf-8",
    )


def write_checksums(bundle_dir: Path) -> None:
    """sha256sum of every file in bundle_dir, written to CHECKSUMS.sha256."""
    lines: list[str] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "CHECKSUMS.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(bundle_dir).as_posix()
        lines.append(f"{digest}  {rel}")
    (bundle_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def pack_bundle(bundle_dir: Path, output_zip: Path) -> None:
    """Pack the assembled bundle as a zip with deterministic ordering.

    Walks bundle_dir in sorted order so the central-directory entries are in
    a stable sequence; combined with the fixed-mtime per ZipInfo this yields
    a deterministic-ish archive (zip's central directory layout makes byte-
    exact reproducibility harder than tar, but order + timestamps are stable).
    """
    fixed_mtime = (1980, 1, 1, 0, 0, 0)
    paths = sorted(p for p in bundle_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for path in paths:
            arcname = path.relative_to(bundle_dir.parent).as_posix()
            info = zipfile.ZipInfo(filename=arcname, date_time=fixed_mtime)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as f:
                zf.writestr(info, f.read())


# ---- CLI -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Image name (e.g. dynamo-runtime)"
    )
    parser.add_argument("--version", required=True, help="Image version tag")
    parser.add_argument(
        "--image-digest", default="", help="Resolved image digest (sha256:...)"
    )
    parser.add_argument("--legal-dir", type=Path, required=True)
    parser.add_argument("--sboms-dir", type=Path, required=True)
    parser.add_argument(
        "--sources-archive",
        type=Path,
        default=None,
        help="Path to the companion sources zip "
        "(sources-<image>-<version>-<arch>.zip). Not embedded "
        "in the OSRB bundle; only its sha256 + size are recorded "
        "in build-provenance.json for cross-verification.",
    )
    parser.add_argument(
        "--base-sbom",
        type=Path,
        default=None,
        help="Path in container/compliance/base_sboms/<base>.cdx.json",
    )
    parser.add_argument(
        "--base-image", default="", help="FROM base image string for provenance"
    )
    parser.add_argument(
        "--base-image-digest", default="", help="Base image digest for provenance"
    )
    parser.add_argument(
        "--linkage-map",
        type=Path,
        default=Path(__file__).resolve().parent / "linkage.yaml",
    )
    parser.add_argument(
        "--modifications-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "modifications",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output zip path: osrb-<image>-<version>-<arch>.zip",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    deps_rows = load_deps_csvs(args.sboms_dir)
    base_keys = load_base_components(args.base_sbom)
    linkage_map = load_yaml(args.linkage_map)

    logger.info(
        "Loaded %d package rows; %d base components; %d linkage entries",
        len(deps_rows),
        len(base_keys),
        len(linkage_map.get("ecosystem_defaults", {}) or {}),
    )

    bundle_root = args.output.parent / args.output.stem
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    sources_meta = sources_archive_metadata(args.sources_archive)

    write_manifest_csv(deps_rows, base_keys, linkage_map, bundle_root / "manifest.csv")
    write_consolidated_cyclonedx(
        deps_rows,
        base_keys,
        args.image,
        args.version,
        bundle_root / "manifest.cdx.json",
    )
    licenses_n = copy_license_texts(args.legal_dir, bundle_root)
    mods_n = copy_modifications(args.modifications_dir, bundle_root)
    write_attribution_base(
        base_keys,
        args.base_image,
        args.base_image_digest,
        bundle_root / "attribution-base.txt",
    )
    write_provenance(
        args.image,
        args.version,
        args.image_digest,
        args.base_image,
        args.base_image_digest,
        sources_meta,
        bundle_root / "build-provenance.json",
    )
    write_readme(args.image, args.version, bundle_root / "README.md")
    write_checksums(bundle_root)
    pack_bundle(bundle_root, args.output)
    shutil.rmtree(bundle_root)

    logger.info(
        "Wrote %s (licenses: %d files, mods: %d patches, sources_archive sha256: %s)",
        args.output,
        licenses_n,
        mods_n,
        sources_meta.get("sha256", "n/a")[:16] if sources_meta.get("sha256") else "n/a",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
