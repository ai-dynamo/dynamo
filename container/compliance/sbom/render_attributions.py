# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render per-ecosystem ATTRIBUTIONS-container-*.{md,csv} files from a CycloneDX BOM.

The ``container-`` prefix distinguishes these comprehensive in-container files
(everything syft sees plus source-compiled binaries) from the repo-root
``ATTRIBUTIONS-*.md`` files (direct dependencies only) produced by
``container/compliance/generate_root_attributions.py``.

Ecosystems are detected by PURL scheme:

  pkg:cargo/    -> ATTRIBUTIONS-container-Rust.{md,csv}
  pkg:pypi/     -> ATTRIBUTIONS-container-Python.{md,csv}
  pkg:golang/   -> ATTRIBUTIONS-container-Go.{md,csv}
  pkg:deb/      -> ATTRIBUTIONS-container-Debian.{md,csv}
  pkg:github/   -> ATTRIBUTIONS-container-Binary.{md,csv}   (source-compiled)
  pkg:npm/      -> ATTRIBUTIONS-container-npm.{md,csv}
  pkg:nuget/    -> ATTRIBUTIONS-container-nuget.{md,csv}
  any other     -> ATTRIBUTIONS-container-Other.{md,csv}    (unknown PURL scheme)

In Markdown output, license text (when present on a component) is rendered
inline inside a fenced code block so the file is self-contained. Components
without license text fall back to an SPDX expression; unknown licenses are
listed under a ``## Unresolved`` section so they are visible to reviewers
rather than silently dropped.

In CSV output (``--format csv`` or ``--format both``), each ecosystem gets one
row per component with the columns listed in :data:`CSV_COLUMNS`. Verbatim
license texts are written once per unique SPDX ID into ``LICENSES/<id>.txt``,
and ``LICENSE-MANIFEST.csv`` maps each ID to the source component and file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# Regex to extract copyright lines from license text
_COPYRIGHT_RE = re.compile(r"(?m)^[ \t>*#/-]*Copyright\b[^\n]*\b(?:19|20)\d{2}\b[^\n]*")

ECOSYSTEMS: list[tuple[str, str, str]] = [
    ("pkg:cargo/", "Rust", "ATTRIBUTIONS-container-Rust"),
    ("pkg:pypi/", "Python", "ATTRIBUTIONS-container-Python"),
    ("pkg:golang/", "Go", "ATTRIBUTIONS-container-Go"),
    ("pkg:deb/", "Debian", "ATTRIBUTIONS-container-Debian"),
    ("pkg:github/", "Source-Compiled Binary", "ATTRIBUTIONS-container-Binary"),
    ("pkg:npm/", "npm", "ATTRIBUTIONS-container-npm"),
    ("pkg:nuget/", "NuGet", "ATTRIBUTIONS-container-nuget"),
]

# CSV column schema for attribution exports
CSV_COLUMNS = [
    "name",
    "version",
    "purl",
    "ecosystem",
    "license_spdx",
    "license_source",
    "copyright",
    "source_url",
    "containers",
    "bom_ref",
]


def _license_spdx(comp: dict[str, Any]) -> str:
    """Extract SPDX license expression from component.

    Per CycloneDX 1.6 §5.4.13, multiple `licenses[]` entries represent
    disjunctive (OR) alternatives, not conjunctive (AND) requirements.
    When a single entry already has an `expression`, return it verbatim.
    """
    entries = comp.get("licenses") or []
    ids: list[str] = []
    for entry in entries:
        # Check for top-level expression first (entry-level, not nested)
        expr = entry.get("expression")
        if expr:
            ids.append(str(expr))
            continue
        # Fall back to license.id or license.name
        lic = entry.get("license") or {}
        for key in ("id", "name"):
            val = lic.get(key)
            if val:
                ids.append(str(val))
                break
    # Single expression: return verbatim (no wrapping)
    if len(ids) == 1:
        return ids[0]
    # Multiple entries: OR semantics per CycloneDX spec
    return " OR ".join(ids) if ids else "NOASSERTION"


def _license_text(comp: dict[str, Any]) -> str | None:
    for entry in comp.get("licenses") or []:
        lic = entry.get("license") or {}
        text_obj = lic.get("text") or {}
        content = text_obj.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _extract_copyright(comp: dict[str, Any]) -> str:
    """Extract copyright notice from license text or component metadata."""
    # First try extracting from license text
    text = _license_text(comp)
    if text:
        matches = _COPYRIGHT_RE.findall(text)
        if matches:
            # Return first copyright line, stripped of leading decoration
            return matches[0].strip().lstrip(">*#/- \t")
    # Fall back to component.copyright field if present
    copyright_field = comp.get("copyright")
    if isinstance(copyright_field, str) and copyright_field.strip():
        return copyright_field.strip()
    return ""


def _containers(comp: dict[str, Any]) -> list[str]:
    for prop in comp.get("properties") or []:
        if prop.get("name") == "dynamo:containers":
            val = prop.get("value") or ""
            return sorted(v for v in val.split(",") if v)
    return []


def _sort_key(comp: dict[str, Any]) -> tuple[str, str]:
    return (comp.get("name", "").lower(), comp.get("version", ""))


def _component_url(comp: dict[str, Any]) -> str | None:
    for ref in comp.get("externalReferences") or []:
        if ref.get("type") in {"vcs", "website", "distribution"}:
            url = ref.get("url")
            if url:
                return url
    purl = comp.get("purl") or ""
    match = re.match(r"pkg:(\w+)/([^@?#]+)", purl)
    if not match:
        return None
    scheme, path = match.group(1), match.group(2)
    mapping = {
        "cargo": f"https://crates.io/crates/{path.split('/')[-1]}",
        "pypi": f"https://pypi.org/project/{path.split('/')[-1]}/",
        "golang": f"https://pkg.go.dev/{path}",
        "github": f"https://github.com/{path}",
    }
    return mapping.get(scheme)


def _code_fence(text: str) -> str:
    """Return a code fence delimiter long enough to safely wrap ``text``.

    If the text contains runs of backticks, the fence must be at least one
    backtick longer than the longest run found in the text.
    """
    max_run = 0
    current_run = 0
    for ch in text:
        if ch == "`":
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return "`" * max(3, max_run + 1)


def _render_section(comp: dict[str, Any]) -> str:
    name = comp.get("name", "<unknown>")
    version = comp.get("version", "")
    purl = comp.get("purl") or ""
    spdx = _license_spdx(comp)
    text = _license_text(comp)
    containers = _containers(comp)
    url = _component_url(comp)

    header = f"## {name} {version}".rstrip()
    lines = [header, ""]
    lines.append(f"- PURL: `{purl}`")
    if url:
        lines.append(f"- Source: {url}")
    lines.append(f"- License: `{spdx}`")
    if containers:
        lines.append(f"- Containers: {', '.join(containers)}")
    lines.append("")

    if text:
        fence = _code_fence(text)
        lines.append("<details><summary>License text</summary>")
        lines.append("")
        lines.append(fence)
        lines.append(text.rstrip())
        lines.append(fence)
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def _render_file(
    release: str,
    ecosystem: str,
    components: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> str:
    header = [
        f"# ATTRIBUTIONS — {ecosystem}",
        "",
        f"Release: **{release}**",
        "",
        "Generated by `container/compliance/sbom/render_attributions.py` from the",
        "per-container CycloneDX BOM (`generate_sbom.sh` + `render_attributions.py`).",
        "",
        f"- Components: {len(components) + len(unresolved)}",
        f"- Unresolved licenses: {len(unresolved)}",
        "",
        "---",
        "",
    ]
    body = [_render_section(c) for c in components]
    if unresolved:
        body.append("\n## Unresolved\n")
        body.append(
            "Components below did not expose an SPDX license in the BOM. They "
            "require manual triage before the attribution file is published.\n"
        )
        for c in unresolved:
            body.append(_render_section(c))
    # Blank line between header and body
    return "\n".join(header) + "\n" + "\n".join(body)


def _select(
    components: Iterable[dict[str, Any]], purl_prefix: str
) -> list[dict[str, Any]]:
    """Select components matching prefix, de-duplicated by PURL (last-wins)."""
    by_purl: dict[str, dict[str, Any]] = {}
    for c in components:
        purl = c.get("purl") or ""
        if purl.startswith(purl_prefix):
            by_purl[purl] = c  # last-wins
    selected = list(by_purl.values())
    selected.sort(key=_sort_key)
    return selected


def _safe_spdx_filename(spdx_id: str) -> str:
    """Convert SPDX ID to safe filename (no slashes, spaces)."""
    return re.sub(r"[^\w\-.]", "_", spdx_id)


def _write_csv(
    path: Path,
    ecosystem: str,
    components: list[dict[str, Any]],
) -> None:
    """Write a per-ecosystem CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for comp in components:
            text = _license_text(comp)
            writer.writerow(
                {
                    "name": comp.get("name", ""),
                    "version": comp.get("version", ""),
                    "purl": comp.get("purl", ""),
                    "ecosystem": ecosystem,
                    "license_spdx": _license_spdx(comp),
                    "license_source": "text" if text else "spdx",
                    "copyright": _extract_copyright(comp),
                    "source_url": _component_url(comp) or "",
                    "containers": ",".join(_containers(comp)),
                    "bom_ref": comp.get("bom-ref", ""),
                }
            )


def _collect_license_corpus(
    components: list[dict[str, Any]],
    licenses_dir: Path,
) -> list[dict[str, str]]:
    """Collect verbatim license texts into LICENSES/<spdx-id>.txt.

    Returns manifest rows: [{spdx_id, source_component, char_count, path}, ...]
    """
    licenses_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    seen_spdx: set[str] = set()

    for comp in components:
        spdx = _license_spdx(comp)
        if spdx == "NOASSERTION" or spdx in seen_spdx:
            continue
        text = _license_text(comp)
        if not text:
            continue
        # Use first component with text as canonical source
        safe_name = _safe_spdx_filename(spdx)
        out_file = licenses_dir / f"{safe_name}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        manifest.append(
            {
                "spdx_id": spdx,
                "source_component": comp.get("name", ""),
                "char_count": str(len(text)),
                "path": f"LICENSES/{safe_name}.txt",
            }
        )
        seen_spdx.add(spdx)

    return manifest


def _write_license_manifest(path: Path, manifest: list[dict[str, str]]) -> None:
    """Write LICENSE-MANIFEST.csv mapping SPDX IDs to license files."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["spdx_id", "source_component", "char_count", "path"]
        )
        writer.writeheader()
        for row in sorted(manifest, key=lambda r: r["spdx_id"]):
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bom",
        required=True,
        type=Path,
        help="Path to per-container CycloneDX BOM",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Release identifier written into each ATTRIBUTIONS file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write ATTRIBUTIONS-* files",
    )
    parser.add_argument(
        "--format",
        choices=["md", "csv", "both"],
        default="both",
        help="Output format: md (Markdown only), csv (CSV only), both (default)",
    )
    args = parser.parse_args()

    with open(args.bom, "r", encoding="utf-8") as f:
        bom = json.load(f)
    all_components = bom.get("components") or []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # De-duplicate all components by PURL upfront
    by_purl: dict[str, dict[str, Any]] = {}
    for c in all_components:
        purl = c.get("purl") or ""
        if purl:
            by_purl[purl] = c  # last-wins
    all_components = list(by_purl.values())

    # Track which PURLs are assigned to known ecosystems
    assigned_purls: set[str] = set()
    unknown_schemes: set[str] = set()

    # Collect all license texts for LICENSES corpus
    licenses_dir = args.output_dir / "LICENSES"
    all_processed: list[dict[str, Any]] = []

    for prefix, ecosystem, basename in ECOSYSTEMS:
        comps = _select(all_components, prefix)
        for c in comps:
            purl = c.get("purl") or ""
            assigned_purls.add(purl)
        resolved = [c for c in comps if _license_spdx(c) != "NOASSERTION"]
        unresolved = [c for c in comps if _license_spdx(c) == "NOASSERTION"]
        all_processed.extend(resolved + unresolved)

        if args.format in ("md", "both"):
            content = _render_file(args.release, ecosystem, resolved, unresolved)
            out_path = args.output_dir / f"{basename}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(
                f"wrote {out_path} ({len(resolved)} resolved, {len(unresolved)} unresolved)",
                file=sys.stderr,
            )

        if args.format in ("csv", "both"):
            csv_path = args.output_dir / f"{basename}.csv"
            _write_csv(csv_path, ecosystem, resolved + unresolved)
            print(
                f"wrote {csv_path} ({len(resolved) + len(unresolved)} rows)",
                file=sys.stderr,
            )

    # Handle "Other" ecosystem for unmatched PURL prefixes
    other_comps: list[dict[str, Any]] = []
    for c in all_components:
        purl = c.get("purl") or ""
        if purl and purl not in assigned_purls:
            other_comps.append(c)
            # Extract scheme for warning
            match = re.match(r"pkg:([^/]+)/", purl)
            if match:
                unknown_schemes.add(match.group(1))

    if other_comps:
        other_comps.sort(key=_sort_key)
        resolved = [c for c in other_comps if _license_spdx(c) != "NOASSERTION"]
        unresolved = [c for c in other_comps if _license_spdx(c) == "NOASSERTION"]
        all_processed.extend(resolved + unresolved)

        if unknown_schemes:
            print(
                f"warning: {len(other_comps)} components with unknown PURL schemes: "
                f"{', '.join(sorted(unknown_schemes))}",
                file=sys.stderr,
            )

        if args.format in ("md", "both"):
            content = _render_file(args.release, "Other", resolved, unresolved)
            out_path = args.output_dir / "ATTRIBUTIONS-container-Other.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(
                f"wrote {out_path} ({len(resolved)} resolved, {len(unresolved)} unresolved)",
                file=sys.stderr,
            )

        if args.format in ("csv", "both"):
            csv_path = args.output_dir / "ATTRIBUTIONS-container-Other.csv"
            _write_csv(csv_path, "Other", resolved + unresolved)
            print(
                f"wrote {csv_path} ({len(resolved) + len(unresolved)} rows)",
                file=sys.stderr,
            )

    # Write LICENSES corpus and manifest
    manifest = _collect_license_corpus(all_processed, licenses_dir)
    if manifest:
        manifest_path = args.output_dir / "LICENSE-MANIFEST.csv"
        _write_license_manifest(manifest_path, manifest)
        print(
            f"wrote {len(manifest)} license files to {licenses_dir}",
            file=sys.stderr,
        )
        print(f"wrote {manifest_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
