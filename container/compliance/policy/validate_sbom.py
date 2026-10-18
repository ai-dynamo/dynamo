# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate shipped components against the license policy.

This is the independent final-artifact counterpart to the inline ``licenses``
Dockerfile stage. The final-image Syft scan proves which packages are shipped;
the build's ``osrb-deps.csv`` supplies their reviewed, normalized SPDX data.
Syft's heuristic license metadata is deliberately not policy input: installed
package and Cargo-lock catalogers frequently emit UNKNOWN or whole-source-package
expressions that do not describe the shipped binary's reviewed attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from compliance.base_sboms.capture_baseline_sbom import compute_delta
from compliance.policy import validate as policy_validate


def _load_sbom(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(name: str) -> str:
    return (name or "").strip().lower().replace("_", "-").replace(" ", "-")


def _ecosystem(component: dict) -> str | None:
    purl = component.get("purl") or ""
    mappings = {
        "pkg:deb/": "dpkg",
        "pkg:pypi/": "python",
        "pkg:cargo/": "rust",
        "pkg:golang/": "go",
        "pkg:rpm/": "rpm",
    }
    return next(
        (value for prefix, value in mappings.items() if purl.startswith(prefix)), None
    )


def _component_key(component: dict) -> tuple[str, str, str] | None:
    ecosystem = _ecosystem(component)
    name = component.get("name")
    version = component.get("version")
    if ecosystem is None or not name or version is None:
        return None
    return ecosystem, _normalize(name), str(version)


def _attribution_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row.get("ecosystem", ""),
                _normalize(row.get("name", "")),
                row.get("version", ""),
            )
            rows[key] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance.policy.validate_sbom")
    parser.add_argument("--syft-sbom", type=Path, required=True)
    parser.add_argument("--base-sbom", type=Path, default=None)
    parser.add_argument("--our-csv", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args(argv)

    image_sbom = _load_sbom(args.syft_sbom)
    baseline_sbom: dict = {"components": []}
    if args.base_sbom:
        if not args.base_sbom.is_file():
            parser.error(f"baseline SBOM not found: {args.base_sbom}")
        baseline_sbom = _load_sbom(args.base_sbom)

    delta = compute_delta(image_sbom, baseline_sbom)
    shipped_keys = {key for component in delta if (key := _component_key(component))}
    attribution = _attribution_rows(args.our_csv)
    matched_keys = shipped_keys & attribution.keys()

    policy = policy_validate.load_policy(args.policy)
    violations: list[str] = []
    for key in sorted(matched_keys):
        row = attribution[key]
        violation = policy_validate.validate_row(
            policy,
            row["ecosystem"],
            row["name"],
            row["version"],
            row.get("spdx", "") or "UNKNOWN",
        )
        if violation is not None:
            violations.append(str(violation))
    violation_count = len(violations)
    title = f"Final-image license policy{' for ' + args.image if args.image else ''}"

    output = [
        title,
        f"  image components: {len(image_sbom.get('components', []))}",
        f"  components above baseline: {len(delta)}",
        f"  unique attributed components checked: {len(matched_keys)}",
        f"  unique components without exact attribution match: {len(shipped_keys - matched_keys)}",
        f"  POLICY VIOLATIONS: {violation_count}",
    ]
    output.extend(violations)
    rendered = "\n".join(output) + "\n"
    print(rendered, end="")

    if args.report:
        args.report.write_text(rendered, encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"### {title}\n\n")
            stream.write(f"- components above baseline: **{len(delta)}**\n")
            stream.write(f"- attributed components checked: **{len(matched_keys)}**\n")
            stream.write(
                "- components without exact attribution match: "
                f"**{len(shipped_keys - matched_keys)}**\n"
            )
            stream.write(f"- policy violations: **{violation_count}**\n\n")
            if violations:
                stream.write("```text\n")
                stream.write("\n".join(line.strip() for line in violations))
                stream.write("\n```\n")
            else:
                stream.write("✅ no final-image license-policy violations.\n")

    return 1 if violation_count and args.fail_on_findings else 0


if __name__ == "__main__":
    sys.exit(main())
