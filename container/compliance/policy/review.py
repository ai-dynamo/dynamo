#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify SPDX alternatives and guard policy changes without approving legacy debt."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from ..overrides import UniqueKeyLoader, exact_version, load_records
from .validate import load_policy, parse_spdx

# Conservative, explicit set. Unlisted and LicenseRef terms require review;
# an existing global allowance is not evidence that a license is permissive.
PERMISSIVE = frozenset(
    {
        "0BSD",
        "Apache-1.1",
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "ISC",
        "Zlib",
        "BSD-1-Clause",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSD-4-Clause",
        "BSD-4-Clause-UC",
        "BSL-1.0",
        "CC0-1.0",
        "Unlicense",
        "WTFPL",
        "Python-2.0",
        "PSF-2.0",
        "Unicode-3.0",
        "Unicode-DFS-2016",
        "Unicode-DFS-2015",
        "ICU",
        "X11",
        "bzip2-1.0.6",
        "curl",
        "FSFAP",
        "FSFUL",
        "FSFULLR",
        "HPND",
        "HPND-sell-variant",
        "NCSA",
        "ZPL-2.1",
        "BlueOak-1.0.0",
        "PostgreSQL",
        "OpenSSL",
        "TCL",
        "Info-ZIP",
    }
)
_RECIPROCAL = re.compile(
    r"^(?:A?GPL|LGPL|MPL|EPL|CPL|CDDL|EUPL|OSL|RPL|QPL|GFDL|CC-BY-SA|CECILL)(?:-|$)"
)


def _paths(node) -> set[str]:
    kind = node[0]
    if kind == "lic":
        return {
            "permissive"
            if node[1] in PERMISSIVE
            else "reciprocal"
            if _RECIPROCAL.match(node[1])
            else "review_required"
        }
    if kind == "with":
        return _paths(node[1]) if node[2] == "LLVM-exception" else {"review_required"}
    left, right = _paths(node[1]), _paths(node[2])
    if kind == "or":
        return left | right
    if kind == "and":
        order = {"permissive": 0, "reciprocal": 1, "review_required": 2}
        return {max((a, b), key=order.__getitem__) for a in left for b in right}
    raise ValueError(f"Unknown SPDX node {kind}")


def classify(expression: str) -> str:
    try:
        paths = _paths(parse_spdx(expression))
    except ValueError:
        return "review_required"
    if "permissive" in paths:
        return "permissive" if len(paths) == 1 else "permissive_alternative"
    return "review_required" if "review_required" in paths else "reciprocal"


def scoped_fields(row: dict) -> list[str]:
    missing = [] if exact_version(row.get("version")) else ["exact version"]
    for field in ("usage", "evidence", "reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            missing.append(field)
    return missing


def _exception_grant(row: dict) -> str:
    """Permission identity; factual metadata corrections do not create new grants."""
    return json.dumps(
        {
            "type": row.get("type"),
            "name": row.get("name"),
            "version": row.get("version"),
            "allow": sorted(row.get("allow", [])),
        },
        sort_keys=True,
    )


def review_policy(current: Path, base: Path | None = None) -> dict:
    policy = load_policy(current)
    debt = {
        "global_nonpermissive_or_unclassified": [
            {
                "license": value,
                "classification": classify(value),
                "also_denied": value in policy.deny,
            }
            for value in sorted(policy.allow)
            if classify(value) != "permissive"
        ],
        "unscoped_or_undocumented_exceptions": [
            row for row in policy.exceptions if scoped_fields(row)
        ],
        "status": "legacy_review_debt_not_approved",
    }
    errors = []
    if base is not None:
        prior = load_policy(base)
        for license_id in sorted(policy.allow - prior.allow):
            if classify(license_id) != "permissive":
                errors.append(
                    f"New global grant {license_id!r} requires a package/version/use exception"
                )
        if prior.deny - policy.deny:
            errors.append(
                f"Removed denied licenses: {sorted(prior.deny - policy.deny)}"
            )
        for field, value, old in (
            ("unknown", policy.unknown_action, prior.unknown_action),
            ("copyleft", policy.copyleft_action, prior.copyleft_action),
        ):
            if value != old and value != "deny":
                errors.append(
                    f"Relaxed {field}; new unknown/reciprocal terms must fail closed"
                )
        old_rows = {_exception_grant(row) for row in prior.exceptions}
        for row in policy.exceptions:
            if _exception_grant(row) not in old_rows:
                missing = scoped_fields(row)
                if row.get("type") not in {"rust", "python", "go", "dpkg", "native"}:
                    missing.append("valid ecosystem type")
                if not isinstance(row.get("name"), str) or not row["name"].strip():
                    missing.append("package name")
                allowed = row.get("allow")
                if (
                    not isinstance(allowed, list)
                    or not allowed
                    or any(
                        not isinstance(value, str) or parse_spdx(value)[0] != "lic"
                        for value in allowed
                    )
                ):
                    missing.append("nonempty SPDX identifier allow list")
                if missing:
                    errors.append(
                        f"Exception {row.get('type')}/{row.get('name')} requires {', '.join(missing)}"
                    )
    return {"legacy_review_debt": debt, "errors": errors}


def review_overrides(current: Path, base: Path | None = None) -> dict:
    records = load_records(current)
    errors = []
    if base is not None:
        # Historical identical duplicate list entries are tolerated only in the
        # base comparison. The edited catalog always goes through strict loading.
        old = yaml.load(base.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)[
            "overrides"
        ]
        old_grants = {
            (
                r["ecosystem"],
                r["name"],
                r.get("version"),
                r["license"],
                r.get("copyright_sha256"),
                tuple(sorted(r.get("absent_paths", []))),
            )
            for r in old
        }
        current_keys = {row.key for row in records}
        for previous in old:
            key = (previous["ecosystem"], previous["name"], previous.get("version"))
            if (
                (previous.get("copyright_sha256") or previous.get("absent_paths"))
                and key not in current_keys
                and (key[0], key[1], None) in current_keys
            ):
                errors.append(
                    f"Removing scoped override {key} exposes a broader versionless fallback"
                )
        for row in records:
            if (
                *row.key,
                row.license,
                row.copyright_sha256,
                tuple(sorted(row.absent_paths)),
            ) not in old_grants:
                missing = scoped_fields(asdict(row))
                if row.ecosystem == "dpkg" and row.copyright_sha256 is None:
                    missing.append(
                        "reviewed copyright_sha256 (predicate removal broadens scope)"
                    )
                for previous in old:
                    if (
                        previous["ecosystem"],
                        previous["name"],
                        previous.get("version"),
                    ) == row.key and set(previous.get("absent_paths", [])) - set(
                        row.absent_paths
                    ):
                        missing.append(
                            "preserved absent_paths (predicate removal broadens scope)"
                        )
                        break
                if missing:
                    errors.append(
                        f"Override {row.ecosystem}/{row.name} requires {', '.join(missing)}"
                    )
    return {
        "legacy_unversioned_overrides": sum(r.version is None for r in records),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--base-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "policy": review_policy(args.policy, args.base_policy),
        "overrides": review_overrides(args.overrides, args.base_overrides),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    errors = report["policy"]["errors"] + report["overrides"]["errors"]
    for error in errors:
        print(error)
    print(
        f"Policy review: {len(errors)} new-change violation(s); legacy debt is reported, not approved"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
