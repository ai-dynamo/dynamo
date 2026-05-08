#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check NIXL version alignment between container/context.yaml and pyproject.toml.

Each framework that builds NIXL into its container (`nixl_ref` in
context.yaml) AND pins a runtime NIXL version (`nixl[cuXX]` in pyproject.toml
optional-dependencies) must keep those two values in major.minor agreement.
Drift between them produces an ABI mismatch in the runtime image: the C++
NIXL built into /opt/nvidia/nvda_nixl/ is one version, the Python nixl-cu*
wheel installed via pyproject extras is another, and they share `libnixl.so`
SONAME with no version handshake. Symptom: runtime crash at NIXL agent init
with "backend 'UCX' not found" (https://github.com/ai-dynamo/dynamo/issues/6671).

Run from repo root: python3 container/check_nixl_alignment.py
Exit 0 if aligned, 1 otherwise.
"""
import sys
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

# Frameworks where NIXL_REF (build) drives the runtime libnixl.so the
# container ships. Sglang is intentionally excluded: its runtime image
# inherits NIXL from the upstream `lmsysorg/sglang` base, and its `nixl_ref`
# is consumed only by the dev stage to provide the C++ SDK for `nixl-sys`
# Rust links (see container/context.yaml comment block on the sglang section).
RUNTIME_NIXL_FRAMEWORKS = ("vllm", "trtllm")


def parse_ref_version(ref: str) -> Version | None:
    """nixl_ref is a git ref (tag/SHA). Strip a leading `v` and parse as PEP 440."""
    try:
        return Version(ref.lstrip("v"))
    except InvalidVersion:
        return None


def parse_nixl_pin(extras_list: list[str]) -> tuple[str, Requirement] | None:
    """Return (raw_dep_string, Requirement) for the first nixl[cuXX] pin, else None."""
    for dep in extras_list:
        if dep.lower().startswith("nixl["):
            return dep, Requirement(dep)
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    context_yaml = yaml.safe_load(
        (repo_root / "container" / "context.yaml").read_text()
    )
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    errors: list[str] = []
    for fw in RUNTIME_NIXL_FRAMEWORKS:
        ref = context_yaml.get(fw, {}).get("nixl_ref")
        pin = parse_nixl_pin(extras.get(fw, []))

        if ref is None and pin is None:
            continue
        if ref is None:
            errors.append(
                f"{fw}: pyproject pins nixl ({pin[0]}) but no nixl_ref in context.yaml"
            )
            continue
        if pin is None:
            errors.append(
                f"{fw}: context.yaml builds nixl_ref={ref} "
                "but no nixl pin in pyproject [project.optional-dependencies]"
            )
            continue

        dep_str, requirement = pin
        ref_version = parse_ref_version(ref)
        if ref_version is None:
            errors.append(
                f"{fw}: nixl_ref={ref!r} cannot be parsed as a PEP 440 version "
                f"(SHA or non-version git ref?); cannot validate against pyproject pin {dep_str!r}"
            )
            continue
        # `Version in SpecifierSet` evaluates the full set of constraints
        # (e.g. `>=0.10.1,<0.11.0`) per PEP 440 semantics, so a constraint
        # like `>=0.10.1,<0.12.0` correctly accepts nixl_ref=0.11.3.
        if ref_version not in requirement.specifier:
            errors.append(
                f"{fw}: nixl_ref={ref} ({ref_version}) does not satisfy "
                f"pyproject pin {dep_str!r}"
            )

    if errors:
        print("nixl alignment check FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nFix by aligning `nixl_ref` in container/context.yaml with "
            "`nixl[cuXX]` in pyproject.toml [project.optional-dependencies] for each framework.",
            file=sys.stderr,
        )
        return 1

    print(f"nixl alignment check OK ({', '.join(RUNTIME_NIXL_FRAMEWORKS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
