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
import re
import sys
import tomllib
from pathlib import Path

import yaml

# Frameworks where NIXL_REF (build) drives the runtime libnixl.so the
# container ships. Sglang is intentionally excluded: its runtime image
# inherits NIXL from the upstream `lmsysorg/sglang` base, and its `nixl_ref`
# is consumed only by the dev stage to provide the C++ SDK for `nixl-sys`
# Rust links (see container/context.yaml comment block on the sglang section).
RUNTIME_NIXL_FRAMEWORKS = ("vllm", "trtllm")


def major_minor(version: str) -> tuple[int, int] | None:
    m = re.match(r"v?(\d+)\.(\d+)", version)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_nixl_pin(extras_list: list[str]) -> tuple[str, list[tuple[int, int]]] | None:
    """Return (raw_dep_string, [major.minor bounds]) for the first nixl[cuXX] pin.

    Bounds are extracted from operators like `>=0.10.1`, `<=0.10.1`, `<0.11.0`.
    Returns None if no nixl pin found.
    """
    for dep in extras_list:
        if not dep.lower().startswith("nixl["):
            continue
        bounds = [
            (int(a), int(b)) for a, b in re.findall(r"[<>=]+\s*v?(\d+)\.(\d+)", dep)
        ]
        return dep, bounds
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
        ref_mm = major_minor(ref) if ref else None
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

        dep_str, bounds = pin
        if not bounds:
            errors.append(
                f"{fw}: pyproject nixl pin {dep_str!r} has no parseable version bound"
            )
            continue
        if ref_mm not in bounds and not any(b == ref_mm for b in bounds):
            # major.minor of nixl_ref must appear in the pin's listed bounds.
            # This is a deliberately loose check: patch-level drift is OK,
            # but an X.Y bump on either side that the other doesn't follow
            # gets caught.
            errors.append(
                f"{fw}: nixl_ref={ref} (major.minor={ref_mm}) does not match any bound "
                f"in pyproject pin {dep_str!r} (bounds={bounds})"
            )

    if errors:
        print("nixl alignment check FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nFix by aligning major.minor versions of `nixl_ref` in container/context.yaml "
            "with `nixl[cuXX]` in pyproject.toml [project.optional-dependencies] for each framework.",
            file=sys.stderr,
        )
        return 1

    print(f"nixl alignment check OK ({', '.join(RUNTIME_NIXL_FRAMEWORKS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
